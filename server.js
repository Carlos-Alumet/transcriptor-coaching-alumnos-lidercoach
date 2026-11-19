// Transcriptor de sesiones de coaching (formato ICF)
// Flujo: subir audio -> responder al instante con un jobId -> procesar en segundo plano
// (ffmpeg normaliza y corta en silencios -> OpenAI transcribe con diarización por partes
// -> se unen las partes con tiempos reales -> se asignan roles COACH / CLIENTE).
// La página consulta /api/jobs/:id cada pocos segundos, así no hay timeouts del hosting.

import express from "express";
import formidable from "formidable";
import ffmpegPath from "ffmpeg-static";
import { spawn } from "node:child_process";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import crypto from "node:crypto";

// ---------------- Configuración ----------------
const PORT = process.env.PORT || 3000;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_BASE_URL = (process.env.OPENAI_BASE_URL || "https://api.openai.com/v1").replace(/\/$/, "");
const MODEL = process.env.TRANSCRIBE_MODEL || "gpt-4o-transcribe-diarize";
const LANGUAGE = process.env.TRANSCRIBE_LANGUAGE ?? "es";
const CHUNK_SECONDS = Number(process.env.CHUNK_SECONDS || 300); // ~5 min por parte
const MAX_UPLOAD_MB = Number(process.env.MAX_UPLOAD_MB || 500);
const PAUSE_SPLIT_SECONDS = Number(process.env.PAUSE_SPLIT_SECONDS || 2);
const JOB_TTL_MS =2 * 60 * 60 * 1000; // los resultados se guardan 2 horas
const WORK_ROOT = path.join(os.tmpdir(), "transcriptor-icf");

if (!OPENAI_API_KEY) console.warn("⚠️  Falta la variable de entorno OPENAI_API_KEY");
fs.mkdirSync(WORK_ROOT, { recursive: true });

const app = express();
app.use(express.static("public"));
// Librerías para generar el PDF en el navegador (se sirven desde el propio servidor).
app.use("/vendor/jspdf", express.static("node_modules/jspdf/dist"));
app.use("/vendor/autotable", express.static("node_modules/jspdf-autotable/dist"));
app.get("/healthz", (_req, res) => res.send("ok"));

// ---------------- Trabajos en memoria ----------------
const jobs = new Map();
let queue = Promise.resolve(); // procesa de a un audio por vez (cuida la RAM del plan gratis)

setInterval(() => {
  const now = Date.now();
  for (const [id, job] of jobs) if (now - job.updatedAt > JOB_TTL_MS) jobs.delete(id);
}, 10 * 60 * 1000).unref();

const setJob = (job, patch) => Object.assign(job, patch, { updatedAt: Date.now() });
const first = (v) => (Array.isArray(v) ? v[0] : v);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const rmDir = (dir) => fsp.rm(dir, { recursive: true, force: true }).catch(() => {});

app.post("/api/jobs", (req, res) => {
  const jobId = crypto.randomUUID();
  const workDir = path.join(WORK_ROOT, jobId);
  fs.mkdirSync(workDir, { recursive: true });

  const form = formidable({
    uploadDir: workDir,
    keepExtensions: true,
    maxFiles: 1,
    maxFileSize: MAX_UPLOAD_MB * 1024 * 1024,
  });

  form.parse(req, (err, fields, files) => {
    if (err) {
      rmDir(workDir);
      const msg = err.httpCode === 413
        ? `El archivo supera los ${MAX_UPLOAD_MB} MB.`
        : "No se pudo recibir el archivo.";
      return res.status(400).json({ error: msg });
    }

    const file = first(files.audio) || first(Object.values(files)[0]);
    if (!file?.filepath || !file.size) {
      rmDir(workDir);
      return res.status(400).json({ error: "No llegó ningún archivo de audio." });
    }
    if (!OPENAI_API_KEY) {
      rmDir(workDir);
      return res.status(500).json({ error: "El servidor no tiene configurada la OPENAI_API_KEY." });
    }

    const coachName = String(first(fields.coach) || "").trim().slice(0, 80);
    const job = {
      id: jobId, status: "queued", progress: 0, message: "En cola…",
      fileName: file.originalFilename || "audio", createdAt: Date.now(), updatedAt: Date.now(),
    };
    jobs.set(jobId, job);
    res.status(202).json({ jobId });

    queue = queue.then(() =>
      processJob(job, file.filepath, workDir, coachName)
        .catch((e) => {
          console.error(`[${jobId}]`, e);
          setJob(job, { status: "error", message: "Error", error: e.message || "Error al procesar." });
        })
        .finally(() => rmDir(workDir))
    );
  });
});

app.get("/api/jobs/:id", (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: "Trabajo no encontrado (quizás el servidor se reinició)." });
  res.json(job);
});

// ---------------- Proceso principal ----------------
async function processJob(job, inputPath, workDir, coachName) {
  setJob(job, { status: "processing", progress: 2, message: "Preparando el audio…" });

  // 1) Normalizar: mono, 16 kHz, FLAC (sin pérdida, liviano). Acepta audio o video.
  const full = path.join(workDir, "full.flac");
  await ffmpeg(["-i", inputPath, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", full]);
  await fsp.rm(inputPath, { force: true });

  // 2) Detectar silencios para cortar entre frases, nunca en medio de una palabra.
  setJob(job, { progress: 5, message: "Buscando pausas para dividir el audio…" });
  const { duration, silences } = await analyzeAudio(full);
  if (!duration) throw new Error("No se pudo leer la duración del audio. ¿Es un archivo válido?");

  const cuts = planCuts(duration, silences);
  const chunks = [];
  for (let i = 0; i < cuts.length - 1; i++) {
    const file = path.join(workDir, `chunk_${i}.flac`);
    // Se agrega 1,5 s de silencio al final: evita que el modelo corte las últimas palabras.
    const trim = `atrim=start=${cuts[i].toFixed(3)}:end=${cuts[i + 1].toFixed(3)},asetpts=PTS-STARTPTS,apad=pad_dur=1.5`;
    await ffmpeg(["-i", full, "-af", trim, "-c:a", "flac", file]);
    chunks.push({ file, start: cuts[i] });
  }

  // 3) Transcribir cada parte. Después de la primera, se mandan muestras de voz
  //    de cada hablante para que las etiquetas sean las mismas en todas las partes.
  const refs = []; // { name, dataUrl }
  const warnings = [];
  const segments = [];
  let nextSpeaker = 1;

  for (let i = 0; i < chunks.length; i++) {
    setJob(job, {
      progress: 8 + Math.round((88 * i) / chunks.length),
      message: `Transcribiendo parte ${i + 1} de ${chunks.length}…`,
    });

    // En paralelo: diarización (quién y cuándo) + Whisper (palabras literales con muletillas).
    const [data, whisper] = await Promise.all([
      transcribeChunk(chunks[i].file, refs),
      whisperChunk(chunks[i].file).catch((e) => {
        console.warn(`Whisper falló en la parte ${i + 1}, se usa el texto de la diarización:`, e.message);
        warnings.push(`Entre ${hms(chunks[i].start)} y ${hms(cuts[i + 1])} pueden faltar muletillas (falló la pasada literal).`);
        return null;
      }),
    ]);
    const known = new Set(refs.map((r) => r.name));
    const localMap = new Map();
    const chunkSegs = [];

    for (const s of data.segments || []) {
      const text = (s.text || "").trim();
      if (!text) continue;
      const label = String(s.speaker ?? "?");
      let speaker = label;
      if (!known.has(label)) {
        if (!localMap.has(label)) localMap.set(label, `H${nextSpeaker++}`);
        speaker = localMap.get(label);
      }
      chunkSegs.push({
        speaker, text,
        localStart: Number(s.start || 0),
        localEnd: Number(s.end || 0),
      });
    }

    // Texto final: cada palabra de Whisper se asigna al hablante que la diarización
    // tiene sonando en ese instante. Si Whisper no está, se usa el texto de la diarización.
    const words = whisper ? punctuatedWords(whisper) : [];
    if (words.length && chunkSegs.length) {
      for (const w of words) {
        const seg = segmentAt(chunkSegs, (w.start + w.end) / 2);
        segments.push({ speaker: seg.speaker, text: w.word, start: chunks[i].start + w.start, end: chunks[i].start + w.end });
      }
    } else {
      for (const s of chunkSegs) {
        segments.push({ speaker: s.speaker, text: s.text, start: chunks[i].start + s.localStart, end: chunks[i].start + s.localEnd });
      }
    }

    // Guardar una muestra de voz (2-10 s) de cada hablante nuevo (máximo 4).
    for (const speaker of new Set(chunkSegs.map((s) => s.speaker))) {
      if (refs.length >= 4) break;
      if (refs.some((r) => r.name === speaker)) continue;
      const best = chunkSegs
        .filter((s) => s.speaker === speaker && s.localEnd - s.localStart >= 2.5)
        .sort((a, b) => (b.localEnd - b.localStart) - (a.localEnd - a.localStart))[0];
      if (!best) continue;
      const dataUrl = await makeVoiceSample(chunks[i].file, best.localStart, best.localEnd, workDir, speaker);
      refs.push({ name: speaker, dataUrl });
    }
  }

  if (!segments.length) throw new Error("No se detectó voz en el audio.");

  // Control de integridad: ¿la transcripción llega hasta el final del audio?
  const lastWord = Math.max(...segments.map((s) => s.end));
  const speechAfter = silences.length ? duration - Math.max(lastWord, lastSilenceStart(silences, duration)) : duration - lastWord;
  if (speechAfter > 10) {
    warnings.push(`La transcripción termina en ${hms(lastWord)} pero el audio dura ${hms(duration)}. Revisá si falta el final de la conversación.`);
  }

  // 4) Unir palabras consecutivas del mismo hablante. Si hay una pausa larga,
  //    se abre una nueva intervención (nueva fila con su propia marca de tiempo).
  segments.sort((a, b) => a.start - b.start);
  const turns = [];
  for (const s of segments) {
    const last = turns[turns.length - 1];
    if (last && last.speaker === s.speaker && s.start - last.end < PAUSE_SPLIT_SECONDS) {
      last.text += " " + s.text;
      last.end = s.end;
    } else {
      turns.push({ speaker: s.speaker, start: round(s.start), end: round(s.end), text: s.text });
    }
  }

  setJob(job, {
    status: "done", progress: 100, message: "Listo",
    result: { coachName, duration: round(duration), roles: assignRoles(turns), turns, warnings },
  });
}

// El coach es el que proporcionalmente hace más preguntas. Si queda al revés,
// la página tiene un botón para intercambiar COACH y CLIENTE.
function assignRoles(turns) {
  const stats = {};
  for (const t of turns) {
    const st = (stats[t.speaker] ||= { words: 0, questions: 0, firstAt: t.start });
    st.words += t.text.split(/\s+/).filter(Boolean).length;
    st.questions += (t.text.match(/\?/g) || []).length;
  }
  const ranked = Object.entries(stats).sort((a, b) => b[1].words - a[1].words);
  const roles = Object.fromEntries(ranked.map(([sp]) => [sp, "OTRO"]));
  if (ranked.length === 1) {
    roles[ranked[0][0]] = "COACH";
  } else if (ranked.length >= 2) {
    const [a, b] = ranked;
    const ratio = ([, s]) => s.questions / Math.max(1, s.words);
    let coach = ratio(a) === ratio(b) ? (a[1].firstAt <= b[1].firstAt ? a : b) : ratio(a) > ratio(b) ? a : b;
    roles[coach[0]] = "COACH";
    roles[(coach === a ? b : a)[0]] = "CLIENTE";
  }
  return roles;
}

// ---------------- Unión de palabras y hablantes ----------------

// Whisper devuelve palabras con tiempo pero sin puntuación, y segmentos con puntuación.
// Se recorre en orden para recuperar mayúsculas, comas, signos de pregunta, etc.
function punctuatedWords(w) {
  const words = (w.words || [])
    .map((x) => ({ word: String(x.word || "").trim(), start: Number(x.start), end: Number(x.end) }))
    .filter((x) => x.word);
  const tokens = (w.segments || []).flatMap((s) => String(s.text || "").trim().split(/\s+/)).filter(Boolean);
  let j = 0;
  for (const wd of words) {
    const target = norm(wd.word);
    if (!target) continue;
    for (let k = j; k < Math.min(tokens.length, j + 5); k++) {
      if (norm(tokens[k]) === target) {
        wd.word = tokens[k];
        j = k + 1;
        break;
      }
    }
  }
  return words;
}

const norm = (s) =>
  s.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/[^\p{L}\p{N}]/gu, "");

function segmentAt(segs, t) {
  let best = segs[0], bestDist = Infinity;
  for (const s of segs) {
    if (t >= s.localStart && t <= s.localEnd) return s;
    const d = t < s.localStart ? s.localStart - t : t - s.localEnd;
    if (d < bestDist) { bestDist = d; best = s; }
  }
  return best;
}

// ---------------- OpenAI ----------------
// Frase de ejemplo para que Whisper conserve las muletillas en vez de "limpiarlas".
const FILLER_PROMPT =
  process.env.WHISPER_PROMPT ||
  "Eh... mhm, sí, ajá. Bueno, eh, este... o sea, ¿viste? Mmm, claro, ehh, sí, sí. Ajá.";

async function whisperChunk(file) {
  const audio = await fsp.readFile(file);
  return callOpenAI(() => {
    const form = new FormData();
    form.append("file", new Blob([audio], { type: "audio/flac" }), path.basename(file));
    form.append("model", "whisper-1");
    form.append("response_format", "verbose_json");
    form.append("timestamp_granularities[]", "word");
    form.append("timestamp_granularities[]", "segment");
    form.append("temperature", "0");
    form.append("prompt", FILLER_PROMPT);
    if (LANGUAGE) form.append("language", LANGUAGE);
    return form;
  });
}

let sendLanguage = Boolean(LANGUAGE);

async function transcribeChunk(file, refs) {
  const audio = await fsp.readFile(file);
  const build = () => {
    const form = new FormData();
    form.append("file", new Blob([audio], { type: "audio/flac" }), path.basename(file));
    form.append("model", MODEL);
    form.append("response_format", "diarized_json");
    form.append("chunking_strategy", "auto");
    if (sendLanguage) form.append("language", LANGUAGE);
    for (const r of refs) {
      form.append("known_speaker_names[]", r.name);
      form.append("known_speaker_references[]", r.dataUrl);
    }
    return form;
  };
  try {
    return await callOpenAI(build);
  } catch (e) {
    if (e.status === 400 && sendLanguage && /language/i.test(e.body || "")) {
      sendLanguage = false; // el modelo no acepta "language": se reintenta sin ese campo
      return callOpenAI(build);
    }
    throw e;
  }
}

async function callOpenAI(buildForm) {
  let lastErr;
  for (let attempt = 1; attempt <= 5; attempt++) {
    const form = buildForm();
    let res, body;
    try {
      res = await fetch(`${OPENAI_BASE_URL}/audio/transcriptions`, {
        method: "POST",
        headers: { Authorization: `Bearer ${OPENAI_API_KEY}` },
        body: form,
        signal: AbortSignal.timeout(5 * 60 * 1000),
      });
      body = await res.text();
    } catch (e) {
      lastErr = new Error(`Falla de conexión con OpenAI: ${e.message}`);
      await sleep(backoff(attempt));
      continue;
    }

    if (res.ok) return JSON.parse(body);

    if (res.status === 401) throw new Error("La OPENAI_API_KEY no es válida.");
    if (/insufficient_quota/.test(body)) throw new Error("La cuenta de OpenAI no tiene saldo disponible.");

    lastErr = Object.assign(new Error(`OpenAI respondió ${res.status}: ${body.slice(0, 300)}`), {
      status: res.status,
      body,
    });
    if (res.status === 429 || res.status >= 500) {
      await sleep(backoff(attempt));
      continue;
    }
    throw lastErr;
  }
  throw lastErr;
}

const backoff = (attempt) => Math.min(30000, 2000 * 2 ** (attempt - 1));

// ---------------- ffmpeg ----------------
function ffmpeg(args) {
  return new Promise((resolve, reject) => {
    const p = spawn(ffmpegPath, ["-hide_banner", "-nostdin", "-y", ...args]);
    let log = "";
    p.stderr.on("data", (d) => (log += d.toString()));
    p.on("error", reject);
    p.on("close", (code) =>
      code === 0 ? resolve(log) : reject(new Error("No se pudo procesar el audio: " + log.slice(-400)))
    );
  });
}

async function analyzeAudio(file) {
  const log = await ffmpeg(["-i", file, "-af", "silencedetect=noise=-35dB:d=0.3", "-f", "null", "-"]);
  const d = log.match(/Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)/);
  const duration = d ? +d[1] * 3600 + +d[2] * 60 + parseFloat(d[3]) : 0;
  const silences = [];
  let start = null;
  for (const line of log.split("\n")) {
    const s = line.match(/silence_start:\s*(-?[\d.]+)/);
    if (s) start = Math.max(0, parseFloat(s[1]));
    const e = line.match(/silence_end:\s*([\d.]+)/);
    if (e && start !== null) {
      silences.push({ start, end: parseFloat(e[1]) });
      start = null;
    }
  }
  if (start !== null) silences.push({ start, end: duration }); // silencio que llega hasta el final
  return { duration, silences };
}

// Corta cerca de cada múltiplo de CHUNK_SECONDS, eligiendo la pausa más larga y cercana.
function planCuts(duration, silences) {
  const cuts = [0];
  const window = 60;
  let target = CHUNK_SECONDS;
  while (duration - target > CHUNK_SECONDS * 0.3) {
    let cut = target;
    let bestScore = -Infinity;
    for (const s of silences) {
      const mid = (s.start + s.end) / 2;
      if (Math.abs(mid - target) > window) continue;
      const score = (s.end - s.start) - Math.abs(mid - target) / window;
      if (score > bestScore) { bestScore = score; cut = mid; }
    }
    cuts.push(cut);
    target = cut + CHUNK_SECONDS;
  }
  cuts.push(duration);
  return cuts;
}

async function makeVoiceSample(chunkFile, start, end, workDir, name) {
  const len = Math.min(end - start, 8);
  const from = start + (end - start - len) / 2;
  const out = path.join(workDir, `voz_${name}.wav`);
  await ffmpeg(["-i", chunkFile, "-ss", from.toFixed(3), "-t", len.toFixed(3),
    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out]);
  return `data:audio/wav;base64,${(await fsp.readFile(out)).toString("base64")}`;
}

// Si el audio termina en silencio, devuelve dónde empieza ese silencio final.
function lastSilenceStart(silences, duration) {
  const last = silences[silences.length - 1];
  return last && duration - last.end < 1 ? last.start : duration;
}

const round = (n) => Math.round(n * 100) / 100;
const hms = (sec) => {
  sec = Math.floor(sec);
  return [Math.floor(sec / 3600), Math.floor((sec % 3600) / 60), sec % 60].map((n) => String(n).padStart(2, "0")).join(":");
};

app.listen(PORT, () => console.log(`Servidor corriendo en el puerto ${PORT}`));
