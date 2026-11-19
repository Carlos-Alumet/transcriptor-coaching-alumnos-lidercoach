import express from "express";
import OpenAI from "openai";
import { GoogleGenerativeAI } from "@google/generative-ai";
import formidable from "formidable";
import fs from "fs";
import ffmpeg from "fluent-ffmpeg";

const app = express();
const port = process.env.PORT || 3000;

app.use(express.static("public"));

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Función de compresión (solo se usa si el archivo supera los 25 MB y ffmpeg está disponible)
function compressAudio(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .toFormat("mp3")
      .audioChannels(1)
      .audioBitrate("32k")
      .on("end", () => resolve(outputPath))
      .on("error", (err) => reject(err))
      .save(outputPath);
  });
}

app.post("/api/transcribe", async (req, res) => {
  const uploadDir = "/tmp";
  if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
  }

  const form = formidable({ 
    uploadDir, 
    keepExtensions: true,
    maxFileSize: 100 * 1024 * 1024 
  });

  form.parse(req, async (err, fields, files) => {
    let filePath = null;
    let fileToProcess = null;

    try {
      if (err) {
        return res.status(400).json({ error: "Error al recibir el archivo de audio." });
      }

      const possibleKeys = Object.keys(files);
      if (possibleKeys.length === 0) {
        return res.status(400).json({ error: "No se encontró ningún archivo de audio." });
      }

      const audioFileArray = files[possibleKeys[0]];
      const audioFile = Array.isArray(audioFileArray) ? audioFileArray[0] : audioFileArray;
      
      if (!audioFile || !audioFile.filepath) {
        return res.status(400).json({ error: "El archivo de audio recibido es inválido." });
      }

      filePath = audioFile.filepath;
      fileToProcess = filePath;

      const fileSizeMB = audioFile.size / (1024 * 1024);
      console.log(`Tamaño del archivo recibido: ${fileSizeMB.toFixed(2)} MB`);

      // Si pesa más de 25 MB intentamos comprimir, si falla o pesa menos, lo usamos directo
      if (fileSizeMB > 25) {
        console.log("El archivo supera los 25 MB. Intentando comprimir...");
        const compressedPath = `${uploadDir}/compressed_${Date.now()}.mp3`;
        try {
          await compressAudio(filePath, compressedPath);
          fileToProcess = compressedPath;
        } catch (compError) {
          console.log("Advertencia: No se pudo comprimir (ffmpeg no disponible). Se enviará el archivo original.");
        }
      }

      // Transcribir con OpenAI Whisper
      console.log("Enviando a OpenAI Whisper...");
      const transcriptionResponse = await openai.audio.transcriptions.create({
        file: fs.createReadStream(fileToProcess),
        model: "whisper-1",
        response_format: "text"
      });

      const rawText = transcriptionResponse;

      // Dar formato con Gemini siguiendo las reglas de coaching
      console.log("Dando formato ontológico con Gemini...");
      const coachName = fields.coach ? (Array.isArray(fields.coach) ? fields.coach[0] : fields.coach) : "COACH";
      
      const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
      const prompt = `
        Actúa como un transcriptor profesional y analista de sesiones de coaching ontológico.
        A partir del siguiente texto transcrito por Whisper, estructúralo de forma íntegra aplicando estas reglas estrictas:
        1. Identifica y etiqueta a los hablantes estrictamente como "**COACH** (${coachName})" y "**CLIENTE**".
        2. Añade marcas de tiempo proporcionales y estimadas al inicio de cada intervención basadas en la lectura secuencial.
        3. Mantén las muletillas naturales ("eh", "mhm", "sí", "ajá") relevantes para la evaluación, eliminando repeticiones mecánicas de palabras cortadas.
        4. Presenta el resultado en párrafos claros y ordenados cronológicamente de principio a fin sin omitir fragmentos.
        
        Texto crudo a estructurar:
        ${rawText}
      `;

      const result = await model.generateContent(prompt);
      const formattedText = result.response.text();

      // Limpieza de archivos temporales
      if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath);

      return res.status(200).json({ success: true, transcription: formattedText });

    } catch (error) {
      console.error("Error en el servidor:", error);
      if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath);
      
      return res.status(500).json({ error: error.message || "Error al procesar la transcripción." });
    }
  });
});

app.listen(port, () => {
  console.log(`Servidor corriendo en el puerto ${port}`);
});
