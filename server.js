import express from "express";
import OpenAI from "openai";
import { GoogleGenerativeAI } from "@google/generative-ai";
import formidable from "formidable";
import fs from "fs";

const app = express();
const port = process.env.PORT || 3000;

app.use(express.static("public"));
// Ampliar el tiempo límite del servidor a 10 minutos para evitar cortes con audios largos
app.use((req, res, next) => {
  req.setTimeout(600000);
  res.setTimeout(600000);
  next();
});

const openai = new OpenAI({ 
  apiKey: process.env.OPENAI_API_KEY,
  timeout: 540000 // 9 minutos de timeout para la API de OpenAI
});

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

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
      const fileSizeMB = audioFile.size / (1024 * 1024);
      console.log(`Procesando archivo: ${fileSizeMB.toFixed(2)} MB`);

      if (fileSizeMB > 25) {
        if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath);
        return res.status(400).json({ error: "El archivo supera los 25 MB permitidos por OpenAI. Por favor comprimilo un poco antes de subirlo." });
      }

      // 1. Transcribir con OpenAI Whisper
      console.log("Enviando a OpenAI Whisper...");
      const transcriptionResponse = await openai.audio.transcriptions.create({
        file: fs.createReadStream(filePath),
        model: "whisper-1",
        response_format: "text"
      });

      const rawText = transcriptionResponse;

      // 2. Dar formato con Gemini
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
