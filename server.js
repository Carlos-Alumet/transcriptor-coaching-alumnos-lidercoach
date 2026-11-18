import express from "express";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { GoogleAIFileManager } from "@google/generative-ai/server";
import formidable from "formidable";
import fs from "fs";

const app = express();
const port = process.env.PORT || 3000;

app.use(express.static("public"));

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const fileManager = new GoogleAIFileManager(process.env.GEMINI_API_KEY);

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
        console.error("Error al recibir el archivo:", err);
        return res.status(400).json({ error: "Error al recibir el archivo de audio." });
      }

      const possibleKeys = Object.keys(files);
      if (possibleKeys.length === 0) {
        return res.status(400).json({ error: "No se encontró ningún archivo de audio en la petición." });
      }

      const audioFileArray = files[possibleKeys[0]];
      const audioFile = Array.isArray(audioFileArray) ? audioFileArray[0] : audioFileArray;
      
      if (!audioFile || !audioFile.filepath) {
        return res.status(400).json({ error: "El archivo de audio recibido es inválido." });
      }

      filePath = audioFile.filepath;
      const originalFilename = audioFile.originalFilename || "audio.mp3";

      // Subida al Administrador de Archivos de Google AI Studio
      const uploadResponse = await fileManager.uploadFile(filePath, {
        mimeType: "audio/mp3",
        displayName: originalFilename,
      });

      let fileState = await fileManager.getFile(uploadResponse.file.name);
      while (fileState.state === "PROCESSING") {
        await new Promise((resolve) => setTimeout(resolve, 3000));
        fileState = await fileManager.getFile(uploadResponse.file.name);
      }

      if (fileState.state === "FAILED") {
        throw new Error("El procesamiento del audio falló en los servidores de Google.");
      }

      const model = genAI.getGenerativeModel({ model: "gemini-3.8-flash" });

      const prompt = `
        Actúa como un transcriptor profesional y analista de sesiones de coaching ejecutivo.
        Analiza este archivo de audio completo de principio a fin, manteniendo la duración íntegra sin omitir fragmentos.
        
        Requisitos estrictos de formato:
        1. Identifica y etiqueta a los hablantes estrictamente como "CLIENTE" y el nombre institucional del Coach según corresponda.
        2. Anonimiza cualquier dato sensible, nombres de terceros o información privada mencionada.
        3. Estructura el resultado con marcas de tiempo (timestamps) ordenadas y párrafos claros.
      `;

      const result = await model.generateContent([
        {
          fileData: {
            fileUri: uploadResponse.file.uri,
            mimeType: uploadResponse.file.mimeType,
          },
        },
        prompt,
      ]);

      let responseText = "";
      try {
        responseText = result.response.text();
      } catch (textErr) {
        console.error("Error extrayendo texto de la respuesta:", textErr);
        responseText = "Transcripción generada.";
      }

      if (!responseText) {
        responseText = "La IA procesó el audio pero no devolvió contenido de texto.";
      }

      if (filePath && fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
      }

      return res.status(200).json({ success: true, transcription: responseText });

    } catch (error) {
      console.error("Error en el servidor:", error);
      if (filePath && fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
      }
      return res.status(500).json({ error: error.message || "Error al procesar el audio con la IA." });
    }
  });
});

app.listen(port, () => {
  console.log(`Servidor de Render corriendo en el puerto ${port}`);
});
