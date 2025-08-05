import React, { useState, useRef } from 'react';
import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile } from '@ffmpeg/util';
import * as ort from 'onnxruntime-web'; // <-- Import ONNX Runtime for Web

function Test() {
    const [status, setStatus] = useState('');
    const [processing, setProcessing] = useState(false);
    const [outputUrl, setOutputUrl] = useState('');
    const ffmpegRef = useRef(new FFmpeg());
    const videoInputRef = useRef();
    const maskInputRef = useRef();

    const imageToTensor = async (imageBitmap) => {
        // Always resize to 224x128
        const targetWidth = 224;
        const targetHeight = 128;
        const canvas = new OffscreenCanvas(targetWidth, targetHeight);
        const ctx = canvas.getContext("2d");
        ctx.drawImage(imageBitmap, 0, 0, targetWidth, targetHeight);

        const imgData = ctx.getImageData(0, 0, targetWidth, targetHeight);
        const data = imgData.data;
        const floatData = new Float32Array(targetWidth * targetHeight * 3);

        for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
            floatData[j] = data[i] / 255;
            floatData[j + 1] = data[i + 1] / 255;
            floatData[j + 2] = data[i + 2] / 255;
        }

        return new ort.Tensor("float32", floatData, [1, targetHeight, targetWidth, 3]);
    };

    const applyGreenScreen = async (imageBitmap, mask) => {
        const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
        const ctx = canvas.getContext("2d");
        ctx.drawImage(imageBitmap, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        for (let i = 0; i < mask.length; i++) {
            const alpha = mask[i] > 0.5 ? 1 : 0;
            const idx = i * 4;
            if (alpha === 0) {
                imageData.data[idx] = 0;
                imageData.data[idx + 1] = 255;
                imageData.data[idx + 2] = 0;
            }
        }

        ctx.putImageData(imageData, 0, 0);
        return canvas;
    };

    const handleProcess = async () => {
        const videoFile = videoInputRef.current.files[0];
        const maskFile = maskInputRef.current.files[0];

        if (!videoFile || !maskFile) {
            alert("Please upload both video and first-frame PNG mask.");
            return;
        }

        setProcessing(true);
        setStatus("Initializing FFmpeg...");

        const ffmpeg = ffmpegRef.current;
        await ffmpeg.load();

        await ffmpeg.writeFile("input.mp4", await fetchFile(videoFile));
        setStatus("Extracting frames...");
        await ffmpeg.exec(["-i", "input.mp4", "-qscale:v", "2", "frame_%03d.png"]);

        setStatus("Loading model...");
        const session = await ort.InferenceSession.create("model.onnx"); // <-- Use ort.InferenceSession

        const frameCount = 30; // Optional: Replace with dynamic detection

        for (let i = 1; i <= frameCount; i++) {
            const frameName = `frame_${String(i).padStart(3, '0')}.png`;
            const frameData = await ffmpeg.readFile(frameName);

            const imageBitmap = await createImageBitmap(new Blob([frameData.buffer], { type: "image/png" }));
            const tensor = await imageToTensor(imageBitmap);

            const maskTensor = await session.run({ input_1: tensor }); // <-- use 'input_1' as the key

            // Get the output name (usually session.outputNames[0])
            const outputName = session.outputNames[0];
            const mask = maskTensor[outputName].data; // <-- Use correct output key

            const composited = await applyGreenScreen(imageBitmap, mask);
            const compositedBlob = await new Promise(res => composited.convertToBlob({ type: "image/png" }).then(res));

            await ffmpeg.writeFile(`masked_${frameName}`, await fetchFile(compositedBlob));
        }

        setStatus("Encoding final video...");
        await ffmpeg.exec([
            "-framerate", "30",
            "-i", "masked_frame_%03d.png",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "output.mp4"
        ]);

        const outputData = await ffmpeg.readFile("output.mp4");
        const outputBlob = new Blob([outputData.buffer], { type: "video/mp4" });
        setOutputUrl(URL.createObjectURL(outputBlob));
        setStatus("✅ Done!");
        setProcessing(false);
    };

    return (
        <div>
            <h1>Client-side Background Remover</h1>
            <label>Upload video (.mp4):</label><br />
            <input type="file" ref={videoInputRef} accept="video/mp4" disabled={processing} /><br />

            <label>Upload background-removed PNG of first frame:</label><br />
            <input type="file" ref={maskInputRef} accept="image/png" disabled={processing} /><br />

            <button onClick={handleProcess} disabled={processing}>Remove Background</button>

            <p>{status}</p>

            {outputUrl && (
                <video src={outputUrl} controls style={{ display: 'block', marginTop: 20 }} />
            )}
        </div>
    );
}

export default Test;