import React, { useState, useRef } from 'react';
import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile } from '@ffmpeg/util';
import * as ort from 'onnxruntime-web'; // <-- Import ONNX Runtime for Web

function Test2() {
    const [status, setStatus] = useState('');
    const [processing, setProcessing] = useState(false);
    const [outputUrl, setOutputUrl] = useState('');
    const ffmpegRef = useRef(new FFmpeg());
    const videoInputRef = useRef();

    const imageToTensor = async (imageBitmap) => {
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

    const applyMask = async (imageBitmap, mask) => {
        const canvas = new OffscreenCanvas(224, 128);
        const ctx = canvas.getContext("2d");
        ctx.drawImage(imageBitmap, 0, 0, 224, 128);
        const imageData = ctx.getImageData(0, 0, 224, 128);

        for (let i = 0; i < mask.length; i++) {
            const alpha = mask[i] > 0.5 ? 255 : 0;
            imageData.data[i * 4 + 3] = alpha;
        }

        ctx.putImageData(imageData, 0, 0);
        return canvas;
    };

    const handleProcess = async () => {
        const videoFile = videoInputRef.current.files[0];

        if (!videoFile) {
            alert("Please upload a video file.");
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
        const session = await ort.InferenceSession.create(process.env.PUBLIC_URL + "/model.onnx");

        // Detect frame count
        const files = ffmpeg.FS('readdir', '/');
        const frameFiles = files.filter(f => /^frame_\d{3}\.png$/.test(f));
        const frameCount = frameFiles.length;

        for (let i = 1; i <= frameCount; i++) {
            const frameName = `frame_${String(i).padStart(3, '0')}.png`;
            const frameData = await ffmpeg.readFile(frameName);

            const imageBitmap = await createImageBitmap(new Blob([frameData.buffer], { type: "image/png" }));
            const tensor = await imageToTensor(imageBitmap);

            const maskTensor = await session.run({ input_1: tensor });
            const outputName = session.outputNames[0];
            const mask = maskTensor[outputName].data;

            const composited = await applyMask(imageBitmap, mask);
            const compositedBlob = await composited.convertToBlob({ type: "image/png" });

            await ffmpeg.writeFile(`masked_${frameName}`, await fetchFile(compositedBlob));
            setStatus(`Processed frame ${i} / ${frameCount}`);
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
            <h1>Video Frame Segmentation</h1>
            <label>Upload video (.mp4):</label><br />
            <input type="file" ref={videoInputRef} accept="video/mp4" disabled={processing} /><br />
            <button onClick={handleProcess} disabled={processing}>Process Video</button>
            <p>{status}</p>
            {outputUrl && (
                <video src={outputUrl} controls style={{ display: 'block', marginTop: 20 }} />
            )}
        </div>
    );
}

export default Test2;