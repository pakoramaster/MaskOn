import React, { useState, useRef } from 'react';
import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile } from '@ffmpeg/util';
import * as ort from 'onnxruntime-web';

function MainPage() {
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
        // Overlay mask and generate green screen image
        const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
        const ctx = canvas.getContext("2d");
        ctx.drawImage(imageBitmap, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        // mask is expected to be resized to imageBitmap.width x imageBitmap.height
        for (let i = 0; i < mask.length; i++) {
            const maskValue = mask[i];
            if (maskValue > 0.5) {
                // Foreground: keep original pixel
                // ...do nothing, keep original RGBA...
            } else {
                // Background: set to green
                imageData.data[i * 4 + 0] = 0;   // R
                imageData.data[i * 4 + 1] = 255; // G
                imageData.data[i * 4 + 2] = 0;   // B
                imageData.data[i * 4 + 3] = 255; // A
            }
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
        const session = await ort.InferenceSession.create("model.onnx");

        // Detect number of frames by listing files
        let frameCount = 0;
        // Try up to 999 frames
        for (let i = 1; i <= 999; i++) {
            try {
                await ffmpeg.readFile(`frame_${String(i).padStart(3, '0')}.png`);
                frameCount = i;
            } catch {
                break;
            }
        }
        if (frameCount === 0) {
            setStatus("No frames extracted.");
            setProcessing(false);
            return;
        }

        for (let i = 1; i <= frameCount; i++) {
            setStatus(`Processing frame ${i} of ${frameCount}...`);
            const frameName = `frame_${String(i).padStart(3, '0')}.png`;
            const frameData = await ffmpeg.readFile(frameName);
            const imageBitmap = await createImageBitmap(new Blob([frameData.buffer], { type: "image/png" }));
            const tensor = await imageToTensor(imageBitmap);
            const maskTensor = await session.run({ input_7: tensor });
            const outputName = session.outputNames[0];
            const mask = maskTensor[outputName].data;

            // Visualize mask as grayscale image, resized to original frame size
            const maskWidth = 224;
            const maskHeight = 128;
            const maskCanvasSmall = new OffscreenCanvas(maskWidth, maskHeight);
            const maskCtxSmall = maskCanvasSmall.getContext("2d");
            const maskImageData = maskCtxSmall.createImageData(maskWidth, maskHeight);
            for (let j = 0; j < mask.length; j++) {
                const value = Math.round(mask[j] * 255);
                maskImageData.data[j * 4 + 0] = value;
                maskImageData.data[j * 4 + 1] = value;
                maskImageData.data[j * 4 + 2] = value;
                maskImageData.data[j * 4 + 3] = 255;
            }
            maskCtxSmall.putImageData(maskImageData, 0, 0);

            // Resize mask to match original frame size
            const maskCanvasLarge = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
            const maskCtxLarge = maskCanvasLarge.getContext("2d");
            maskCtxLarge.drawImage(maskCanvasSmall, 0, 0, imageBitmap.width, imageBitmap.height);
            const maskLargeImageData = maskCtxLarge.getImageData(0, 0, imageBitmap.width, imageBitmap.height);
            const maskLarge = new Float32Array(imageBitmap.width * imageBitmap.height);
            for (let j = 0; j < maskLarge.length; j++) {
                maskLarge[j] = maskLargeImageData.data[j * 4] / 255;
            }
            const composited = await applyMask(imageBitmap, maskLarge, imageBitmap.width, imageBitmap.height);
            const compositedBlob = await composited.convertToBlob({ type: "image/png" });
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
        setStatus("✅ Done! Download your green screen video below.");
        setProcessing(false);
    };

    return (
        <div>
            <h1>Video Green Screen Masker</h1>
            <label>Upload video (.mp4):</label><br />
            <input type="file" ref={videoInputRef} accept="video/mp4" disabled={processing} /><br />
            <button onClick={handleProcess} disabled={processing}>Process Video</button>
            <p>{status}</p>
            {outputUrl && (
                <div>
                    <h3>Masked Video Output</h3>
                    <video src={outputUrl} controls style={{ maxWidth: "100%", marginTop: 20 }} />
                    <br />
                    <a href={outputUrl} download="masked_output.mp4">
                        <button>Download Masked Video</button>
                    </a>
                </div>
            )}
        </div>
    );
}

export default MainPage;