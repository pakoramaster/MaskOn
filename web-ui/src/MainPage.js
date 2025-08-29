import React, { useState, useRef, useEffect } from "react";
import { FFmpeg } from "@ffmpeg/ffmpeg";
import { fetchFile } from "@ffmpeg/util";
import Nav from "./Nav";
import * as ort from "onnxruntime-web";

function MainPage() {
  const [status, setStatus] = useState("");
  const [progress, setProgress] = useState(0);
  const [processing, setProcessing] = useState(false);
  const [outputUrl, setOutputUrl] = useState("");
  const [screenColor, setScreenColor] = useState("");
  const [maskThreshold, setMaskThreshold] = useState(0.5);
  const ffmpegRef = useRef(new FFmpeg());
  const videoInputRef = useRef();
  const [selectedFileName, setSelectedFileName] = useState("");

  const imageToTensor = async (imageBitmap) => {
    const targetWidth = 256;
    const targetHeight = 256;
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

    return new ort.Tensor("float32", floatData, [
      1,
      targetHeight,
      targetWidth,
      3,
    ]);
  };

  const applyMask = async (imageBitmap, mask) => {
    // Overlay mask and generate green/blue screen image
    const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(imageBitmap, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Choose background color based on user selection
    let bgColor=[0, 0, 0, 0];
    if (screenColor === "green") {
      bgColor = [0, 255, 0, 255];
    } else if (screenColor === "blue") {
      bgColor = [0, 0, 255, 255];
    }

    for (let i = 0; i < mask.length; i++) {
      const maskValue = mask[i];
      if (maskValue > maskThreshold) {
        // <-- Use maskThreshold from state
        // Foreground: keep original pixel
        // ...do nothing, keep original RGBA...
      } else {
        // Background: set to selected color
        imageData.data[i * 4 + 0] = bgColor[0]; // R
        imageData.data[i * 4 + 1] = bgColor[1]; // G
        imageData.data[i * 4 + 2] = bgColor[2]; // B
        imageData.data[i * 4 + 3] = bgColor[3]; // A
      }
    }
    ctx.putImageData(imageData, 0, 0);
    return canvas;
  };

  const handleProcessPicture = async () => {
    const imageFile = videoInputRef.current.files[0];
    if (!imageFile) {
      alert("Please upload an image file.");
      return;
    }

    setProcessing(true);
    setStatus("Loading image...");
    setProgress(10);

    // Load image as bitmap
    const imageBitmap = await createImageBitmap(imageFile);

    setStatus("Loading model...");
    const session = await ort.InferenceSession.create("model.onnx");

    setStatus("Segmenting...");
    setProgress(30);

    // Prepare tensor
    const tensor = await imageToTensor(imageBitmap);
    const maskTensor = await session.run({ input_4: tensor });
    const outputName = session.outputNames[0];
    const mask = maskTensor[outputName].data;

    // Prepare mask for original image size
    const maskWidth = 256;
    const maskHeight = 256;
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

    // Resize mask to match original image size
    const maskCanvasLarge = new OffscreenCanvas(
      imageBitmap.width,
      imageBitmap.height
    );
    const maskCtxLarge = maskCanvasLarge.getContext("2d");
    maskCtxLarge.drawImage(
      maskCanvasSmall,
      0,
      0,
      imageBitmap.width,
      imageBitmap.height
    );
    const maskLargeImageData = maskCtxLarge.getImageData(
      0,
      0,
      imageBitmap.width,
      imageBitmap.height
    );
    const maskLarge = new Float32Array(imageBitmap.width * imageBitmap.height);
    for (let j = 0; j < maskLarge.length; j++) {
      maskLarge[j] = maskLargeImageData.data[j * 4] / 255;
    }

    setStatus("Compositing...");
    setProgress(70);

    // Apply mask and get composited image
    const composited = await applyMask(imageBitmap, maskLarge);

    setStatus("Preparing download...");
    setProgress(90);

    const compositedBlob = await composited.convertToBlob({ type: "image/png" });
    setOutputUrl(URL.createObjectURL(compositedBlob));

    setStatus("");
    setProgress(100);
    setProcessing(false);
  }
  const handleProcessVideo = async () => {
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

    // --- Get original frame rate ---
    setStatus("Detecting original frame rate...");
    // Write ffprobe output to a file (stdout redirected to file)
    await ffmpeg.ffprobe([
      "-v",
      "error",
      "-select_streams",
      "v:0",
      "-show_entries",
      "stream=r_frame_rate",
      "-of",
      "default=noprint_wrappers=1:nokey=1",
      "-i",
      "input.mp4",
      "-o",
      "framerate.txt",
    ]);
    // Read the file back
    const frTextData = await ffmpeg.readFile("framerate.txt");
    const outputStr = new TextDecoder().decode(frTextData);

    // Parse frame rate from file content
    let frameRate = 30; // fallback
    try {
      const match = outputStr.match(/(\d+)\/(\d+)/);
      if (match) {
        frameRate = Math.round(parseInt(match[1], 10) / parseInt(match[2], 10));
      }
    } catch (e) {
      // fallback to 30
    }

    setStatus("Loading model...");
    const session = await ort.InferenceSession.create("model.onnx");

    // Detect number of frames by listing files
    let frameCount = 0;
    // Try up to 999 frames
    for (let i = 1; i <= 999; i++) {
      try {
        await ffmpeg.readFile(`frame_${String(i).padStart(3, "0")}.png`);
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
      setProgress(Math.round((i / frameCount) * 100));
      const frameName = `frame_${String(i).padStart(3, "0")}.png`;
      const frameData = await ffmpeg.readFile(frameName);
      const imageBitmap = await createImageBitmap(
        new Blob([frameData.buffer], { type: "image/png" })
      );
      const tensor = await imageToTensor(imageBitmap);
      const maskTensor = await session.run({ input_4: tensor });
      const outputName = session.outputNames[0];
      const mask = maskTensor[outputName].data;

      // Visualize mask as grayscale image, resized to original frame size
      const maskWidth = 256;
      const maskHeight = 256;
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
      const maskCanvasLarge = new OffscreenCanvas(
        imageBitmap.width,
        imageBitmap.height
      );
      const maskCtxLarge = maskCanvasLarge.getContext("2d");
      maskCtxLarge.drawImage(
        maskCanvasSmall,
        0,
        0,
        imageBitmap.width,
        imageBitmap.height
      );
      const maskLargeImageData = maskCtxLarge.getImageData(
        0,
        0,
        imageBitmap.width,
        imageBitmap.height
      );
      const maskLarge = new Float32Array(
        imageBitmap.width * imageBitmap.height
      );
      for (let j = 0; j < maskLarge.length; j++) {
        maskLarge[j] = maskLargeImageData.data[j * 4] / 255;
      }
      const composited = await applyMask(
        imageBitmap,
        maskLarge,
        imageBitmap.width,
        imageBitmap.height
      );
      const compositedBlob = await composited.convertToBlob({
        type: "image/png",
      });
      await ffmpeg.writeFile(
        `masked_${frameName}`,
        await fetchFile(compositedBlob)
      );
    }

    setStatus("Encoding final video...");
    await ffmpeg.exec([
      "-framerate",
      frameRate.toString(),
      "-i",
      "masked_frame_%03d.png",
      "-c:v",
      "libx264",
      "-pix_fmt",
      "yuv420p",
      "output.mp4",
    ]);

    const outputData = await ffmpeg.readFile("output.mp4");
    const outputBlob = new Blob([outputData.buffer], { type: "video/mp4" });
    setOutputUrl(URL.createObjectURL(outputBlob));
    setStatus("");
    setProgress(100);
    setProcessing(false);
  };

  // Set dark blue background for the entire page
  useEffect(() => {
    // Save original background
    const originalBg = document.body.style.background;

    // Apply gradient background
    // document.body.style.background = "linear-gradient(135deg, #091734ff, #081a2dff, #171717)";
    document.body.style.backgroundColor = "#18202bff";

    // Cleanup: restore original background when unmounted
    return () => {
      document.body.style.background = originalBg;
    };
  }, []);
  useEffect(() => {
    setOutputUrl("");
    if (selectedFileName && selectedFileName.toLowerCase().endsWith(".mp4")) {
      setScreenColor("green");
    } else {
      setScreenColor("");
    }
  }, [selectedFileName]);

  return (
    <div className="min-h-screen bg-[#18202b]">
      <Nav />
      <div className="w-full flex flex-col items-center px-4 sm:px-0">
        {/* Styled video uploader */}
        <label
          htmlFor="video-upload"
          className={`mt-6 w-full max-w-md flex flex-col items-center rounded-xl border border-indigo-900 bg-[#202a38] p-6 text-white shadow transition-all duration-200 ${
            processing ? "opacity-60 cursor-not-allowed" : "hover:shadow-lg"
          }`}
          style={{ marginBottom: 18 }}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
            className="size-6"
            style={{ width: 32, height: 32 }}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M7.5 7.5h-.75A2.25 2.25 0 0 0 4.5 9.75v7.5a2.25 2.25 0 0 0 2.25 2.25h7.5a2.25 2.25 0 0 0 2.25-2.25v-7.5a2.25 2.25 0 0 0-2.25-2.25h-.75m0-3-3-3m0 0-3 3m3-3v11.25m6-2.25h.75a2.25 2.25 0 0 1 2.25 2.25v7.5a2.25 2.25 0 0 1-2.25 2.25h-7.5a2.25 2.25 0 0 1-2.25-2.25v-.75"
            />
          </svg>
          <span className="mt-4 font-semibold text-base">
            {selectedFileName ? selectedFileName : "Upload your file"}
          </span>
          <span className="mt-2 inline-block rounded-full border border-indigo-900 bg-indigo-800 px-4 py-2 text-center text-sm font-semibold text-white shadow-md hover:bg-indigo-700 transition-colors duration-150">
            Browse files
          </span>
          <input
            type="file"
            id="video-upload"
            ref={videoInputRef}
            accept="video/mp4,image/png,image/jpeg,image/jpg"
            className="sr-only"
            disabled={processing}
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                setSelectedFileName(e.target.files[0].name);
              } else {
                setSelectedFileName("");
              }
            }}
          />
        </label>
        {selectedFileName && selectedFileName.toLowerCase().endsWith(".mp4") && (
          <div className="w-full max-w-md">
            <label className="font-semibold mb-2 block text-base text-white">
              Background Color:
            </label>
            <fieldset
              className="flex flex-row gap-4"
              style={{ border: "none", padding: 0, margin: 0 }}
            >
              <legend className="sr-only">Color</legend>

              {/* Green */}
              <label
                htmlFor="ColorGreen"
                className={` ml-1 block size-4 rounded-full bg-green-500 shadow-sm cursor-pointer ${
                  screenColor === "green"
                    ? "ring-2 ring-green-500 ring-offset-2"
                    : ""
                }`}
                style={{ display: "inline-block" }}
              >
                <input
                  type="radio"
                  name="ColorOption"
                  value="green"
                  id="ColorGreen"
                  className="sr-only"
                  checked={screenColor === "green"}
                  onChange={() => setScreenColor("green")}
                  disabled={processing}
                />
                <span className="sr-only">Green</span>
              </label>

              {/* Blue */}
              <label
                htmlFor="ColorBlue"
                className={`block size-4 rounded-full bg-blue-500 shadow-sm cursor-pointer ${
                  screenColor === "blue"
                    ? "ring-2 ring-blue-500 ring-offset-2"
                    : ""
                }`}
                style={{ display: "inline-block" }}
              >
                <input
                  type="radio"
                  name="ColorOption"
                  value="blue"
                  id="ColorBlue"
                  className="sr-only"
                  checked={screenColor === "blue"}
                  onChange={() => setScreenColor("blue")}
                  disabled={processing}
                />
                <span className="sr-only">Blue</span>
              </label>
            </fieldset>
          </div>
        )}
        <div className="w-full max-w-md mt-6">
          <label className="font-semibold mb-2 block text-base text-white">
            Mask Threshold:{" "}
            <span className="text-indigo-400">{maskThreshold.toFixed(2)}</span>
            <input
              type="range"
              min={0.2}
              max={0.8}
              step={0.05}
              value={maskThreshold}
              onChange={(e) => setMaskThreshold(Number(e.target.value))}
              disabled={processing}
              className="w-full accent-indigo-800 mt-2 hover:accent-indigo-700 transition-colors duration-150"
            />
          </label>
        </div>
        <button
          onClick={
            selectedFileName && selectedFileName.toLowerCase().endsWith(".mp4")
              ? handleProcessVideo
              : handleProcessPicture
          }
          disabled={processing}
          className={`w-2/3 max-w-fit mt-2 rounded-full border border-indigo-900 bg-indigo-800 py-2 px-5 text-base font-semibold text-white shadow-md hover:bg-indigo-700 transition-colors duration-150 ${
            processing ? "opacity-60 cursor-not-allowed" : ""
          }`}
        >
          Process {selectedFileName && selectedFileName.toLowerCase().endsWith(".mp4") ? "Video" : "Picture"}
        </button>
        {processing && (
          <div className="w-2/3 max-w-xs mt-4 h-3 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-indigo-500 transition-all duration-200"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        )}
        {status !== "" && (
          <p className="text-white font-medium text-sm mt-2">{status}</p>
        )}
        {outputUrl && (
          <div className="w-full max-w-md mt-8 flex flex-col items-center">
            {selectedFileName && selectedFileName.toLowerCase().endsWith(".mp4") ? (
              <>
                <video
                  src={outputUrl}
                  controls
                  className="w-full rounded-lg border border-indigo-900 shadow-lg"
                />
                <a href={outputUrl} download="output.mp4" className="w-full">
                  <button className="mt-4 mb-6 w-full rounded-full border border-indigo-900 bg-indigo-800 px-6 py-3 text-base font-semibold text-white shadow-md hover:bg-indigo-700 transition-colors duration-150">
                    Download Video
                  </button>
                </a>
              </>
            ) : (
              <>
                <img
                  src={outputUrl}
                  alt="Processed"
                  className="w-full rounded-lg border border-indigo-900 shadow-lg object-contain"
                />
                <a href={outputUrl} download="output.png" className="w-full">
                  <button className="mt-4 mb-6 w-full rounded-full border border-indigo-900 bg-indigo-800 px-6 py-3 text-base font-semibold text-white shadow-md hover:bg-indigo-700 transition-colors duration-150">
                    Download Image
                  </button>
                </a>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default MainPage;
