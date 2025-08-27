# MaskOn: Web Based Green/Blue Screen Generaion Tool

MaskOn is a two-part project for instant, privacy-friendly video background removal, powered by deep learning and modern web technologies.

---

## 1. Model Training

- **Framework:** TensorFlow / Keras
- **Architecture:** U-Net with ResNet-50 encoder
- **Dataset:** [COCO 2017](https://cocodataset.org/) (trained on the "person" category)
- **Features:**
  - Custom data pipeline using `pycocotools`
  - Data augmentation (horizontal flip)
  - Loss: Combined BCE, Dice, Edge, and Tversky losses
  - Metrics: Dice coefficient, IoU
  - Model exported to ONNX for browser inference

**Training Notebook:**  
See [`model-training/model-trainer.ipynb`](model-training/model-trainer.ipynb) for full code and details.

---

## 2. Web UI

- **Framework:** React
- **Libraries:** [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web), [@ffmpeg/ffmpeg](https://github.com/ffmpegwasm/ffmpeg.wasm), Tailwind CSS
- **Features:**
  - All processing runs locally in your browser (no uploads, privacy-first)
  - Upload a video, select green or blue screen, and download the masked result
  - Adjustable mask threshold slider for fine-tuning
  - Modern, responsive UI



---

## Project Structure

```
rotoscope-tool/
│
├── model-training/
│   └── model-trainer.ipynb      # Jupyter notebook for model training and export
│   └── coco2017                 # Coco Dataset 
├── web-ui/
│   ├── public/
│   │   └── model.onnx           # Exported ONNX model for browser inference
│   ├── src/
│   │   ├── MainPage.js          # Main React component
│   │   ├── About.js             # About page
│   │   └── ...                  # Other React components and assets
│   └── ...                      # React app config and build files
│
└── README.md
```

---

## How It Works

1. **Model Training:**  
   - Train a segmentation model to detect people in images using COCO dataset.
   - Export the trained model to ONNX format for browser use.

2. **Web UI:**  
   - User uploads a video.
   - Each frame is segmented in-browser using the ONNX model.
   - Background is replaced with a green or blue screen.
   - Video is re-encoded and available for download, all in the browser.

> **Note:** The segmentation model was trained specifically on the "person" category from the COCO dataset. Results may vary or be less accurate for videos containing subjects other than people.
---

## Try It Out

Visit the live site:  
**[https://maskon-gs.netlify.app/](https://maskon-gs.netlify.app/)**

---

## License

MIT License