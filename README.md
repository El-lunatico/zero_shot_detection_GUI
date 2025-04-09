# 🔍 Zero-Shot Object Detection GUI - Vyorius Internship Test

This project is a real-time object detection system using the OWL-ViT (Open World Learning Vision Transformer) model from HuggingFace. It allows users to detect **any object** by just entering its name — no need for retraining! A simple GUI is provided for editing prompts live, visualizing bounding boxes, and tracking detection performance.

---

## 🖥️ Features

- ✅ Real-time webcam-based detection using OWL-ViT
- ✅ GUI for live prompt editing
- ✅ Bounding box + confidence display
- ✅ CSV logging of detections
- ✅ FPS counter & visual feedback
- ✅ Runs on CPU or GPU

---

## ⚙️ Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/yourusername/ZeroShot-Object-Detection-GUI.git
cd ZeroShot-Object-Detection-GUI
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Install required Python libraries

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch (Choose one)

#### ➤ For GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### ➤ For CPU:
```bash
pip install torch torchvision torchaudio
```

---

## 🧠 Model Download and Usage

This project uses the `owlvit-base-patch32` model from HuggingFace's Transformers library.

- You **do not need** to manually download the model.
- It will be downloaded **automatically** the first time you run the code.
- HuggingFace’s `transformers` will handle loading from the cloud and caching locally.

---

## ▶️ How to Run

Simply execute the main script:

```bash
python main.py
```

### 📝 Instructions:
- Type your detection prompts in the box (e.g., `person, helmet, sunglasses`)
- Click `Update Prompts`
- Then click `Start Detection`

You will see:
- A live webcam feed with bounding boxes
- Object names and confidence scores
- CSV file (`detections.csv`) logging the detected objects

---

## 📁 Project Structure

```
ZeroShot-Object-Detection-GUI/
├── main.py                  # Main application with GUI
├── prompts.txt              # Default prompt list
├── detections.csv           # CSV log of detections
├── requirements.txt         # All dependencies
├── models/
│   └── model_loader.py      # Loads OWL-ViT model
├── utils/
│   └── video_utils.py       # FPS drawing utility
└── README.md                # You're reading this!
```

---

## 📜 License

MIT License - free for personal or commercial use. Just give credit if reused!

---

Built with 💙 using Python, PyTorch, Transformers, OpenCV, and Tkinter.
