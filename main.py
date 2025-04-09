import torch
import cv2
from PIL import Image, ImageTk
import numpy as np
import time
import csv
import datetime
import threading
import tkinter as tk
from tkinter import scrolledtext

from models.model_loader import load_owlvit_model
from utils.video_utils import draw_fps

# ========== Setup ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor, model = load_owlvit_model()
model.to(device)

prompt_file = "prompts.txt"
csv_file = "detections.csv"
DETECT_EVERY = 15

def load_prompts():
    with open(prompt_file, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def save_prompts(prompts):
    with open(prompt_file, "w") as f:
        f.write("\n".join(prompts))

prompts = load_prompts()
detections_to_draw = []

# ========== GUI App ==========
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Zero-Shot Detection GUI")
        self.detection_active = False
        self.frame_count = 0
        self.detections = []
        self.lock = threading.Lock()

        self.setup_ui()
        self.cap = cv2.VideoCapture(1)
        self.start_video_loop()

    def setup_ui(self):
        self.video_label = tk.Label(self.root)
        self.video_label.grid(row=0, column=0, rowspan=6, padx=10, pady=10)

        tk.Label(self.root, text="Detection Prompts").grid(row=0, column=1)
        self.prompt_box = scrolledtext.ScrolledText(self.root, width=30, height=10)
        self.prompt_box.grid(row=1, column=1)
        self.prompt_box.insert(tk.END, "\n".join(prompts))

        self.update_btn = tk.Button(self.root, text="Update Prompts", command=self.update_prompts)
        self.update_btn.grid(row=2, column=1, pady=5)

        self.toggle_btn = tk.Button(self.root, text="Start Detection", command=self.toggle_detection)
        self.toggle_btn.grid(row=3, column=1, pady=5)

        tk.Label(self.root, text="Detections:").grid(row=4, column=1)
        self.detection_list = tk.Listbox(self.root, width=40, height=10)
        self.detection_list.grid(row=5, column=1, padx=5, pady=5)

    def update_prompts(self):
        updated = self.prompt_box.get("1.0", tk.END).strip().split("\n")
        save_prompts(updated)
        print("[UI] Prompts updated!")

    def toggle_detection(self):
        self.detection_active = not self.detection_active
        self.toggle_btn.config(text="Stop Detection" if self.detection_active else "Start Detection")

    def start_video_loop(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))

            with self.lock:
                for label, score, box in self.detections:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({score:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if self.detection_active and self.frame_count % DETECT_EVERY == 0:
                threading.Thread(target=self.run_detection, args=(frame.copy(),), daemon=True).start()

            # Overlay FPS
            curr_time = time.time()
            if hasattr(self, "prev_time"):
                fps = 1 / max((curr_time - self.prev_time), 1e-5)
                frame = draw_fps(frame, fps)
            self.prev_time = curr_time

            # Show video
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img))
            self.video_label.imgtk = img
            self.video_label.configure(image=img)

        self.frame_count += 1
        self.root.after(10, self.start_video_loop)

    def run_detection(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        text_prompts = load_prompts()
        inputs = processor(text=text_prompts, images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                outputs = model(**inputs)

        results = processor.post_process_object_detection(
            outputs=outputs,
            threshold=0.2,
            target_sizes=torch.tensor([image.size[::-1]]).to(device)
        )[0]

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detections = []

        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                confidence = round(score.item(), 2)
                label_text = text_prompts[label.item()]
                writer.writerow([timestamp, label_text, confidence, *box])
                detections.append((label_text, confidence, box))
                self.root.after(0, lambda txt=f"{label_text} ({confidence})": self.add_detection(txt))

        with self.lock:
            self.detections = detections

    def add_detection(self, text):
        self.detection_list.insert(0, text)
        if self.detection_list.size() > 10:
            self.detection_list.delete(10)

# ========== Main ==========
if __name__ == "__main__":
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "label", "confidence", "x1", "y1", "x2", "y2"])

    root = tk.Tk()
    app = App(root)
    root.mainloop()
