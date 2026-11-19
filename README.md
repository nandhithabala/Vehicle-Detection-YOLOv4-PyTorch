# 🚗 Real-Time Vehicle Detection using YOLOv4 & PyTorch

A real-time computer vision web application built with **Python, PyTorch, OpenCV, and Streamlit** to detect and classify vehicles (cars, motorbikes, buses, and trucks) in traffic video streams.

## ✨ Key Features
* **PyTorch Inference Pipeline:** Runs forward-pass YOLOv4 object detection using PyTorch tensors (`model.eval()` and `torch.no_grad()`).
* **Interactive Web Dashboard:** Built with Streamlit, featuring live sliders for **Confidence Threshold** and **Non-Maximum Suppression (NMS)** tuning.
* **Vehicle Class Filtering:** Automatically filters detections to highlight traffic classes (`car`, `motorbike`, `bus`, `truck`).

## 🛠️ Installation & Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Download YOLOv4 Pretrained Weights:**
   Download [yolov4.weights](https://github.com) and place it inside the root project folder.
3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

## 🙏 Acknowledgments
* Core PyTorch Darknet implementation adapted from [Tianxiaomo/pytorch-YOLOv4](https://github.com) (Apache-2.0 License).
* Original YOLOv4 architecture and weights by [AlexeyAB/darknet](https://github.com).
