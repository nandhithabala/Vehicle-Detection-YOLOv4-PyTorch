import os
import cv2
import torch
import tempfile
import requests
import streamlit as st
from tool.darknet2pytorch import Darknet
from tool.torch_utils import do_detect

# --- 1. Page Configuration ---
st.set_page_config(page_title="PyTorch YOLOv4 Vehicle Detection", layout="wide")
st.title("🚗 Real-Time Vehicle Detection (PyTorch + YOLOv4)")
st.write("Deep learning inference pipeline using PyTorch to detect and classify vehicles in video feeds.")

# --- 2. Sidebar Settings ---
st.sidebar.header("Inference Hyperparameters")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
nms_threshold = st.sidebar.slider("NMS Threshold", 0.1, 1.0, 0.4, 0.05)

# Automatically locate files inside pytorch-YOLOv4-master
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(BASE_DIR, "cfg", "yolov4-tiny.cfg")
WEIGHTS_PATH = os.path.join(BASE_DIR, "yolov4-tiny.weights")
NAMES_PATH = os.path.join(BASE_DIR, "data", "coco.names")

# --- 3. Load PyTorch Model ---
@st.cache_resource
def load_pytorch_yolo():
    model = Darknet(CFG_PATH)
    model.load_weights(WEIGHTS_PATH)



    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    model.eval()
    
    with open(NAMES_PATH, "r") as f:
        classes = [line.strip() for line in f.readlines()]
        
    vehicle_classes = {"car", "motorbike", "bus", "truck"}
    return model, classes, vehicle_classes, use_cuda

model, classes, vehicle_classes, use_cuda = load_pytorch_yolo()
st.sidebar.info(f"Running PyTorch on: **{'GPU (CUDA)' if use_cuda else 'CPU'}**")

# --- 4. Video Upload & PyTorch Inference Loop ---
uploaded_file = st.sidebar.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        height, width, _ = frame.shape
        
        sized = cv2.resize(frame, (model.width, model.height))
        sized_rgb = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            boxes = do_detect(model, sized_rgb, conf_threshold, nms_threshold, use_cuda)
        # Extract bounding boxes for the current frame
        detections = boxes[0] if len(boxes) > 0 else []
        
        for box in detections:
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            conf = box[5]
            class_id = int(box[6])

            
            class_name = classes[class_id]
            if class_name in vehicle_classes:
                label = f"{class_name.upper()} ({conf*100:.0f}%)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
        
    cap.release()
    st.success("PyTorch inference complete!")
else:
    st.info("👈 Upload a traffic video from the sidebar to run PyTorch detection.")
