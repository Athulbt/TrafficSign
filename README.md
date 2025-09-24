# Traffic Sign Detection and Voice Alert System 🚦🔊

## Overview
This project implements a **real-time traffic sign detection system** using **YOLOv3** and **CNN**, integrated with a **Raspberry Pi 4**.  
Detected signs are announced through a **Bluetooth speaker** as voice alerts, improving driver awareness and safety.
Dataset Created by Athul Babu by clicking and training 500+ pictures

---

## Features
- Real-time detection of traffic signs using **YOLOv3**
- Classification with **CNN** for high accuracy
- **Camera integration** for live video feed
- Voice alerts delivered via **Bluetooth speaker**
- Runs on **Raspberry Pi 4** (lightweight edge computing)

---

## Tech Stack
- **Languages:** Python
- **Frameworks & Libraries:** TensorFlow / Keras, OpenCV
- **Model:** YOLOv3 + CNN
- **Hardware:** Raspberry Pi 4, Pi Camera, Bluetooth Speaker

---

## System Workflow
1. Capture live video using the Pi Camera
2. Preprocess frames with OpenCV
3. Detect traffic signs using YOLOv3
4. Classify sign with CNN
5. Generate voice alert via Text-to-Speech on Bluetooth speaker

---

## Installation

### Prerequisites
- Raspberry Pi 4 with Raspbian OS
- Python 3.8+
- Pi Camera enabled
- Bluetooth speaker paired

### Steps
```bash
# Clone this repository
git clone https://github.com/Athulbt/TrafficSign
cd TrafficSign

# Install required packages

# Run the system
python detect.py
