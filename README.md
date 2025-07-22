# **Vehicle Detection, Tracking, Counting & Speed Estimation using YOLO, OpenCV, and Supervision**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange.svg)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-red.svg)](https://github.com/ultralytics/ultralytics)
[![Supervision](https://img.shields.io/badge/Supervision-Annotation%20%26%20Tracking-brightgreen.svg)](https://github.com/roboflow/supervision)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## **Project Overview**

This project implements a complete pipeline for **real-time vehicle detection, multi-object tracking, counting, and speed estimation**. It combines **YOLO** for object detection, **OpenCV** for video processing, and **Supervision** for annotation, tracking, and visualization.

Use cases include:

* Intelligent traffic monitoring.
* Vehicle flow analysis.
* Speed compliance and traffic law enforcement.

---

## **Demo Video**

Watch the full demonstration on YouTube:

[![Watch the video](https://img.youtube.com/vi/1HSTwBKCELk/hqdefault.jpg)](https://youtu.be/1HSTwBKCELk)

---

## **Features**

* **Real-Time Vehicle Detection** using YOLO.
* **Accurate Multi-Object Tracking** with ByteTrack.
* **Vehicle Counting** using configurable line zones (In/Out counts).
* **Speed Estimation** using **Perspective Transformation**.
* Supports **custom YOLO models** and various camera perspectives.

---

## **Perspective Transformation**

We use perspective transformation to convert camera view coordinates into a real-world top-down view. This is essential for accurate speed calculations.

![Perspective Transformation](https://github.com/Raafat-Nagy/Vehicle-Speed-Estimation-and-Counting-YOLO-Supervision/blob/main/data/annotated_frame.png)

**Speed Formula**:

```
speed (km/h) = (distance_in_meters / time_in_seconds) * 3.6
```

---

## **Technologies Used**

* **Python** (3.8+)
* **YOLO** (Ultralytics)
* **OpenCV** for video processing
* **Supervision** for tracking, line zones, and annotations
* **NumPy** for numerical computations

---

## **Project Structure**

```
Vehicle-Speed-Estimation-and-Counting-YOLO-Supervision
│
├── data
│   ├── vehicles.mp4
│   ├── vehicles_output.mp4
│   ├── frame.png
│   └── annotated_frame.png
│
├── models
│   ├── yolov8n.pt
│   └── VisDrone_YOLO_x2.pt
│
├── src
│   ├── annotator.py
│   ├── speed_estimator.py
│   ├── view_transformer.py
│   └── __init__.py
│
├── config.py
├── main.py
├── requirements.txt
└── Vehicle_Speed_Estimation_main.ipynb
```

---

## **Installation**

```bash
# Clone the repository
git clone https://github.com/Raafat-Nagy/Vehicle-Speed-Estimation-and-Counting-YOLO-Supervision.git
cd Vehicle-Speed-Estimation-and-Counting-YOLO-Supervision

# Install dependencies
pip install -r requirements.txt
```

## **How to Run**

```bash
# Run the script
python main.py
```

You can also check the **Jupyter Notebook** version:
`Vehicle_Speed_Estimation_main.ipynb`

---

## **Future Enhancements**

* Add **lane detection** and traffic density analysis.
* Deploy as a **web application** for real-time traffic monitoring.
* Support **multi-camera input** with IoT integration.
* Export statistics to **dashboards** for analytics.

---

## **Acknowledgments**

* [YOLO by Ultralytics](https://github.com/ultralytics/ultralytics)
* [OpenCV](https://opencv.org/)
* [Supervision by Roboflow](https://github.com/roboflow/supervision)
* [NumPy](https://numpy.org/)

---
