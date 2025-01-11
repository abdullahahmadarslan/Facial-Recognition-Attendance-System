# AI-Powered Facial Recognition Attendance System

Revolutionizing attendance management for academic institutions with an advanced, AI-driven facial recognition system. This project combines cutting-edge technology and innovative digital image processing techniques to deliver a scalable and efficient solution for attendance tracking.

---

## 🚀 Features

### 1. Real-Time Face Detection & Recognition
- **Single Shot Multibox Detector (SSD):** A deep learning-based object detection framework for real-time face detection.
- **ResNet-50:** A Convolutional Neural Network (CNN) used for generating high-quality feature vectors for facial recognition, ensuring accuracy in vector comparisons.

### 2. Advanced Digital Image Processing
Overcoming hardware limitations with a 0.9 MP laptop camera by applying:
- **Contrast Enhancement:** Clearer visuals for better detection.
- **Edge Sharpening with High-Boost Filtering:** Crisp, high-quality images.
- **Brightness Adjustment:** Improved performance under suboptimal lighting conditions.
- **Noise Reduction:** Histogram analysis for cleaner input images.

### 3. Robust and Efficient System Design
- **High Accuracy:** Achieved 95% similarity match between faces using cosine similarity for vector comparisons.
- **Frame-by-Frame Processing:** Real-time video frame capture, face detection, vector generation, and instant comparison for attendance marking.
- **Secure Database Integration:** PostgreSQL for structured and secure storage of student records and attendance logs.
- **Duplicate Prevention:** Restricts attendance to a single entry per session.
- **Real-Time Updates:** Powered by Flask-SocketIO for instant notifications on the frontend.

### 4. User-Friendly Frontend
- **React-based Dashboard:** Intuitive and visually appealing dashboard for easy navigation and management.

---

## 🛠️ Technology Stack

### Backend:
- **Python**: Core implementation for logic and image processing.
- **Flask**: Backend framework for RESTful APIs.
- **Flask-SocketIO**: Real-time updates for instant notifications.
- **OpenCV**: Digital image processing and frame capture.
- **ResNet-50**: Facial feature extraction and vector matching.

### Database:
- **PostgreSQL**: Secure and structured data storage for attendance logs and student records.

### Frontend:
- **React.js**: Interactive and responsive web application for managing attendance.

---

## 📈 Results
- **Reliability:** Delivered accurate face detection and recognition, even with limited hardware capabilities.
- **Optimization:** Improved performance under dynamic conditions using advanced image processing techniques.
- **Accuracy:** Achieved high similarity matches for precise attendance marking.

---

## 🌟 Future Scope
This system has immense potential for future integration into **university management systems** to:
- Prevent student **proxies** during attendance.
- Enable a more **efficient, secure, and transparent** attendance tracking process.

## 🚀 Getting Started

### Prerequisites
- **Python 3.8+**
- **Node.js 14+**
- **PostgreSQL**

### Setup Instructions

#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/FacialRecognitionAttendanceSystem.git
cd FacialRecognitionAttendanceSystem
```

#### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

#### 3. Frontend Setup
```bash
cd ../frontend
npm install
npm start
```

#### 4. Database Setup
- Install PostgreSQL and create a database for the system.
- Update the database connection string in the backend configuration file.

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

---
