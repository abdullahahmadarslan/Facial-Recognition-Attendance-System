import cv2
import time
import numpy as np
import preprocessing  # Import the preprocessing module
import ToEmbeddings  # Import the embeddings module
from database import Database
from cosine_similarity import match_embedding
import requests

# Base URL of the Flask API
API_URL = "http://127.0.0.1:5000/mark_attendance"

# Cache to track recently marked students
recently_marked = {}  

# Load the pre-trained DNN model for face detection
model = cv2.dnn.readNetFromCaffe( #load the pre-trained model in the Caffe format
    "./model/deploy.prototxt", #text file defining the network's architecture (layers, operations, etc)
    "./model/res10_300x300_ssd_iter_140000.caffemodel" # Single Shot MultiBox Detector (SSD) model weights with a ResNet-10 based backbone, specifically trained to detect faces
)

def mark_attendance(student_id,student_name,similarity):
    current_time = time.time()

    # Skip if the student was marked within the last 10 seconds
    if student_id in recently_marked and current_time - recently_marked[student_id] < 1:
        print(f"Skipping duplicate attendance for student_id={student_id}, student_name={student_name}")
        return

    # Update the cache and send the request
    recently_marked[student_id] = current_time
    try:
        response = requests.post(API_URL, json={"student_id": student_id})
        if response.status_code == 201:
            print(f"Attendance marked successfully for student_id={student_id}, student_name={student_name}, similarity= {similarity}")
        elif response.status_code == 202:
            print(f"Attendance already marked for student_id={student_id}, student_name={student_name}, similarity= {similarity}")
        else:
            print(f"Error: {response.status_code}, {response.text}, similarity={similarity}")
    except Exception as e:
        print(f"Error while sending request: {e}")



def capture():
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)   

    # Database connection
    db = Database(db_name="student_attendance", user="postgres", password="arslanbtw123")
    db.create_table()  # Ensure table exists

    # Fetch stored embeddings from the database
    db.cursor.execute("SELECT id, name, embedding FROM students")
    stored_data = db.cursor.fetchall()
    stored_embeddings = [np.array(record[2]) for record in stored_data]  # Convert embeddings to numpy arrays
    student_ids = [record[0] for record in stored_data]
    student_names = [record[1] for record in stored_data]

    # Buffer time setup
    last_capture_time = time.time()
    buffer_time = 2  # Buffer time in seconds to avoid duplicate captures

    while True:
        # Capture the frame
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare the frame for the model
        # Increase scale factor for better distant face detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))  # Larger input size
        model.setInput(blob)
        detections = model.forward() 
        # detections : all the detected faces in the current frame , 4D numpy array with the shape: (1, 1, num_detections, 7)
        # 1, 1: Fixed dimensions for batch size.
        # num_detections: Number of detected faces in the frame.
        # 7: For each detection, there are 7 values:
        # Index 0: Always 0.
        # Index 1: Always 0.
        # Index 2: Confidence score (probability that the detection is a face).
        # Indexes 3-6: Bounding box coordinates:
        # (startX, startY, endX, endY) in the range [0, 1]

        # Quality check variables
        # frame.shape : (height, width, channels)
        frame_height, frame_width = frame.shape[:2] #excluding 2 index i.e. channels of the frame 
        face_boxes = []
        face_center = None  # Variable to track the center of the face

        for i in range(detections.shape[2]): # detections.shape[2]: The number of detections (i.e., how many faces the model detected in the frame).
            confidence = detections[0, 0, i, 2]
            # 0, 0: Refers to the first batch and the first class (we are using face detection, so there is only one class, "face").
            # i: Refers to the i-th detection. Each detection corresponds to a face the model found in the frame.
            # 2: Refers to the confidence score of that detection (the probability that the detected object is a face).

            # Apply a high confidence threshold to ensure it's a strong detection
            if confidence > 0.7:  # Adjusted confidence threshold
                # Get the bounding box coordinates
                box = detections[0, 0, i, 3:7] * [frame_width, frame_height, frame_width, frame_height]
                (x, y, x1, y1) = box.astype("int")
                # Increase the top and bottom boundaries to include more forehead and chin
                # Extend boundaries to include more forehead, chin, and sides of the face
                # y = max(0, y + 20)  # Extend 20 pixels upwards for forehead
                y1 = min(frame_height, y1 + 5)  # Extend 20 pixels downwards for chin
                # x = max(0, x - 20)  # Extend 20 pixels left for more of the face
                # x1 = min(frame_width, x1 + 20)  # Extend 20 pixels right for more of the face

                # Calculate the area of the detected face
                face_area = (x1 - x) * (y1 - y) 
                # (x1 - x) → Width of the bounding box.
                # (y1 - y) → Height of the bounding box.
                # face_area is the area of the detected face.

                # Quality check: Ensure the face is sufficiently large (lower the threshold for smaller faces)
                if face_area > 0.05 * frame_width * frame_height:  # Reduced face area threshold to 5%
                    face_boxes.append((x, y, x1, y1))

                    # Track face center for centering check
                    face_center = ((x + x1) // 2, (y + y1) // 2) # (x,y) central point -> for bounded box of face

        # Process and crop each face
        for (x, y, x1, y1) in face_boxes:
            # Check if the bounding box is valid (non-zero area)
            if (x1 - x) > 0 and (y1 - y) > 0:
                cropped_face = frame[y:y1, x:x1]  # Crop face
                preprocessed_face = preprocessing.process_image(cropped_face)  # Preprocess the cropped face
                new_embedding = ToEmbeddings.get_face_embedding(preprocessed_face)  # Generate embedding

                # Match embedding with stored embeddings
                matches = match_embedding(new_embedding, stored_embeddings, threshold=0.92)

                if matches:
                    for idx, similarity in matches:
                        student_id = student_ids[idx]
                        student_name = student_names[idx]

                        # Mark attendance for the detected student
                        mark_attendance(student_id,student_name,similarity)
                        
                        # print(f"Match found: Student ID: {student_id}, Name: {student_name}, Similarity: {similarity:.2f}")
                else:
                    print("No match found for this face.")

        # Check if the bounded box's face is mostly centered (within a region around the center of the frame)
        if face_center:
            frame_center = (frame_width // 2, frame_height // 2)
            distance_from_center = np.linalg.norm(np.array(face_center) - np.array(frame_center))

            # Set a threshold for how far from the center the face can be
            if distance_from_center < frame_width * 0.2:  # Face should be within 20% of the frame width from the center
                # Capture image if the face is centered and not too close
                if time.time() - last_capture_time > buffer_time:
                    last_capture_time = time.time()
                    print("Capturing image for attendance...")
            else:
                print("Face not centered enough.")
        else:
            print("No face detected.")

        # Show the frame with bounding boxes
        for (x, y, x1, y1) in face_boxes:
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)  # Draw bounding box

        cv2.imshow("Face Detection and Processing", frame)

        # Press 'q' to manually exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Release the webcam and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# running the face capturing system
if __name__ == "__main__":
    capture()