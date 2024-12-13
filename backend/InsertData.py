import os
import preprocessing
import ToEmbeddings
from database import Database
import cv2

# Database connection
db = Database(db_name="student_attendance", user="postgres", password="arslanbtw123")

# Create the students table (if it doesn't already exist)
db.create_table()

# Folder containing images
image_folder = "./images"
base_student_id = 100  # Starting point for student IDs
gender = "male"  # Default gender (can be customized)
department = "Computer Science"  # Default department (can be customized)

# Iterate through all images in the folder
for i, image_file in enumerate(os.listdir(image_folder)):
    # Check if the file is an image
    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Extract student name from the file name (without extension)
    name = os.path.splitext(image_file)[0]

    # Generate a unique student ID
    student_id = f"CS{base_student_id + i}"

    # Full path to the image
    image_path = os.path.join(image_folder, image_file)

    # Preprocess the image and generate embedding
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found at path: {image_path}. Skipping.")
        continue

    try:
        preprocessed_face = preprocessing.process_image(image)
        embedding = ToEmbeddings.get_face_embedding(preprocessed_face)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}. Skipping.")
        continue

    # Read the image file as binary
    with open(image_path, "rb") as file:
        image_binary = file.read()

    # Insert data into the database
    try:
        db.insert_student(name, student_id, gender, department, image_binary, embedding)
        print(f"Inserted: {name} (ID: {student_id})")
    except Exception as e:
        print(f"Error inserting data for {name}: {e}. Skipping.")

# Close the database connection
db.close()
print("inserted all records successfully.")
