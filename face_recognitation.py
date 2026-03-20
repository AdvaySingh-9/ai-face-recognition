import cv2
import os
import numpy as np
import pickle

dataset_path = "dataset"
model_path = "face_recognizer.yml"
label_map_path = "label_map.pkl"
haarcacade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

os.makedirs(dataset_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(haarcacade_path)

FACE_SIZE = (200, 200)

def capture_faces(name):
    person_path = os.path.join(dataset_path, name)
    os.makedirs(person_path, exist_ok=True)

    existing_files = len(os.listdir(person_path))
    count = existing_files

    cap = cv2.VideoCapture(0)

    print(f"Capturing images for {name}. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, FACE_SIZE)

            cv2.imwrite(os.path.join(person_path, f"{count}.jpg"), face)
            count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Capturing Training Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {count} images for {name}.")

    
recognizer = cv2.face.LBPHFaceRecognizer_create()


def train_model():
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, name)
        if not os.path.isdir(person_path):
            continue

        label_map[name] = current_label

        for file in os.listdir(person_path):
            if not file.endswith((".jpg", ".png")):
                continue

            img_path = os.path.join(person_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, FACE_SIZE)
            faces.append(img)
            labels.append(current_label)

        current_label += 1

        if not faces:
            print("No traning data found. Please capture faces first.")
            return
        
        recognizer.train(faces, np.array(labels))
        recognizer.write(model_path)

        with open(label_map_path, "wb") as f:
            pickle.dump(label_map, f)

        print("Model trained and saved successfully.")


def recognize_faces():
    if not os.path.exists(model_path):
        print("Please train the model first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    with open(label_map_path, "rb") as f:
        label_map = pickle.load(f)

    # reverse the label map
    label_map = {v: k for k, v in label_map.items()}

    cap = cv2.VideoCapture(0)
    print("Starting face recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, FACE_SIZE)

            label, confidence = recognizer.predict(face)

            if confidence < 70:
                name = label_map.get(label, "Unknown")
                display_text = f"{name} ({confidence:.2f})"
            else:
                display_text = "Animal: Crow"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, display_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    while True:
        print("""
1. Capture Faces
2. Train Model
3. Recognize Faces
4. Exit
""")


        choice = input("Enter your choice: ")
    
        if choice == '1':
            name = input("Enter the name of the person: ")
            capture_faces(name)
        elif choice == '2':
            train_model()
        elif choice == '3':
            recognize_faces()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()