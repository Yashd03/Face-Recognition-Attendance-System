import cv2
from simple_facerec import SimpleFacerec
import csv
import datetime

# Initialize SimpleFacerec
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
cap = cv2.VideoCapture(0)

# Create or open a CSV file for attendance
csv_filename = "attendance.csv"
attendance_recorded = set()  # Keep track of attendance already recorded in this run

while True:
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        continue  # Skip this iteration if the frame is empty

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        if name not in attendance_recorded:
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            # Get the current date and time
            current_datetime = datetime.datetime.now()
            date = current_datetime.date()
            time = current_datetime.time()

            # Write the attendance to the CSV file
            with open(csv_filename, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([name, date, time])

            attendance_recorded.add(name)  # Mark this person as recorded

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
