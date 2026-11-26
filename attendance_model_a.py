import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace
from deepface.modules.exceptions import FaceNotDetected

os.environ.setdefault("yunet_score_threshold", "0.7")

KNOWN_FACES_DIR = Path("known_faces")
CSV_FILE = Path("attendance_log.csv")
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "yunet"
DISTANCE_THRESHOLD = 0.30
RECOGNIZE_EVERY_N_FRAMES = 8
COOLDOWN_SECONDS = 60

last_logged_time = {}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_known_embeddings(known_faces_dir):
    if not known_faces_dir.is_dir():
        raise FileNotFoundError(f"Known faces folder does not exist: {known_faces_dir}")

    known_db = {}
    image_paths = sorted(
        path
        for member_dir in known_faces_dir.iterdir()
        if member_dir.is_dir()
        for path in member_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in image_paths:
        try:
            faces = DeepFace.represent(
                img_path=str(image_path),
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
            )
        except Exception as exc:
            print(f"Skipping {image_path}: could not create face embedding ({exc})", file=sys.stderr)
            continue

        for face in faces:
            embedding = np.asarray(face["embedding"], dtype=np.float32)
            if embedding.ndim == 1 and np.linalg.norm(embedding) > 0:
                known_db.setdefault(image_path.parent.name, []).append(embedding)

    if not known_db:
        raise RuntimeError(
            f"No face embeddings could be loaded from {known_faces_dir}. "
            "Add clear face photos to a member-named subfolder."
        )

    return known_db


def recognize_face(embedding, known_db, threshold):
    embedding = np.asarray(embedding, dtype=np.float32)
    embedding_norm = np.linalg.norm(embedding)
    if embedding_norm == 0:
        return "Unknown", float("inf")

    best_name = "Unknown"
    best_distance = float("inf")
    for name, known_embeddings in known_db.items():
        for known_embedding in known_embeddings:
            known_norm = np.linalg.norm(known_embedding)
            if known_norm == 0 or known_embedding.shape != embedding.shape:
                continue

            cosine_similarity = np.dot(embedding, known_embedding) / (
                embedding_norm * known_norm
            )
            distance = 1.0 - float(np.clip(cosine_similarity, -1.0, 1.0))
            if distance < best_distance:
                best_name = name
                best_distance = distance

    if best_distance > threshold:
        return "Unknown", best_distance
    return best_name, best_distance


def log_attendance(name):
    now = time.time()
    if now - last_logged_time.get(name, 0) < COOLDOWN_SECONDS:
        return

    file_exists = CSV_FILE.is_file() and CSV_FILE.stat().st_size > 0
    with CSV_FILE.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["Student Name", "Date", "Time of Detection"])
        detected_at = datetime.now()
        writer.writerow([name, detected_at.strftime("%Y-%m-%d"), detected_at.strftime("%H:%M:%S")])

    last_logged_time[name] = now


def draw_label(frame, detection):
    x, y, width, height, label, is_known = detection
    color = (0, 200, 0) if is_known else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    cv2.putText(
        frame,
        label,
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def main():
    known_db = load_known_embeddings(KNOWN_FACES_DIR)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.", file=sys.stderr)
        cap.release()
        return 1

    frame_count = 0
    detections = []
    previous_time = time.time()
    fps = 0.0
    print("Starting attendance system. Press q to quit.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Could not read a frame from the webcam.", file=sys.stderr)
                break

            current_time = time.time()
            elapsed = current_time - previous_time
            if elapsed > 0:
                instant_fps = 1 / elapsed
                fps = 0.9 * fps + 0.1 * instant_fps if fps else instant_fps
                previous_time = current_time

            if frame_count % RECOGNIZE_EVERY_N_FRAMES == 0:
                current_detections = []
                try:
                    faces = DeepFace.represent(
                        img_path=frame,
                        model_name=MODEL_NAME,
                        detector_backend=DETECTOR_BACKEND,
                        enforce_detection=True,
                    )
                    for face in faces:
                        area = face.get("facial_area", {})
                        x, y = int(area.get("x", 0)), int(area.get("y", 0))
                        width, height = int(area.get("w", 0)), int(area.get("h", 0))
                        if width <= 0 or height <= 0:
                            continue

                        name, distance = recognize_face(
                            face["embedding"], known_db, DISTANCE_THRESHOLD
                        )
                        is_known = name != "Unknown"
                        label = f"{name} ({distance:.2f})" if is_known else "Unknown"
                        current_detections.append(
                            (x, y, width, height, label, is_known)
                        )
                        if is_known:
                            log_attendance(name)
                except FaceNotDetected:
                    current_detections = []
                except Exception as exc:
                    print(f"Face recognition failed: {exc}", file=sys.stderr)

                detections = current_detections

            for detection in detections:
                draw_label(frame, detection)

            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
