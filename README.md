# Smart Attendance System (SAIA 1023)

Automated classroom attendance prototype built with Python and computer vision that detects and recognizes group members' faces, displays labeled bounding boxes on screen, and logs attendance records to a CSV file.

## Team Members (Group A)

- Muhammad Shahizz Ifwat Bin Shahrizal — Core Pipeline & Deep Learning / CNN Model
- Muaz Hilman bin Sudin — Baseline Model (Haar Cascade / HOG) & Demo/Slides
- Massafiya binti Khalid — Benchmark Comparison & Analysis
- Chee Joe Yee — Dataset & Edge-Case / Limitation Testing
- Muhammad Adib Farhan bin Ahmad Kusairi — Report Compilation & Q&A Prep

## Project Structure

```text
smart-attendance/
├── known_faces/
│   ├── Shahizz/        # Put 2-3 clear photos of yourself here first
│   ├── Muaz/
│   ├── Safiya/
│   ├── Chee/
│   └── Adib/
├── attendance_log.csv
└── attendance_model_a.py
```

- `attendance_model_a.py` — Main real-time detection, recognition, bounding-box display, and CSV logger (Model A)
- `known_faces/` — Labeled reference images organized by group member
- `attendance_log.csv` — Attendance log with `Student Name`, `Date`, and `Time of Detection` columns

## Setup & Usage

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

2. Add 2-3 clear reference photos to `known_faces/Shahizz/`. Add other members' photos to their own labeled folders as they become available.
3. Install the Python dependencies required by the implementation.
4. Run the attendance program:

   ```bash
   python attendance_model_a.py
   ```
