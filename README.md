# CabinSpect 

### AI-Powered Industrial Electrical Cabinet Inspection System

CabinSpect BEAST is an AI-assisted electrical cabinet inspection platform designed to analyze industrial control panel images, extract component labels, compare physical installations against engineering references, and generate evidence-based quality inspection reports.

The platform combines computer vision, OCR, engineering document analysis, component matching, and automated QC evaluation.

## Features

### 1. AI-Powered Component Detection
- Electrical component identification.
- MCB and circuit breaker detection.
- PLC identification.
- Contactor and relay detection.
- Industrial power supply detection.
- Wiring and terminal inspection.
- Component bounding boxes and confidence scores.

### 2. Advanced OCR
- Electrical label extraction.
- Manufacturer and model identification.
- Voltage and current rating extraction.
- Terminal marking recognition.
- Multi-engine OCR support where configured.
- OCR confidence and text-region coordinates.

### 3. Engineering Drawing Comparison
- Engineering PDF processing.
- Expected component extraction.
- Physical component matching.
- Missing and extra component identification.
- Component position comparison.
- Rating and label verification.

### 4. Intelligent QC Evaluation
Each inspection criterion receives an evidence-based status:

| Status | Meaning |
|---|---|
| PASS | Criterion supported by sufficient evidence |
| FAIL | Confirmed mismatch or visible defect |
| REVIEW | Insufficient evidence or additional verification required |

QC results are calculated from actual inspection findings.

### 5. Evidence Management
- Original cabinet image preservation.
- Component-level evidence crops.
- Annotated inspection images.
- OCR evidence.
- Bounding-box visualization.
- Evidence-based inspection findings.

### 6. Reports
- PDF inspection reports.
- JSON inspection results.
- QC summaries.
- Component comparison findings.
- Evidence references.

## Technology Stack

| Category | Technologies |
|---|---|
| Backend | Python, Flask |
| Computer Vision | OpenCV |
| AI Models | Grounding DINO, Florence-2, Transformers |
| Object Detection | Ultralytics YOLO, when compatible weights are configured |
| OCR | EasyOCR and configured OCR engines |
| Image Processing | Pillow, NumPy |
| PDF Processing | PyMuPDF |
| Report Generation | ReportLab |
| Data Processing | Pandas |
| Deep Learning | PyTorch, TorchVision |
| Frontend | HTML, CSS, JavaScript |

Actual model availability depends on configuration, installed dependencies, hardware compatibility, and model weights.

## Project Structure

```text
CabinSpect/
│
├── app.py
├── detection.py
├── comparision.py
├── config.json
├── index.html
├── requirements.txt
│
├── models/
├── uploads/
├── reports/
├── temp/
│
└── README.md
```

## System Requirements

- Python 3.10 or newer, subject to dependency compatibility.
- macOS, Windows, or Linux.
- 16 GB RAM recommended for larger vision models.
- Internet connection for initial model downloads.
- Sufficient storage for model weights and inspection evidence.

Apple Silicon devices may use PyTorch MPS acceleration when supported.

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CabinSpect
```

### 2. Create a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

### 5. Open the interface

```text
http://127.0.0.1:8000
```

The application port depends on the Flask configuration.

# To download the required AI model run:
python download_model.py


## Inspection Workflow

1. Open the CabinSpect interface.
2. Upload an industrial cabinet image.
3. Upload the corresponding engineering PDF.
4. Run AI-powered inspection.
5. Review detected components and extracted text.
6. Examine engineering comparison findings.
7. Review PASS, FAIL, and REVIEW checklist results.
8. Inspect component-level evidence.
9. Export PDF and JSON reports.

## Hardware Acceleration

The application may use Apple Metal Performance Shaders (MPS) on supported Apple Silicon hardware.

Model loading and inference depend on:
- Available unified memory.
- Installed PyTorch version.
- Model compatibility.
- Model configuration.
- Available hardware acceleration.

Larger models may require more memory and longer processing times.

## Important Notes

- Detection confidence is not automatically equivalent to correctness.
- OCR text alone does not prove physical component installation.
- Electrical continuity, actual voltage, and hidden wiring require appropriate physical testing.
- Missing or ambiguous evidence should be marked REVIEW rather than assumed PASS.
- Custom YOLO detection requires suitable trained weights.

## Development Status

CabinSpect BEAST is under active development.

Model accuracy, component recognition, engineering comparison, evidence visualization, and report generation are continuously being improved and validated.

## License

All rights reserved unless a separate license is provided.

---

**CabinSpect BEAST — Intelligent Electrical Cabinet Inspection**
