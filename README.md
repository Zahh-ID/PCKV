# CodeScribe — Handwritten Python OCR & Syntax Analysis

Semester project **Pengolahan Citra dan Computer Vision**. A proof-of-concept application that recognizes handwritten Python code from scanned paper using an OpenCV baseline and Microsoft's pretrained TrOCR, then analyzes the reconstructed code using Python AST and tokenization.

> **Research caveat:** The displayed rubric percentage is experimental rule coverage, **not** the percentage of overall program correctness or an authoritative student grade. OCR errors can change syntax; verify manually.

## Project structure
- `backend/` FastAPI, OpenCV line segmentation, TrOCR, syntax analyzer, tests
- `frontend/` React, TypeScript, Vite application
- `docs/` research methodology and roadmap

## Backend setup (Python 3.11 recommended)
```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -r requirements-ocr.txt
uvicorn app.main:app --reload
```

To test the user interface without downloading model weights:
```powershell
$env:CODESCRIBE_MOCK_OCR="1"
uvicorn app.main:app --reload
```
The mock mode is visibly labeled and **does not produce real OCR**.

## Frontend
```powershell
cd frontend
npm install
npm run dev
```
Open http://localhost:5173. API documentation: http://localhost:8000/docs.

## Tests
```powershell
cd backend
pytest -q
```

## Current limitations
- Line segmentation is a simple horizontal projection heuristic and is not robust against deskew, ruled paper, multi-column layouts, or camera perspective.
- The geometric indentation reconstruction is experimental.
- Pretrained TrOCR was designed for handwriting lines, not necessarily Python code operators.
- Syntax scoring requires research validation, and code is never executed by the API.

See `docs/METHODOLOGY.md` and `docs/ROADMAP.md`.
