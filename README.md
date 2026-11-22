# CodeScribe — Handwritten Python OCR & Syntax Analysis

Semester project **Pengolahan Citra dan Computer Vision**. A proof-of-concept application that recognizes handwritten Python code from scanned paper using an OpenCV baseline and Microsoft's pretrained TrOCR, then analyzes the reconstructed code using Python AST and tokenization.

The results show how many applicable syntax checks passed. After an image upload,
the web app asks the reviewer to compare the transcription and indentation with
the original before running syntax analysis.

For lines with invalid syntax or suspicious standalone expressions, OCR checks
alternative readings of the original and a high-contrast crop. A replacement must pass the line syntax check
and preserve the numbers and strings recognized in the first pass. The review
panel exposes the original reading when an alternative changes the text.

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

### Fine-tune TrOCR on synthetic Python lines

From `backend/` with the OCR dependencies installed, run:

```powershell
python finetune_ocr.py --base-model microsoft/trocr-base-handwritten --output ../../pckv-experiment
```

The CPU experiment renders Python lines with Windows fonts and learns a rank-8
update to the TrOCR output projection while keeping the other weights frozen.
Validation loss selects the training epoch, and character error rate on generated
validation lines selects the update strength. The baseline and adapted models
then run on the same held-out text and font. The output includes test images and
`results.json` with predictions, character error rate, and exact-line counts.
The script exports a model when an update improves generated validation results.
This experiment measures synthetic line recognition.

### Test and fine-tune on real paper photos

Download the dataset ZIP from [CodeOCR](https://github.com/mdoumbouya/codeocr).
The preparation script rectifies line crops using published Azure boxes and
aligns them with the manually transcribed source. The OCR service does all
inference locally. Run these commands from `backend/` with the OCR dependencies
installed.

```powershell
python prepare_codeocr.py ./code-ocr-dataset-20240516.zip ../../codeocr-lines
python finetune_codeocr.py ../../codeocr-lines --base-model microsoft/trocr-base-handwritten --output ../../codeocr-run
```

The splits use distinct page IDs. Writer metadata is unavailable. Validation
selects the model using generated character error with whitespace excluded, and
the JSON report includes strict text matches and numeric-token edits on held-out
lines.

For phone photos, set `CODESCRIBE_TEXT_DETECTOR` to a local copy of the
[RapidOCR PP-OCRv4 ONNX detection model](https://huggingface.co/spaces/RapidAI/RapidOCR/blob/main/models/text_det/ch_PP-OCRv4_det_infer.onnx)
before starting Uvicorn. This optional detector uses OpenCV and can recover rows
when page shadows or notebook rules cause horizontal projection to merge them.

To use the exported model in the API, set its output directory before starting Uvicorn.

```powershell
$env:CODESCRIBE_MODEL = (Resolve-Path ../../codeocr-run).Path
uvicorn app.main:app
```

## Frontend
```powershell
cd frontend
npm install
npm run dev
```
Open http://localhost:5173. API documentation: http://localhost:8000/docs.

## Example uploads

- [Handwritten Python example](docs/examples/handwritten-python.png) is an
  eight-line, AI-generated benchmark image with decimals, a conditional, and
  an indented block.
- For a real paper example, upload `dataset/images/49.jpg` from the
  [CodeOCR dataset archive](https://github.com/mdoumbouya/codeocr/blob/96a3e0288e34ab18dbf582ad75b6bfe6bd3d1963/code-ocr-dataset-20240516.zip).
  The hybrid layout found all five lines in the local test. Character error
  was 16.8% and the raw transcription required syntax corrections.

## Tests
```powershell
cd backend
pytest -q
```

## Current limitations
- Deskew searches rotations up to 12° in either direction before segmenting lines by horizontal projection. Ruled paper, multi-column layouts, and camera perspective can still disrupt segmentation.
- Indentation reconstruction groups similar left margins and orders them into nesting levels. The editor allows corrections before review.
- Pretrained TrOCR was designed for handwriting lines, not necessarily Python code operators.
- Syntax scoring requires research validation, and code is never executed by the API.

See `docs/METHODOLOGY.md` and `docs/ROADMAP.md`.
