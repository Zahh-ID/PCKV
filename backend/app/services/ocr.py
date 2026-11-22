"""Lazy model loading. Mock output is explicitly marked as demo-only."""
import os
import threading
import cv2
from PIL import Image
from app.services.image_processing import LineCrop
from app.services.code_transcription import choose_line, format_token_spacing, line_parses

_lock=threading.Lock()
_model=None
_processor=None

def _load():
    global _model,_processor
    if _model is not None:
        return _processor,_model
    with _lock:
        if _model is None:
            try:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            except ImportError as exc:
                raise RuntimeError('Instal dependency OCR: pip install -r requirements-ocr.txt') from exc
            name=os.getenv('CODESCRIBE_MODEL','microsoft/trocr-base-handwritten')
            _processor=TrOCRProcessor.from_pretrained(name)
            _model=VisionEncoderDecoderModel.from_pretrained(name)
            _model.eval()
    return _processor,_model

def recognize(lines: list[LineCrop], levels: list[int]) -> tuple[list[str], list[str]]:
    if os.getenv('CODESCRIBE_MOCK_OCR')=='1':
        mock = ['# DEMO OCR: placeholder, bukan hasil model'] * len(lines)
        return mock, mock
    processor,model=_load()
    import torch
    raw = []

    def decode(images, max_tokens=128, **generation):
        inputs = processor(images=images, return_tensors='pt')
        tokens = model.generate(inputs.pixel_values, max_new_tokens=max_tokens, use_cache=True, **generation)
        return processor.batch_decode(tokens, skip_special_tokens=True)

    images = [Image.fromarray(cv2.cvtColor(line.image, cv2.COLOR_BGR2RGB)) for line in lines]
    with torch.inference_mode():
        for start in range(0, len(images), 4):
            raw.extend(decode(images[start:start + 4]))
        selected = raw.copy()
        retries = []
        for index, original in enumerate(raw):
            opens_block = index + 1 < len(levels) and levels[index + 1] > levels[index]
            if not line_parses(original, opens_block):
                retries.append((index, opens_block))
        for start in range(0, len(retries), 2):
            batch = retries[start:start + 2]
            variants = []
            for index, _ in batch:
                gray = cv2.cvtColor(lines[index].image, cv2.COLOR_BGR2GRAY)
                _, contrast = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                variants.extend([images[index], Image.fromarray(contrast).convert("RGB")])
            max_tokens = min(128, max(32, max(len(processor.tokenizer.tokenize(raw[i])) for i, _ in batch) + 12))
            alternatives = decode(variants, max_tokens, num_beams=3, num_return_sequences=3)
            for offset, (index, opens_block) in enumerate(batch):
                selected[index] = choose_line(raw[index], alternatives[offset * 6:(offset + 1) * 6], opens_block)
    return [format_token_spacing(text) for text in selected], raw
