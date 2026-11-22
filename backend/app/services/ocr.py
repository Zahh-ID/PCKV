"""Lazy model loading. Mock output is explicitly marked as demo-only."""
import os
import threading
import cv2
from PIL import Image
from app.services.image_processing import LineCrop

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

def recognize(lines: list[LineCrop]) -> list[str]:
    if os.getenv('CODESCRIBE_MOCK_OCR')=='1':
        return ['# DEMO OCR: placeholder, bukan hasil model']*len(lines)
    processor,model=_load()
    import torch
    recognized=[]
    with torch.inference_mode():
        for line in lines:
            rgb=cv2.cvtColor(line.image,cv2.COLOR_BGR2RGB)
            inputs=processor(images=Image.fromarray(rgb),return_tensors='pt')
            predicted=model.generate(inputs.pixel_values,max_new_tokens=128)
            recognized.append(processor.batch_decode(predicted,skip_special_tokens=True)[0])
    return recognized
