import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.models import AnalyzeRequest, ImageResponse, OCRLine, SyntaxResult
from app.services.syntax import analyze_python
from app.services.image_processing import decode_and_segment, indent_levels
from app.services.ocr import recognize

app=FastAPI(title='CodeScribe API',version='0.1.0')
origins=os.getenv('CODESCRIBE_ALLOWED_ORIGINS','http://localhost:5173').split(',')
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=['GET','POST'],allow_headers=['*'])

@app.get('/api/health')
def health():
    return {'status':'ok','ocr_mode':'mock' if os.getenv('CODESCRIBE_MOCK_OCR')=='1' else 'trocr'}

@app.post('/api/analyze',response_model=SyntaxResult)
def analyze(payload: AnalyzeRequest):
    return analyze_python(payload.code)

@app.post('/api/process-image',response_model=ImageResponse)
async def process_image(image: UploadFile=File(...)):
    if image.content_type not in ('image/png','image/jpeg','image/webp'):
        raise HTTPException(status_code=415,detail='Gunakan PNG, JPG atau WebP')
    data=await image.read(8_000_001)
    if len(data)>8_000_000:
        raise HTTPException(status_code=413,detail='Maksimum 8 MB')
    try:
        crops,warnings=decode_and_segment(data)
        levels=indent_levels(crops)
        recognized=recognize(crops)
        if os.getenv('CODESCRIBE_MOCK_OCR')=='1':
            warnings.append('DEMO MODE: teks OCR adalah placeholder')
        warnings.append('Indentasi bersifat heuristik dan harus diverifikasi manusia')
        code='\n'.join(('    '*level)+txt for txt,level in zip(recognized,levels))
        parsed=analyze_python(code)
        lines=[OCRLine(line_number=i+1,text=txt,x=c.x,y=c.y,width=c.width,height=c.height,indent_level=level)
               for i,(c,txt,level) in enumerate(zip(crops,recognized,levels))]
        return ImageResponse(code=code,lines=lines,syntax=parsed,warnings=warnings)
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503,detail=str(exc)) from exc
