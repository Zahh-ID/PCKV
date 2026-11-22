"""Projection-profile line segmentation baseline; not robust to multi-column or ruled pages."""
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class LineCrop:
    image: np.ndarray
    x: int
    y: int
    width: int
    height: int

def decode_and_segment(data: bytes) -> tuple[list[LineCrop], list[str]]:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('File bukan gambar yang dapat dibaca')
    h, w = image.shape[:2]
    if h * w > 20_000_000:
        raise ValueError('Gambar terlalu besar (maksimal 20 megapiksel)')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, inverted = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = inverted > 0
    counts = np.count_nonzero(ink, axis=1)
    active = counts > max(2, int(w * 0.004))
    runs = []
    start = None
    for y, present in enumerate(active):
        if present and start is None:
            start = y
        if start is not None and (not present or y == h-1):
            end = y if not present else y+1
            if end-start >= 5:
                runs.append((start,end))
            start = None
    merged = []
    gap = max(3,int(h*0.003))
    for a,b in runs:
        if merged and a-merged[-1][1]<=gap:
            merged[-1][1]=b
        else:
            merged.append([a,b])
    result = []
    for a,b in merged:
        ys,xs=np.where(ink[a:b])
        if len(xs)<8:
            continue
        x0=int(xs.min()); x1=int(xs.max()+1)
        y0=a+int(ys.min()); y1=a+int(ys.max()+1)
        pad=max(5,int((y1-y0)*0.2))
        l=max(0,x0-pad);r=min(w,x1+pad);t=max(0,y0-pad);d=min(h,y1+pad)
        result.append(LineCrop(image=image[t:d,l:r],x=x0,y=y0,width=x1-x0,height=y1-y0))
    if not result:
        raise ValueError('Tidak ditemukan baris tulisan; gunakan gambar yang lebih jelas.')
    warnings=['Segmentasi proyeksi horizontal masih baseline; deskew belum diimplementasikan.']
    if len(result)>35:
        warnings.append('Terlalu banyak baris; kemungkinan segmentasi keliru.')
    return result,warnings

def indent_levels(lines: list[LineCrop]) -> list[int]:
    if not lines:
        return []
    base=min(line.x for line in lines)
    unit=max(10.0,float(np.median([line.height for line in lines]))*1.3)
    return [max(0,min(8,round((line.x-base)/unit))) for line in lines]
