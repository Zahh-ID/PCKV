"""Deskew, segment, and reconstruct indentation for single-column code images."""
from dataclasses import dataclass
import os
import cv2
import numpy as np

from app.services.text_detection import detect_rows

@dataclass
class LineCrop:
    image: np.ndarray
    x: int
    y: int
    width: int
    height: int
    source_box: tuple[int, int, int, int] | None = None

def _ink_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def _deskew(
    image: np.ndarray, mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray | None]:
    h, w = mask.shape
    scale = min(1.0, 800 / max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    sample = cv2.resize(mask, size, interpolation=cv2.INTER_AREA) if scale < 1 else mask
    y, x = np.nonzero(sample >= 128)
    if len(x) < 20:
        return image, mask, 0.0, None
    x = x - x.mean()
    y = y - y.mean()

    def score(angle: float) -> float:
        radians = np.deg2rad(angle)
        rows = np.rint(y * np.cos(radians) - x * np.sin(radians)).astype(np.int32)
        profile = np.bincount(rows - rows.min())
        return float(np.dot(profile, profile))

    angles = np.arange(-12.0, 12.01, 0.5)
    angle = float(max(angles, key=score))
    if abs(angle) < 0.5 or score(angle) < 1.04 * score(0):
        return image, mask, 0.0, None
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    width = int(np.ceil(w * abs(matrix[0, 0]) + h * abs(matrix[0, 1])))
    height = int(np.ceil(w * abs(matrix[0, 1]) + h * abs(matrix[0, 0])))
    matrix[:, 2] += [(width - w) / 2, (height - h) / 2]
    rotated = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    rotated_mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
    return rotated, rotated_mask, angle, cv2.invertAffineTransform(matrix)


def decode_and_segment(data: bytes) -> tuple[list[LineCrop], list[str], float]:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('File bukan gambar yang dapat dibaca')
    h, w = image.shape[:2]
    if h * w > 20_000_000:
        raise ValueError('Gambar terlalu besar (maksimal 20 megapiksel)')
    model_path = os.getenv("CODESCRIBE_TEXT_DETECTOR")
    if not model_path:
        return _projection_segment(image)
    try:
        detected, boxes, angle = detect_rows(image, model_path)
    except cv2.error as exc:
        raise RuntimeError('Model deteksi baris tidak dapat dimuat atau dijalankan.') from exc
    try:
        projection = _projection_segment(image)
    except ValueError:
        if not boxes:
            raise
    else:
        if not angle and len(projection[0]) >= 0.75 * len(boxes):
            return projection
    height, width = detected.shape[:2]
    crops = []
    for left, top, right, bottom in boxes:
        pad = max(4, round((bottom - top) * 0.3))
        crop = detected[max(0, top - pad):min(height, bottom + pad),
                        max(0, left - pad):min(width, right + pad)]
        source_box = (w - bottom, left, bottom - top, right - left) if angle else None
        crops.append(LineCrop(crop, left, top, right - left, bottom - top, source_box))
    warnings = ['Terlalu banyak baris; kemungkinan segmentasi keliru.'] if len(crops) > 35 else []
    return crops, warnings, float(angle)


def _projection_segment(image: np.ndarray) -> tuple[list[LineCrop], list[str], float]:
    h, w = image.shape[:2]
    source_height, source_width = h, w
    image, mask, rotation, to_source = _deskew(image, _ink_mask(image))
    ink = mask > 0
    h, w = ink.shape
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
        source_box = None
        if to_source is not None:
            corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=float)
            points = cv2.transform(corners[None, :, :], to_source)[0]
            left, top = np.maximum(0, np.floor(points.min(axis=0))).astype(int)
            right, bottom = np.minimum([source_width, source_height], np.ceil(points.max(axis=0))).astype(int)
            source_box = (int(left), int(top), int(right - left), int(bottom - top))
        result.append(LineCrop(image[t:d, l:r], x0, y0, x1 - x0, y1 - y0, source_box))
    if not result:
        raise ValueError('Tidak ditemukan baris tulisan; gunakan gambar yang lebih jelas.')
    warnings = []
    if len(result)>35:
        warnings.append('Terlalu banyak baris; kemungkinan segmentasi keliru.')
    return result, warnings, rotation

def indent_levels(lines: list[LineCrop]) -> list[int]:
    if not lines:
        return []
    height = float(np.median([line.height for line in lines]))
    tolerance = max(3.0, height * 0.4)
    groups: list[list[int]] = []
    for x in sorted(line.x for line in lines):
        if groups and x - np.median(groups[-1]) <= tolerance:
            groups[-1].append(x)
        else:
            groups.append([x])
    margins = np.array([np.median(group) for group in groups])
    # Python nesting depth follows the order of distinct margins. The physical
    # gap for one level can vary a lot with pen size and camera distance.
    return [min(8, int(np.argmin(abs(margins - line.x)))) for line in lines]
