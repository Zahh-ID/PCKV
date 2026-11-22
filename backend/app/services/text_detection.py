"""Optional PP-OCRv4 text detection for phone photos of paper."""

import threading
from functools import lru_cache

import cv2
import numpy as np


_lock = threading.Lock()
_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@lru_cache(maxsize=1)
def _network(model_path: str):
    return cv2.dnn.readNetFromONNX(model_path)


def _boxes(image: np.ndarray, model_path: str) -> list[tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    scale = min(1.0, 1280 / max(h, w))
    size = (max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    normalized = (resized.astype(np.float32) / 255 - _mean) / _std
    tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
    with _lock:
        net = _network(model_path)
        net.setInput(tensor)
        probabilities = net.forward()[0, 0]
    contours, _ = cv2.findContours(
        (probabilities > 0.28).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    sy, sx = h / probabilities.shape[0], w / probabilities.shape[1]
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if min(width, height) < 4 or cv2.contourArea(contour) < 12:
            continue
        if float(probabilities[y:y + height, x:x + width].max()) < 0.53:
            continue
        boxes.append((round(x * sx), round(y * sy), round((x + width) * sx), round((y + height) * sy)))
    return boxes


def _merge_rows(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    size = float(np.median([bottom - top for _, top, _, bottom in boxes]))
    groups: list[list[tuple[int, int, int, int]]] = []
    for box in sorted(boxes, key=lambda box: box[1] + box[3]):
        center = (box[1] + box[3]) / 2
        if groups and abs(center - np.median([(b[1] + b[3]) / 2 for b in groups[-1]])) <= size * 0.67:
            groups[-1].append(box)
        else:
            groups.append([box])
    return [
        (min(b[0] for b in group), min(b[1] for b in group),
         max(b[2] for b in group), max(b[3] for b in group))
        for group in groups
    ]


def detect_rows(image: np.ndarray, model_path: str):
    boxes = _boxes(image, model_path)
    vertical = sum(bottom - top > 2 * (right - left) for left, top, right, bottom in boxes)
    rotation = 0
    if len(boxes) >= 2 and vertical > len(boxes) * 0.7:
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_boxes = _boxes(rotated, model_path)
        horizontal = sum(right - left > 2 * (bottom - top) for left, top, right, bottom in rotated_boxes)
        if horizontal >= 2 and horizontal > len(rotated_boxes) / 2:
            image, boxes, rotation = rotated, rotated_boxes, 90
    return image, _merge_rows(boxes), rotation
