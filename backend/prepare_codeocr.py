"""Prepare line-image references from the public CodeOCR research archive."""

import argparse
import csv
import io
import json
import math
from difflib import SequenceMatcher
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageOps


VALIDATION = {0, 11, 18, 26, 39}
TEST = {4, 10, 17, 24, 28, 43, 49, 52}


def similarity(left, right):
    parts = ("".join(text.casefold().split()) for text in (left, right))
    return SequenceMatcher(None, *parts, autojunk=False).ratio()


def align(refs, detections):
    """Align ordered boxes to manual references, allowing missing and extra boxes."""
    costs = np.full((len(refs) + 1, len(detections) + 1), np.inf)
    steps = {}
    costs[0, 0] = 0
    for i in range(len(refs) + 1):
        for j in range(len(detections) + 1):
            if i:
                costs[i, j] = costs[i - 1, j] + 0.55
                steps[i, j] = (i - 1, j)
            if j and costs[i, j - 1] + 0.55 < costs[i, j]:
                costs[i, j] = costs[i, j - 1] + 0.55
                steps[i, j] = (i, j - 1)
            if i and j:
                score = similarity(refs[i - 1], detections[j - 1]["content"])
                match = costs[i - 1, j - 1] + 1 - score
                if score >= 0.48 and match < costs[i, j]:
                    costs[i, j] = match
                    steps[i, j] = (i - 1, j - 1)
    paired = []
    current = (len(refs), len(detections))
    while current != (0, 0):
        i, j = current
        previous = steps[current]
        if previous == (i - 1, j - 1):
            paired.append((i - 1, detections[j - 1]))
        current = previous
    return list(reversed(paired))


def crop_line(image, bbox):
    box = np.array(bbox, dtype=np.float32).reshape(4, 2)
    width = math.ceil(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    height = math.ceil(max(np.linalg.norm(box[1] - box[2]), np.linalg.norm(box[3] - box[0])))
    if width < 4 or height < 4:
        return None
    padding = max(3, round(height * 0.17))
    target = np.float32([
        [padding, padding], [padding + width, padding],
        [padding + width, padding + height], [padding, padding + height],
    ])
    matrix = cv2.getPerspectiveTransform(box, target)
    return cv2.warpPerspective(
        image, matrix, (width + 2 * padding, height + 2 * padding),
        borderMode=cv2.BORDER_REPLICATE,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "lines").mkdir(exist_ok=True)
    (args.output / "pages").mkdir(exist_ok=True)
    manifest = []
    page_refs = []
    with ZipFile(args.archive) as archive:
        refs = list(csv.DictReader(io.StringIO(archive.read("dataset/rawdata.csv").decode("utf-8-sig"))))
        payloads = json.loads(archive.read("dataset/apipayload.json"))
        for payload in payloads:
            page_id = int(payload["image_id"])
            if page_id >= 45 and page_id not in TEST:
                continue
            split = "test" if page_id in TEST else "validation" if page_id in VALIDATION else "train"
            page_reference = refs[page_id]["Ground Truth"].strip("\r\n")
            ref_lines = [text.strip() for text in page_reference.splitlines() if text.strip()]
            detections = payload["Azure"]["readResult"]["pages"][0]
            source = archive.read(f"dataset/images/{page_id}.jpg")
            photo = ImageOps.exif_transpose(Image.open(io.BytesIO(source))).convert("RGB")
            image = np.asarray(photo)
            size = (int(detections["width"]), int(detections["height"]))
            if photo.size != size:
                raise ValueError(f"Page {page_id} OCR coordinates do not match the image.")
            page_path = f"pages/{page_id:02}.jpg"
            if split != "train":
                photo.save(args.output / page_path, quality=94)
                page_refs.append({"id": page_id, "split": split, "image": page_path, "text": page_reference})
            pairs = align(ref_lines, detections["lines"])
            for index, detection in pairs:
                crop = crop_line(image, detection["boundingBox"])
                if crop is None:
                    continue
                name = f"lines/{page_id:02}-{index + 1:02}.jpg"
                Image.fromarray(crop).save(args.output / name, quality=94)
                manifest.append({
                    "page_id": page_id, "split": split, "image": name,
                    "line": index + 1, "text": ref_lines[index],
                })
            print(f"Page {page_id:02} {split}: {len(pairs)}/{len(ref_lines)} matched lines", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "pages.json").write_text(json.dumps(page_refs, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
