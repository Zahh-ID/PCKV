import cv2
import numpy as np

from app.services import image_processing
from app.services.text_detection import _merge_rows


def test_detector_merges_fragments_on_the_same_row():
    assert _merge_rows([(12, 10, 45, 31), (65, 14, 100, 34), (8, 57, 75, 78)]) == [
        (12, 10, 100, 34), (8, 57, 75, 78)
    ]


def test_detector_recovers_rows_when_projection_finds_no_ink(monkeypatch):
    page = np.full((200, 140, 3), 255, dtype=np.uint8)
    _, png = cv2.imencode(".png", page)
    monkeypatch.setenv("CODESCRIBE_TEXT_DETECTOR", "fake.onnx")
    monkeypatch.setattr(
        image_processing, "detect_rows",
        lambda image, _: (cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
                          [(12, 30, 105, 45), (25, 70, 120, 83)], 90),
    )
    crops, _, angle = image_processing.decode_and_segment(png.tobytes())
    assert angle == 90
    assert [crop.source_box for crop in crops] == [(95, 12, 15, 93), (57, 25, 13, 95)]


def test_wide_handwritten_margins_do_not_skip_nesting_levels():
    image = np.zeros((15, 60, 3), dtype=np.uint8)
    crops = [image_processing.LineCrop(image, x, 40 * i, 60, 15)
             for i, x in enumerate((12, 110, 112, 224, 109, 15))]
    assert image_processing.indent_levels(crops) == [0, 1, 1, 2, 1, 0]
