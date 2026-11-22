import cv2
import numpy as np
from fastapi.testclient import TestClient
from app.main import app

client=TestClient(app)

def test_health():
    assert client.get('/api/health').status_code==200

def test_valid_code():
    data=client.post('/api/analyze',json={'code':'for i in range(5):\n    print(i)'})
    assert data.status_code==200
    assert data.json()['parse_valid'] is True

def test_syntax_error():
    data=client.post('/api/analyze',json={'code':'for i in range(5)\n    print(i)'})
    assert data.status_code==200
    assert data.json()['parse_valid'] is False

def test_empty_code():
    assert client.post('/api/analyze',json={'code':' '}).json()['parse_valid'] is False

def test_reject_non_image():
    assert client.post('/api/process-image',files={'image':('x.txt',b'x','text/plain')}).status_code==415

def test_reject_bad_image():
    assert client.post('/api/process-image',files={'image':('x.png',b'x','image/png')}).status_code==422


def test_rotated_upload_recovers_lines_and_indentation(monkeypatch):
    monkeypatch.setenv('CODESCRIBE_MOCK_OCR', '1')
    page = np.full((270, 600, 3), 255, dtype=np.uint8)
    for x, y, code in [(35, 55, 'value = 12'), (37, 130, 'if value > 2:'), (79, 205, 'print(value)')]:
        cv2.putText(page, code, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
    matrix = cv2.getRotationMatrix2D((300, 135), 8, 1)
    rotated = cv2.warpAffine(page, matrix, (600, 270), borderValue=(255, 255, 255))
    _, png = cv2.imencode('.png', rotated)

    response = client.post('/api/process-image', files={'image': ('page.png', png.tobytes(), 'image/png')})

    assert response.status_code == 200
    result = response.json()
    assert abs(result['deskew_degrees'] + 8) <= 1
    assert [line['indent_level'] for line in result['lines']] == [0, 0, 1]
    assert result['code'].splitlines()[-1].startswith('    # DEMO OCR')
    for line in result['lines']:
        assert 0 <= line['x'] < line['x'] + line['width'] <= 600
        assert 0 <= line['y'] < line['y'] + line['height'] <= 270
