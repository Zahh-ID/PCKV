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
