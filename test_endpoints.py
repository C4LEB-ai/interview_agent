import requests
import base64

# Start interview
response = requests.post("http://localhost:8000/api/interview/start", 
    json={
        "candidate_id": "test_123",
        "position": "software_engineer",
        "candidate_name": "John Doe"
    }
)
session_data = response.json()
session_id = session_data["session_id"]


# Read audio file as bytes
with open("audio_response/output4.wav", "rb") as f:
    audio_bytes = f.read()

# Submit audio response (dummy data)
audio_data = base64.b64encode(b"dummy_audio_data").decode()
response = requests.post("http://localhost:8000/api/interview/response",
    json={
        "session_id": session_id,
        "audio_base64": audio_data
    }
)
result = response.json()

# Get summary
response = requests.get(f"http://localhost:8000/api/interview/summary/{session_id}")
summary = response.json()

print(f"Interview completed with score: {summary['session_summary']['overall_score']}")