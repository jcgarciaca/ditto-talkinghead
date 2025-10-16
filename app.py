import os
import uuid
import soundfile as sf
from io import BytesIO
from pathlib import Path
from typing import Optional
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException

from server_utils import run_inference

"""
Usage:
uvicorn app:app --host 0.0.0.0 --port 8893 --workers=1 --reload
"""

app = FastAPI(title='Avatar Generator')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return "Server running"

sessionId = str(uuid.uuid4())
conversation = []

audio_file_location = Path("/workdir/ditto-talkinghead/tmp/oviedo")

@app.post("/avatar-generator")
async def avatar(audio_file: UploadFile=File(...)):
    image_path = '/workdir/ditto-talkinghead/example/Oviedo-Dane.png'
    
    contents = await audio_file.read()

    sessionId = str(uuid.uuid4())
    output_path = f'/workdir/ditto-talkinghead/tmp/oviedo/{sessionId}.mp4'
    audio_path = audio_file_location / f'{sessionId}_{audio_file.filename}'
    audio_path.write_bytes(contents)


    success = await run_inference(audio_path, image_path, output_path)
    print('done!!')
    if success and os.path.exists(output_path):
        output_video = open(output_path, mode="rb")
        return StreamingResponse(output_video, media_type="video/mp4")
    raise HTTPException(status_code=415, detail="Unsupported file provided.")    
