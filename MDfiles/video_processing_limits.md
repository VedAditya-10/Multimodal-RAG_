# Video Processing - Limits & Optimization

## Processing Caps

| Parameter | Limit | Reason |
|-----------|-------|--------|
| **Max File Size** | 100 MB | Protect upload + FFmpeg pipelines |
| **Frame Rate** | 0.5 fps (1 frame per 2 sec) | Avoid frame explosion |
| **Max Clip Length** | 10 minutes | Memory + processing time |
| **Max Keyframes** | 30 per video | LanceDB storage |
| **Max Resolution** | 720p | CLIP doesn't need 4K |

---

## Implementation

```python
# config.py
VIDEO_CONFIG = {
    "frame_rate": 0.5,           # Frames per second (1 every 2 seconds)
    "max_duration_seconds": 600,  # 10 minutes max
    "max_keyframes": 30,          # Cap total frames
    "resize_width": 1280,         # Resize to 720p max
    "resize_height": 720,
}
```

```python
# video.py
import cv2
from moviepy.editor import VideoFileClip
from config import VIDEO_CONFIG

def process_video(path: str) -> list[dict]:
    chunks = []
    
    # Load video
    clip = VideoFileClip(path)
    duration = min(clip.duration, VIDEO_CONFIG["max_duration_seconds"])
    
    # Calculate frame interval
    interval = 1 / VIDEO_CONFIG["frame_rate"]  # 2 seconds
    
    # Extract frames at interval (capped)
    frame_times = []
    t = 0
    while t < duration and len(frame_times) < VIDEO_CONFIG["max_keyframes"]:
        frame_times.append(t)
        t += interval
    
    # Process each frame
    for t in frame_times:
        frame = clip.get_frame(t)
        
        # Resize if needed
        h, w = frame.shape[:2]
        if w > VIDEO_CONFIG["resize_width"]:
            scale = VIDEO_CONFIG["resize_width"] / w
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Save frame, get CLIP embedding, OCR, etc.
        chunk = process_frame(frame, t, path)
        chunks.append(chunk)
    
    # Extract and transcribe audio (segments only, not word-level)
    audio_chunks = transcribe_audio_segments(clip.audio, duration)
    chunks.extend(audio_chunks)
    
    clip.close()
    return chunks

def transcribe_audio_segments(audio, duration: float) -> list[dict]:
    """Whisper transcription with segment-level timestamps."""
    import whisper
    
    # Extract audio to temp file
    temp_path = "/tmp/audio.wav"
    audio.write_audiofile(temp_path, fps=16000, verbose=False, logger=None)
    
    # Transcribe
    model = whisper.load_model("base")  # Use "base" for speed
    result = model.transcribe(temp_path, verbose=False)
    
    chunks = []
    for segment in result["segments"]:
        chunks.append({
            "text_content": segment["text"],
            "timestamp_start": segment["start"],
            "timestamp_end": segment["end"],
            "modality": "audio_transcript",
            "asr_confidence": 1.0 - segment.get("no_speech_prob", 0),
        })
    
    return chunks
```

---

## YouTube Downloads (yt-dlp)

```python
def download_youtube(url: str, output_dir: str) -> str:
    """Download YouTube video with limits."""
    import yt_dlp
    
    ydl_opts = {
        'format': 'best[height<=720]',  # Max 720p
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
        'max_duration': VIDEO_CONFIG["max_duration_seconds"],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return f"{output_dir}/{info['id']}.{info['ext']}"
```

---

## Warning for Long Videos

```python
def check_video_length(path: str) -> dict:
    """Warn user if video exceeds limits."""
    clip = VideoFileClip(path)
    duration = clip.duration
    clip.close()
    
    if duration > VIDEO_CONFIG["max_duration_seconds"]:
        return {
            "warning": True,
            "message": f"Video is {duration/60:.1f} min. Only first {VIDEO_CONFIG['max_duration_seconds']/60:.0f} min will be processed.",
            "will_process": VIDEO_CONFIG["max_duration_seconds"],
        }
    return {"warning": False}
```

---

## Summary

| What | How |
|------|-----|
| Extract 1 frame every 2 seconds | `fps=0.5` |
| Cap at 30 keyframes max | `max_keyframes=30` |
| Max 10 min video | `max_duration=600` |
| Resize to 720p | `resize_width=1280` |
| Use Whisper "base" model | Faster, good enough |
| Segment-level timestamps | Not word-level |
