# VerticalStream AI — Stream Highlight & 9:16 Auto Clipper Studio

VerticalStream AI automatically converts any video stream (YouTube, Twitch, Kick, direct MP4) into viral, ready-to-upload 9:16 vertical clips for **TikTok, Instagram Reels, and YouTube Shorts**.

## ✨ Key Features
- **Stacked Streamer Facecam Split**: Automatically detects or locates the streamer's reaction camera in any corner and stacks it on top of the centered gameplay.
- **Instant Highlight Detection**: Replay Heatmaps and audio energy peak scanning pick high-excitement moments without scanning hours of video.
- **Multilingual Speech-to-Text Subtitles**: Real speech recognition supporting Hindi (Devanagari typography), English, Spanish, French, and German with animated karaoke bounce effects.
- **Color Grading LUTs**: Viral Vibrant (+28% pop), Cinematic Teal & Orange, Golden Warmth, and Original.
- **Anti-Copyright Evasion Shield**: Imperceptible audio pitch/speed shift ($1.006\times$) and subtle vignette to bypass automated ContentID fingerprinting.
- **Hardware Acceleration**: Built-in NVIDIA GPU NVENC acceleration with seamless multi-threaded CPU fallback.

## 🚀 Quick Start

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```
or double-click `run.bat` on Windows.

3. Open your browser:
```
http://127.0.0.1:8000
```
