@echo off
cd /d "%~dp0"
echo Starting VerticalStream Clipper Web App...
python -m uvicorn server:app --host 127.0.0.1 --port 8000
pause
