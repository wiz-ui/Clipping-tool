import os
import sys
import re
import json
import uuid
import subprocess
import asyncio
import imageio_ffmpeg
import numpy as np
import cv2
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="VerticalStream AI - Pro Split Cam Studio v6")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
AudioSegment.converter = FFMPEG_EXE
AudioSegment.ffmpeg = FFMPEG_EXE

def detect_gpu_encoder():
    try:
        p = subprocess.run([FFMPEG_EXE, "-encoders"], capture_output=True, text=True, errors="replace")
        if "h264_nvenc" in p.stdout:
            test_cmd = [
                FFMPEG_EXE, "-y", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
                "-c:v", "h264_nvenc", "-f", "null", "-"
            ]
            test_res = subprocess.run(test_cmd, capture_output=True, text=True)
            if test_res.returncode == 0:
                return "h264_nvenc"
    except Exception as e:
        print("GPU detect exception:", e)
    return "libx264"

GPU_ENCODER = detect_gpu_encoder()
print(f"[*] Engine Initialized. Active Video Encoder: {GPU_ENCODER.upper()}")

JOBS = {}

class FrameProbeRequest(BaseModel):
    url: str

class ClipRequest(BaseModel):
    url: str
    target_clips: int = 4
    clip_duration: int = 30
    layout_mode: str = "split_cam"
    cam_x: float = 1.0 # percentage left (0 to 100)
    cam_y: float = 1.5 # percentage top (0 to 100)
    cam_w: float = 33.0 # percentage width (0 to 100)
    cam_h: float = 36.0 # percentage height (0 to 100)
    cam_split_ratio: int = 38 # 38% top for facecam, 62% bottom for gameplay
    language: str = "hi"
    color_grade: str = "vibrant"
    subtitles_style: str = "bold_yellow"
    anti_copyright: bool = True
    use_gpu: bool = True

def run_cmd(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    return p.returncode, p.stdout, p.stderr

def extract_direct_stream_urls(url: str):
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--ffmpeg-location", FFMPEG_EXE,
        "-g",
        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "--no-playlist",
        url
    ]
    code, stdout, stderr = run_cmd(cmd)
    if code != 0 or not stdout.strip():
        cmd = [sys.executable, "-m", "yt_dlp", "-g", "-f", "best", "--no-playlist", url]
        code, stdout, stderr = run_cmd(cmd)

    lines = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
    if len(lines) >= 2:
        return lines[0], lines[1]
    elif len(lines) == 1:
        return lines[0], lines[0]
    return None, None

def get_stream_metadata(url: str):
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-J",
        "--no-playlist",
        "--skip-download",
        url
    ]
    code, stdout, stderr = run_cmd(cmd)
    duration = 0
    title = "Stream"
    heatmap_timestamps = []
    if code == 0:
        try:
            info = json.loads(stdout)
            duration = info.get("duration", 0) or 0
            title = info.get("title", "Stream")
            heatmap = info.get("heatmap", [])
            if heatmap and isinstance(heatmap, list):
                sorted_heat = sorted(heatmap, key=lambda x: x.get("value", 0), reverse=True)
                for item in sorted_heat:
                    st = item.get("start_time", 0)
                    if st > 15:
                        heatmap_timestamps.append(float(st))
        except Exception as e:
            print("Metadata parse warning:", e)
            
    return duration, title, heatmap_timestamps

def auto_detect_cam_box_from_frame(frame_path: str):
    try:
        img = cv2.imread(frame_path)
        if img is None:
            return 1.0, 1.5, 33.0, 36.0
        h, w, _ = img.shape

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, (35, 70, 70), (85, 255, 255))
        mask_cyan = cv2.inRange(hsv, (80, 80, 80), (105, 255, 255))
        mask_magenta = cv2.inRange(hsv, (140, 80, 80), (175, 255, 255))
        combined_mask = mask_green | mask_cyan | mask_magenta

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 800:
                bx, by, bw, bh = cv2.boundingRect(c)
                if (bw / w) >= 0.15 and (bw / w) <= 0.45 and (bh / h) >= 0.18:
                    candidates.append((bx, by, bw, bh))

        if candidates:
            candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
            bx, by, bw, bh = candidates[0]
            return round((bx / w) * 100, 1), round((by / h) * 100, 1), round((bw / w) * 100, 1), round((bh / h) * 100, 1)

    except Exception as e:
        print("Auto detect warning:", e)

    return 1.0, 1.5, 33.0, 36.0

def fast_detect_highlights(a_input: str, total_duration: float, target_clips: int, clip_duration: int, heatmap_timestamps: list):
    selected_timestamps = []
    if heatmap_timestamps:
        for t in heatmap_timestamps:
            if not any(abs(t - st) < clip_duration * 0.85 for st in selected_timestamps):
                selected_timestamps.append(t)
            if len(selected_timestamps) >= target_clips:
                return sorted(selected_timestamps)

    max_t = total_duration if total_duration > 0 else 600
    start_offset = max(20.0, max_t * 0.05)
    end_offset = max(start_offset + clip_duration, max_t * 0.95)
    
    num_probes = min(12, max(target_clips * 2, 6))
    probe_points = np.linspace(start_offset, end_offset - clip_duration, num_probes)
    
    scored_points = []
    for pt in probe_points:
        quick_probe_cmd = [
            FFMPEG_EXE,
            "-ss", str(int(pt)),
            "-t", "4",
            "-i", a_input,
            "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
            "-f", "null", "-"
        ]
        _, _, probe_err = run_cmd(quick_probe_cmd)
        rms_vals = re.findall(r"lavfi\.astats\.Overall\.RMS_level=([-\.0-9]+)", probe_err)
        if rms_vals:
            numeric_rms = []
            for r in rms_vals:
                try:
                    val = float(r)
                    if val > -60 and not np.isinf(val) and not np.isnan(val):
                        numeric_rms.append(val)
                except:
                    pass
            avg_rms = np.mean(numeric_rms) if numeric_rms else -50
            scored_points.append((avg_rms, pt))
        else:
            scored_points.append((-45, pt))

    scored_points.sort(key=lambda x: x[0], reverse=True)
    for score, pt in scored_points:
        if not any(abs(pt - st) < clip_duration * 0.8 for st in selected_timestamps):
            selected_timestamps.append(int(pt))
        if len(selected_timestamps) >= target_clips:
            break

    if len(selected_timestamps) < target_clips:
        step = (max_t - 40) / (target_clips + 1)
        cur = 30.0
        while len(selected_timestamps) < target_clips and cur + clip_duration <= max_t:
            if not any(abs(cur - st) < clip_duration * 0.7 for st in selected_timestamps):
                selected_timestamps.append(int(cur))
            cur += step

    return sorted(selected_timestamps)

def transcribe_audio_segments(wav_path: str, lang_code: str):
    segments = []
    recognizer = sr.Recognizer()
    primary_lang = "hi-IN" if lang_code in ("hi", "hindi") else "en-US"
    fallback_lang = "en-IN" if lang_code in ("hi", "hindi") else "en-US"

    try:
        sound = AudioSegment.from_wav(wav_path)
        total_len_sec = sound.duration_seconds

        with sr.AudioFile(wav_path) as source:
            full_audio = recognizer.record(source)
        
        full_text = None
        try:
            full_text = recognizer.recognize_google(full_audio, language=primary_lang)
        except:
            try:
                full_text = recognizer.recognize_google(full_audio, language=fallback_lang)
            except:
                pass

        if full_text and full_text.strip():
            words = full_text.strip().split()
            chunk_size = 4
            num_chunks = max(1, (len(words) + chunk_size - 1) // chunk_size)
            chunk_duration = total_len_sec / num_chunks
            for i in range(num_chunks):
                start_t = i * chunk_duration
                end_t = min(total_len_sec, (i + 1) * chunk_duration)
                chunk_words = " ".join(words[i * chunk_size : (i + 1) * chunk_size])
                segments.append((start_t, end_t, chunk_words))
            return segments

        chunks = split_on_silence(sound, min_silence_len=500, silence_thresh=sound.dBFS - 14, keep_silence=200)
        current_time = 0.0
        for i, chunk in enumerate(chunks[:6]):
            chunk_dur = chunk.duration_seconds
            if chunk_dur < 1.0:
                current_time += chunk_dur
                continue
            temp_chunk_path = wav_path.replace(".wav", f"_chunk_{i}.wav")
            chunk.export(temp_chunk_path, format="wav")
            with sr.AudioFile(temp_chunk_path) as source:
                chunk_audio = recognizer.record(source)
            text = None
            try:
                text = recognizer.recognize_google(chunk_audio, language=primary_lang)
            except:
                try:
                    text = recognizer.recognize_google(chunk_audio, language=fallback_lang)
                except:
                    pass
            try:
                os.remove(temp_chunk_path)
            except:
                pass

            if text and text.strip():
                segments.append((current_time, current_time + chunk_dur, text.strip()))
            current_time += chunk_dur

    except Exception as e:
        print("Transcription warning:", e)

    return segments

def generate_pro_ass_subtitles(ass_path: str, duration: int, spoken_segments: list, style: str, clip_num: int, language: str):
    if style == "none":
        return None

    if style == "bold_yellow":
        hook_style = "Style: Hook,Nirmala UI,62,&H0000E5FF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,7,2,8,20,20,160,1"
        sub_style = "Style: Default,Nirmala UI,48,&H0000FFFF,&H000000FF,&H00050505,&HB0000000,-1,0,0,0,100,100,0,0,1,6,2,2,40,40,240,1"
    elif style == "neon_cyan":
        hook_style = "Style: Hook,Segoe UI,64,&H00FFFF00,&H00FF00FF,&H00111111,&H90000000,-1,0,0,0,100,100,0,0,1,7,3,8,20,20,160,1"
        sub_style = "Style: Default,Segoe UI,50,&H00FFFF00,&H00FF00FF,&H000A0A0A,&HB0000000,-1,0,0,0,100,100,0,0,1,6,3,2,40,40,240,1"
    else:
        hook_style = "Style: Hook,Nirmala UI,62,&H0000CCFF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,7,2,8,20,20,160,1"
        sub_style = "Style: Default,Nirmala UI,48,&H00FFFFFF,&H0000FFFF,&H00000000,&HB0000000,-1,0,0,0,100,100,0,0,1,6,2,2,40,40,240,1"

    header = f"""[Script Info]
Title: Pro Vertical Captions
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{hook_style}
{sub_style}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    if language in ("hi", "hindi"):
        viral_hooks = [
            "🔥 भाई ये क्या हो गया?!",
            "लास्ट तक देखो... 😱",
            "गजब का मोमेंट भाई 🎯",
            "पूरी चैट हिल गई भाई 💀",
            "पागल क्लच कर दिया भाई! ⚡",
            "ये कैसे बच गया भाई?! 🚀"
        ]
    else:
        viral_hooks = [
            "🔥 NO WAY THIS HAPPENED!",
            "WAIT FOR THE END... 😱",
            "UNREAL STREAM MOMENT 🎯",
            "THIS HAD THE CHAT CRYING 💀",
            "INSANE CLUTCH PLAY! ⚡",
            "HOW DID HE SURVIVE THIS?! 🚀"
        ]

    hook_text = viral_hooks[(clip_num - 1) % len(viral_hooks)]
    events = []
    
    events.append(f"Dialogue: 1,0:00:00.00,0:00:04.50,Hook,,0,0,0,,{{\\an8\\pos(540,180)\\t(0,250,\\fscx120\\fscy120)\\t(250,500,\\fscx100\\fscy100)}}{hook_text}")

    if spoken_segments and len(spoken_segments) > 0:
        for s, e, text in spoken_segments:
            s_str = f"0:{int(s//60):02d}:{s%60:05.2f}"
            e_str = f"0:{int(e//60):02d}:{e%60:05.2f}"
            events.append(f"Dialogue: 0,{s_str},{e_str},Default,,0,0,0,,{{\\an2\\pos(540,1650)\\t(0,120,\\fscx112\\fscy112)\\t(120,260,\\fscx100\\fscy100)}}{text.strip()}")
    else:
        if language in ("hi", "hindi"):
            fallbacks = [
                (3.5, 8.5, "स्क्रीन पे देखो भाई सब लोग 👀"),
                (8.5, 14.5, "पूरी चैट पागल हो गई भाई 💬"),
                (14.5, 20.0, "क्या क्लच निकाला है भाई! 🏆"),
                (20.0, float(duration), "डेली स्ट्रीम के लिए फॉलो करो 🔔")
            ]
        else:
            fallbacks = [
                (3.5, 8.5, "LOOK AT THE SCREEN RIGHT NOW 👀"),
                (8.5, 14.5, "EVERYONE IN CHAT WENT CRAZY 💬"),
                (14.5, 20.0, "HE ACTUALLY PULLED IT OFF! 🏆"),
                (20.0, float(duration), "FOLLOW FOR DAILY STREAM CLIPS 🔔")
            ]
        for s, e, text in fallbacks:
            if s < duration:
                end_val = min(float(duration), e)
                s_str = f"0:{int(s//60):02d}:{s%60:05.2f}"
                e_str = f"0:{int(end_val//60):02d}:{end_val%60:05.2f}"
                events.append(f"Dialogue: 0,{s_str},{e_str},Default,,0,0,0,,{{\\an2\\pos(540,1650)\\t(0,140,\\fscx110\\fscy110)\\t(140,280,\\fscx100\\fscy100)}}{text}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))
    
    return ass_path

def get_color_filter(style: str):
    if style == "vibrant":
        return "eq=saturation=1.28:contrast=1.12:brightness=0.01"
    elif style == "cinematic":
        return "eq=contrast=1.18:saturation=1.1,colorbalance=rs=.05:gs=-.02:bs=-.05:rm=-.02:gm=.02:bm=.08"
    elif style == "warm":
        return "colorbalance=rs=.12:gs=.04:bs=-.08,eq=saturation=1.15"
    return ""

@app.post("/api/probe-frame")
async def probe_stream_frame(req: FrameProbeRequest):
    try:
        v_stream_url, _ = extract_direct_stream_urls(req.url.strip())
        if not v_stream_url:
            raise HTTPException(status_code=400, detail="Could not resolve video stream.")
        
        probe_id = str(uuid.uuid4())[:8]
        probe_img_path = os.path.join(STATIC_DIR, f"probe_{probe_id}.jpg")
        
        grab_cmd = [
            FFMPEG_EXE, "-y",
            "-ss", "30",
            "-i", v_stream_url,
            "-vframes", "1",
            "-q:v", "2",
            probe_img_path
        ]
        run_cmd(grab_cmd)
        
        if not os.path.exists(probe_img_path):
            raise HTTPException(status_code=500, detail="Failed to grab stream snapshot.")

        bx, by, bw, bh = auto_detect_cam_box_from_frame(probe_img_path)
        
        return {
            "image_url": f"/static/probe_{probe_id}.jpg",
            "detected_cam": {
                "x": bx,
                "y": by,
                "w": bw,
                "h": bh
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_stream_task(job_id: str, req: ClipRequest):
    try:
        JOBS[job_id]["status"] = "resolving"
        JOBS[job_id]["message"] = "Resolving stream endpoints & instant heatmaps..."
        
        job_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        v_stream_url, a_stream_url = extract_direct_stream_urls(req.url)
        duration, title, heatmap = get_stream_metadata(req.url)
        
        use_direct_stream = (v_stream_url is not None)
        raw_video_path = None
        
        if not use_direct_stream:
            JOBS[job_id]["message"] = "Direct stream not available; fetching accelerated segments..."
            raw_video_path = os.path.join(job_dir, "raw_stream.mp4")
            dl_cmd = [
                sys.executable, "-m", "yt_dlp",
                "--ffmpeg-location", FFMPEG_EXE,
                "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
                "--merge-output-format", "mp4",
                "-o", raw_video_path,
                "--no-playlist",
                req.url
            ]
            run_cmd(dl_cmd)
            v_input = raw_video_path
            a_input = raw_video_path
        else:
            v_input = v_stream_url
            a_input = a_stream_url

        JOBS[job_id]["status"] = "analyzing"
        JOBS[job_id]["message"] = "Scanning highlight energy peaks..."

        selected_timestamps = fast_detect_highlights(a_input, duration, req.target_clips, req.clip_duration, heatmap)

        # Proportional crop for facecam box
        cam_crop_filter = f"crop=iw*{req.cam_w/100:.3f}:ih*{req.cam_h/100:.3f}:iw*{req.cam_x/100:.3f}:ih*{req.cam_y/100:.3f}"

        # Split heights
        top_h = int(1920 * (req.cam_split_ratio / 100.0))
        top_h = top_h - (top_h % 2)
        bot_h = 1920 - top_h

        active_encoder = "libx264"
        encoder_args = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "21"]
        if req.use_gpu and GPU_ENCODER == "h264_nvenc":
            active_encoder = "h264_nvenc"
            encoder_args = [
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-rc", "vbr",
                "-cq", "22",
                "-b:v", "5000k"
            ]

        JOBS[job_id]["status"] = "rendering"
        clips_results = []
        color_filter = get_color_filter(req.color_grade)

        for idx, start_sec in enumerate(selected_timestamps):
            clip_name = f"clip_{idx+1}_9x16.mp4"
            out_clip_path = os.path.join(job_dir, clip_name)
            wav_path = os.path.join(job_dir, f"audio_{idx+1}.wav")
            ass_path = os.path.join(job_dir, f"subs_{idx+1}.ass")

            JOBS[job_id]["message"] = f"Transcribing {req.language.upper()} speech for highlight {idx+1}/{len(selected_timestamps)}..."

            audio_extract_cmd = [
                FFMPEG_EXE, "-y",
                "-ss", str(start_sec),
                "-t", str(req.clip_duration),
                "-i", a_input,
                "-ac", "1",
                "-ar", "16000",
                wav_path
            ]
            run_cmd(audio_extract_cmd)

            spoken_segments = []
            if req.subtitles_style != "none" and os.path.exists(wav_path):
                spoken_segments = transcribe_audio_segments(wav_path, req.language)

            has_subtitles = False
            if req.subtitles_style != "none":
                generate_pro_ass_subtitles(ass_path, req.clip_duration, spoken_segments, req.subtitles_style, idx + 1, req.language)
                has_subtitles = os.path.exists(ass_path)

            JOBS[job_id]["message"] = f"Rendering Split Facecam layout (Top: {top_h}px, Bot: {bot_h}px) with {active_encoder.upper()} for clip {idx+1}..."

            # ROBUST FILTER GRAPH WITH CORRECT LABELS
            if req.layout_mode == "split_cam":
                # [0:v] -> split into [c1] and [g1]
                # [c1] -> crop facecam, scale to fill 1080x[top_h]
                # [g1] -> scale to fill 1080x[bot_h], crop to center
                # vstack -> [layout]
                filter_base = (
                    f"[0:v]split=2[c1][g1];"
                    f"[c1]{cam_crop_filter},scale=1080:{top_h}:force_original_aspect_ratio=increase,crop=1080:{top_h}[top_cam];"
                    f"[g1]scale=1080:{bot_h}:force_original_aspect_ratio=increase,crop=1080:{bot_h}[bot_game];"
                    f"[top_cam][bot_game]vstack=inputs=2[layout]"
                )
                last_node = "[layout]"
            elif req.layout_mode == "ambient_blur":
                filter_base = (
                    "[0:v]split=2[bg_in][fg_in];"
                    "[bg_in]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=22:6[bg];"
                    "[fg_in]scale=1080:-1[fg];"
                    "[bg][fg]overlay=(W-w)/2:(H-h)/2[layout]"
                )
                last_node = "[layout]"
            else:
                filter_base = "[0:v]crop=ih*9/16:ih:(iw-ow)/2:0,scale=1080:1920[layout]"
                last_node = "[layout]"

            filter_nodes = [filter_base]

            if color_filter:
                filter_nodes.append(f"{last_node}{color_filter}[colored]")
                last_node = "[colored]"

            if has_subtitles:
                escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
                filter_nodes.append(f"{last_node}ass='{escaped_ass}'[outv]")
            else:
                filter_nodes.append(f"{last_node}null[outv]")

            full_filter_complex = ";".join(filter_nodes)
            audio_filter_arg = "atempo=1.006" if req.anti_copyright else "anull"

            render_cmd = [
                FFMPEG_EXE, "-y",
                "-ss", str(start_sec),
                "-t", str(req.clip_duration),
                "-i", v_input,
                "-ss", str(start_sec),
                "-t", str(req.clip_duration),
                "-i", a_input,
                "-filter_complex", full_filter_complex,
                "-map", "[outv]",
                "-map", "1:a",
                "-af", audio_filter_arg,
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "48000"
            ]

            render_cmd.extend(encoder_args)
            render_cmd.extend([
                "-movflags", "+faststart",
                out_clip_path
            ])

            code, out, err = run_cmd(render_cmd)

            # Fallback to CPU if needed
            if (not os.path.exists(out_clip_path) or os.path.getsize(out_clip_path) < 1000) and active_encoder == "h264_nvenc":
                fallback_cmd = render_cmd[:]
                nv_i = fallback_cmd.index("h264_nvenc")
                fallback_cmd[nv_i - 1 : nv_i + 7] = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22"]
                run_cmd(fallback_cmd)

            if os.path.exists(out_clip_path) and os.path.getsize(out_clip_path) > 1000:
                clips_results.append({
                    "id": idx + 1,
                    "title": f"Clutch Highlight #{idx + 1} ({int(start_sec//60)}m {int(start_sec%60)}s)",
                    "start_time": f"{int(start_sec//60):02d}:{int(start_sec%60):02d}",
                    "duration": f"{req.clip_duration}s",
                    "filename": clip_name,
                    "url": f"/api/download/{job_id}/{clip_name}",
                    "size_mb": round(os.path.getsize(out_clip_path) / (1024 * 1024), 2),
                    "layout": f"Split Cam ({req.cam_split_ratio}% Cam / {100-req.cam_split_ratio}% Game)",
                    "language": req.language.upper(),
                    "encoder": active_encoder.upper()
                })

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["message"] = f"Finished {len(clips_results)} vertical clips ready for upload!"
        JOBS[job_id]["clips"] = clips_results

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)

@app.get("/api/system-info")
async def system_info():
    return {
        "gpu_available": GPU_ENCODER == "h264_nvenc",
        "gpu_model": "NVIDIA GeForce GTX 1650 (NVENC Accelerated)" if GPU_ENCODER == "h264_nvenc" else "None",
        "cpu_threads": os.cpu_count() or 4
    }

@app.post("/api/clip")
async def start_clipping(req: ClipRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "status": "queued",
        "message": "Initializing highlight peak scanner & streaming pipeline...",
        "clips": [],
        "error": None
    }
    background_tasks.add_task(process_stream_task, job_id, req)
    return {"job_id": job_id, "status": "queued"}

@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]

@app.get("/api/download/{job_id}/{filename}")
async def download_clip(job_id: str, filename: str):
    file_path = os.path.join(OUTPUT_DIR, job_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
