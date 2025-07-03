import os
import re
import json
import uuid
import asyncio
import tempfile
import threading
import warnings

import requests
import numpy as np
import librosa
import soundfile as sf

import whisper
import edge_tts
import tiktoken

from flask import (
    Flask,
    request,
    send_from_directory,
    render_template_string,
)
from moviepy import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips,
    concatenate_audioclips,
)
from dotenv import load_dotenv

# ── Environment ───────────────────────────────────────
load_dotenv()
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU*")

API_KEY  = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")
MODEL    = "gpt-3.5-turbo"
MAX_TOKENS_PER_CALL = 3000

encoder = tiktoken.encoding_for_model(MODEL)

# ── Flask setup ───────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1_500 * 1024 * 1024   # 1.5 GB upload limit

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Utility helpers ───────────────────────────────────
def count_tokens(text: str) -> int:
    return len(encoder.encode(text))

def split_batches(texts, max_tokens=MAX_TOKENS_PER_CALL):
    batch, cnt = [], 0
    for t in texts:
        tcnt = count_tokens(t) + 10          # small margin
        if batch and cnt + tcnt > max_tokens:
            yield batch
            batch, cnt = [], 0
        batch.append(t)
        cnt += tcnt
    if batch:
        yield batch

def call_hinglish(sentences):
    """
    Send a batch to GPT and get Hinglish (Roman‑script Hindi/English mix) back.
    Returns a list equal in length to `sentences`.
    """
    prompt = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Convert the following sentences into *modern, conversational "
                    "Hinglish* (Roman script). Reply ONLY with a JSON array of "
                    "strings in the same order."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }
    r = requests.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers, timeout=90)
    txt = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\[[\s\S]*\]", txt)       # find JSON array
    return json.loads(m.group(0)) if m else sentences

def convert_hinglish(texts):
    out = []
    for batch in split_batches(texts):
        try:
            out.extend(call_hinglish(batch))
        except Exception:
            # fallback to original English if conversion fails
            out.extend(batch)
    return out

def detect_gender(wav_file, start, end):
    """
    Quick pitch‑based heuristic:
        >160 Hz → female, else male
    """
    y, sr = librosa.load(wav_file, sr=None, offset=start, duration=end - start)
    try:
        f0, *_ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        f0 = f0[~np.isnan(f0)]
        return "female" if f0.size and f0.mean() > 160 else "male"
    except Exception:
        return "male"

async def _edge_speak(text, voice, path):
    await edge_tts.Communicate(text=text, voice=voice).save(path)

def tts(text, gender, out_path):
    voice = "en-IN-PrabhatNeural" if gender == "male" else "en-IN-NeerjaNeural"
    asyncio.run(_edge_speak(text, voice, out_path))

# ── Core video processing ─────────────────────────────
def process_video(input_path, output_path):
    model = whisper.load_model("base")
    video = VideoFileClip(input_path)
    vid_len, fps = video.duration, video.fps

    # Extract audio to temp WAV
    tmp_wav = tempfile.mktemp(suffix=".wav")
    video.audio.write_audiofile(tmp_wav, logger=None)

    segments = model.transcribe(tmp_wav)["segments"]
    valid_segments = []
    for seg in segments:
        if seg["end"] > seg["start"]:
            valid_segments.append(
                {
                    "start": seg["start"],
                    "end": min(seg["end"], vid_len),
                    "text": seg["text"].strip(),
                    "gender": detect_gender(tmp_wav, seg["start"], seg["end"]),
                }
            )

    texts      = [v["text"] for v in valid_segments]
    hinglishes = convert_hinglish(texts)

    clips_v, clips_a = [], []
    for seg, hi in zip(valid_segments, hinglishes):
        st, ed         = seg["start"], seg["end"]
        dur_original   = ed - st

        # TTS
        mp3_tmp = tempfile.mktemp(suffix=".mp3")
        wav_tmp = tempfile.mktemp(suffix=".wav")
        tts(hi, seg["gender"], mp3_tmp)

        y, sr = librosa.load(mp3_tmp, sr=None)
        sf.write(wav_tmp, y, sr)
        dur_tts = len(y) / sr if sr else dur_original

        # If audio longer than original, slow video; else leave 1.0x
        speed_factor = dur_original / dur_tts if dur_tts else 1.0

        subvid = (
            video.subclipped(st, ed)
            .with_speed_scaled(factor=speed_factor)
        )
        clips_v.append(subvid)
        clips_a.append(AudioFileClip(wav_tmp))

    final_v = concatenate_videoclips(clips_v, method="chain")
    final_a = concatenate_audioclips(clips_a)
    final   = final_v.with_audio(final_a)
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        fps=fps,
        preset="ultrafast",
        threads=os.cpu_count(),
        logger=None,
    )

    # Cleanup
    video.close()
    final.close()
    os.remove(tmp_wav)

# ── Background thread wrapper ─────────────────────────
job_status = {}  # uid → "processing" | "done" | f"error::{msg}"

def background_worker(in_path, out_path, uid):
    try:
        process_video(in_path, out_path)
        job_status[uid] = "done"
    except Exception as e:
        job_status[uid] = f"error::{e}"

# ── Routes ────────────────────────────────────────────
HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>🎤 Hinglish Video Dubber</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body {background:#0f0f0f;color:#fff;font-family:sans-serif;text-align:center;padding:2em}
    h1 {margin-bottom:1em}
    form {display:inline-block;background:#1e1e1e;padding:2em;border-radius:12px}
    input,button {width:100%;max-width:380px;padding:12px;margin:8px 0;font-size:1em;border:none;border-radius:6px}
    input {background:#333;color:#ccc}
    button {background:#1db954;color:#fff;cursor:pointer}
    button:hover{background:#19a34d}
  </style>
</head>
<body>
  <h1>🎤 Hinglish Video Dubber</h1>
  <form method="POST" action="/upload" enctype="multipart/form-data">
    <input type="file" name="video" accept="video/*" required>
    <button type="submit">Upload & Convert</button>
  </form>
</body>
</html>
"""

PROCESSING_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>⏳ Converting…</title>
  <meta http-equiv="refresh" content="5">
  <style>
    body {background:#000;color:#ccc;font-family:sans-serif;text-align:center;padding-top:5em}
    .loader {font-size:1.5em;animation:blink 1s infinite}
    @keyframes blink {50% {opacity:.3}}
  </style>
</head>
<body>
  <h2 class="loader">⏳ Converting… Please wait</h2>
  <p>This page refreshes every 5 s.</p>
</body>
</html>
"""

DONE_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>✅ Done!</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body {{background:#111;color:#fff;font-family:sans-serif;text-align:center;padding:2em}}
    video {{max-width:100%;border-radius:10px;margin-top:1em}}
    a {{color:#1db954;font-weight:bold}}
  </style>
</head>
<body>
  <h2>✅ Dubbed Video Ready!</h2>
  <video src="{url}" controls></video><br>
  <a href="{url}" download>⬇ Download</a>
</body>
</html>
"""

@app.route("/")
def index():
    return HOME_HTML

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["video"]
    ext  = os.path.splitext(file.filename)[-1]
    uid  = uuid.uuid4().hex
    in_path  = os.path.join(UPLOAD_DIR,  uid + ext)
    out_path = os.path.join(OUTPUT_DIR, uid + ".mp4")
    file.save(in_path)

    job_status[uid] = "processing"
    threading.Thread(target=background_worker, args=(in_path, out_path, uid), daemon=True).start()

    return f'<meta http-equiv="refresh" content="0; url=/status/{uid}">'

@app.route("/status/<uid>")
def status(uid):
    status = job_status.get(uid, "not_found")
    if status == "done":
        url = f"/download/{uid}.mp4"
        return DONE_HTML.format(url=url)
    if status.startswith("error::"):
        msg = status.split("::", 1)[-1]
        return f"<h1>❌ Error</h1><pre>{msg}</pre>"
    # default: still processing
    return PROCESSING_HTML

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)

# ── Run ───────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7860)
