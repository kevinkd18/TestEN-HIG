import os
import re
import json
import uuid
import asyncio
import tempfile
import warnings

import requests
import numpy as np
import librosa
import soundfile as sf

import whisper
import edge_tts
import tiktoken

from flask import Flask, request, send_from_directory, render_template_string
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips
from dotenv import load_dotenv

# ── ENV ─────────────────────────────────────────────
load_dotenv()
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU*")

API_KEY  = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")
MODEL    = "gpt-3.5-turbo"
MAX_TOKENS_PER_CALL = 3000

# ── FLASK ────────────────────────────────────────────
app = Flask(__name__)
# allow up to 1.5GB uploads
app.config["MAX_CONTENT_LENGTH"] = 1_500 * 1024 * 1024  

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD MODELS ONCE ─────────────────────────────────
# Whisper tiny uses <1GB RAM
whisper_model = whisper.load_model("tiny")
encoder = tiktoken.encoding_for_model(MODEL)

# ── HELPERS ──────────────────────────────────────────
def count_tokens(text: str) -> int:
    return len(encoder.encode(text))

def split_batches(texts, max_tokens=MAX_TOKENS_PER_CALL):
    batch, cnt = [], 0
    for t in texts:
        tcnt = count_tokens(t) + 10
        if batch and cnt + tcnt > max_tokens:
            yield batch
            batch, cnt = [], 0
        batch.append(t)
        cnt += tcnt
    if batch:
        yield batch

def call_hinglish(sentences):
    prompt = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    headers = {"Content-Type":"application/json","Authorization":f"Bearer {API_KEY}"}
    payload = {
        "model": MODEL,
        "messages":[
            {"role":"system","content":(
                "Convert the following sentences into modern, conversational Hinglish (Roman script). "
                "Reply ONLY with a JSON array of strings in the same order."
            )},
            {"role":"user","content":prompt}
        ],
        "temperature":0.7
    }
    r = requests.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers, timeout=90)
    txt = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\[[\s\S]*\]", txt)
    return json.loads(m.group(0)) if m else sentences

def convert_hinglish(texts):
    out = []
    for batch in split_batches(texts):
        try:
            out.extend(call_hinglish(batch))
        except:
            out.extend(batch)
    return out

def detect_gender(wav_file, start, end):
    y, sr = librosa.load(wav_file, sr=None, offset=start, duration=end-start)
    try:
        f0, *_ = librosa.pyin(y,
                              fmin=librosa.note_to_hz("C2"),
                              fmax=librosa.note_to_hz("C7"))
        f0 = f0[~np.isnan(f0)]
        return "female" if f0.size and f0.mean()>160 else "male"
    except:
        return "male"

async def _edge_speak(text, voice, path):
    await edge_tts.Communicate(text=text, voice=voice).save(path)

def tts(text, gender, out_path):
    voice = "en-IN-PrabhatNeural" if gender=="male" else "en-IN-NeerjaNeural"
    asyncio.run(_edge_speak(text, voice, out_path))

# ── PROCESS VIDEO ────────────────────────────────────
def process_video(in_path, out_path):
    video = VideoFileClip(in_path)
    duration, fps = video.duration, video.fps

    # extract audio
    tmp_wav = tempfile.mktemp(suffix=".wav")
    video.audio.write_audiofile(tmp_wav, logger=None)
    segments = whisper_model.transcribe(tmp_wav)["segments"]

    valid = []
    for seg in segments:
        if seg["end"] > seg["start"]:
            valid.append({
                "start":seg["start"],
                "end":min(seg["end"], duration),
                "text":seg["text"].strip(),
                "gender":detect_gender(tmp_wav, seg["start"], seg["end"])
            })

    texts = [v["text"] for v in valid]
    hinglishes = convert_hinglish(texts)

    clips_v, clips_a = [], []
    for vseg, hl in zip(valid, hinglishes):
        st, ed = vseg["start"], vseg["end"]
        orig_dur = ed - st

        # TTS -> WAV
        mp3f = tempfile.mktemp(suffix=".mp3")
        wavf = tempfile.mktemp(suffix=".wav")
        tts(hl, vseg["gender"], mp3f)
        y, sr = librosa.load(mp3f, sr=None)
        sf.write(wavf, y, sr)
        tts_dur = len(y)/sr if sr else orig_dur

        factor = orig_dur/tts_dur if tts_dur else 1.0
        sub = video.subclipped(st, ed).with_speed_scaled(factor=factor)
        clips_v.append(sub)
        clips_a.append(AudioFileClip(wavf))

    final_v = concatenate_videoclips(clips_v, method="chain")
    final_a = concatenate_audioclips(clips_a)
    final = final_v.with_audio(final_a)
    final.write_videofile(out_path,
                          codec="libx264",
                          audio_codec="aac",
                          audio_bitrate="192k",
                          fps=fps,
                          preset="ultrafast",
                          threads=1,
                          logger=None)

    video.close()
    final.close()
    os.remove(tmp_wav)

# ── ROUTES ────────────────────────────────────────────
HOME = """
<!DOCTYPE html><html><head>
  <title>🎤 Hinglish Dubber</title>
  <style>
    body{background:#111;color:#eee;font-family:sans-serif;text-align:center;padding:2em}
    form{display:inline-block;padding:2em;background:#222;border-radius:8px}
    input,button{width:100%;margin:8px 0;padding:12px;border:none;border-radius:6px;font-size:1em}
    input{background:#333;color:#ccc}button{background:#1db954;color:#fff;cursor:pointer}
  </style>
</head><body>
  <h1>🎤 Hinglish Video Dubber</h1>
  <form method="POST" action="/upload" enctype="multipart/form-data">
    <input type="file" name="video" accept="video/*" required>
    <button type="submit">Convert Now</button>
  </form>
</body></html>
"""

RESULT = """
<!DOCTYPE html><html><head>
  <title>✅ Done</title>
  <style>
    body{background:#111;color:#eee;font-family:sans-serif;text-align:center;padding:2em}
    video{max-width:100%;border-radius:8px;margin:1em 0}
    a{display:inline-block;margin-top:1em;color:#1db954;text-decoration:none;font-weight:bold}
  </style>
</head><body>
  <h2>✅ Conversion Complete!</h2>
  <video src="{url}" controls></video><br>
  <a href="{url}" download>⬇ Download</a>
  <p><a href="/">🔄 Convert another</a></p>
</body></html>
"""

@app.route("/")
def index():
    return HOME

@app.route("/upload", methods=["POST"])
def upload():
    vid = request.files["video"]
    ext = os.path.splitext(vid.filename)[1]
    uid = uuid.uuid4().hex
    in_path = os.path.join(UPLOAD_DIR, uid+ext)
    out_path = os.path.join(OUTPUT_DIR, uid+".mp4")
    vid.save(in_path)

    try:
        process_video(in_path, out_path)
    except Exception as e:
        return f"<h1>Error:</h1><pre>{e}</pre>", 500
    finally:
        # clean up input to save disk
        os.remove(in_path)

    return RESULT.format(url=f"/download/{uid}.mp4")

@app.route("/download/<fname>")
def download(fname):
    return send_from_directory(OUTPUT_DIR, fname)

if __name__ == "__main__":
    # single-threaded to minimize memory
    app.run(host="0.0.0.0", port=7860, debug=False)
