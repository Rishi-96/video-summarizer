# src/frontend.py


import os
import tempfile
import shutil
import warnings
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import numpy as np

# Optional ML libs
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

try:
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, pipeline as hf_pipeline
    MBART_AVAILABLE = True
except Exception:
    MBART_AVAILABLE = False

try:
    from transformers import pipeline as hf_pipeline_basic
    HF_SUMMARY_AVAILABLE = True
except Exception:
    HF_SUMMARY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    SENTS_AVAILABLE = True
except Exception:
    SENTS_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
    MOVIEPY_AVAILABLE = True
except Exception:
    MOVIEPY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

try:
    from PIL import Image
except Exception:
    Image = None

# BLIP optional
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except Exception:
    BLIP_AVAILABLE = False

# langdetect optional
try:
    import langdetect
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False

# Gemini (google) — used only for chat
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# -------------------------
# UI: Page + Dark theme
# -------------------------
st.set_page_config(layout="wide", page_title="🎬 Video Summarizer (30%)")

st.markdown(
    """
    <style>
      .stApp { background-color: #0f1115; color: #e6eef8; }
      .stSidebar { background-color: #0f1115; color: #e6eef8; }
      .css-1d391kg { background-color: #0f1115; }
      .stButton>button { background-color:#1f6feb;color: white; }
      .stTextInput>div>div>input {background-color:#111317;color:#e6eef8;}
      .stFileUploader { background-color:#111317; color:#e6eef8; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 AI Video Summarizer ")
st.caption("Uploads -> one-time processing -> summarized video + long text summary. ")

# -------------------------
# Sidebar settings
# -------------------------
st.sidebar.markdown("## Controls")
WHISPER_CHOICES = ["tiny", "base", "small", "medium", "large"]
whisper_choice = st.sidebar.selectbox("Whisper model (smaller=faster)", WHISPER_CHOICES, index=2)
frame_sample_count = st.sidebar.slider("Frame sampling count (for context/captions)", 3, 24, 8)
add_subtitles = st.sidebar.checkbox("Overlay subtitles on summarized video", value=True)
summary_target_lines = st.sidebar.slider("Target summary lines (approx)", 8, 25, 12)
target_ratio = st.sidebar.slider("Summarized video length ratio (of original)", 10, 50, 30) / 100.0  # e.g., 0.3
st.sidebar.markdown("---")
st.sidebar.caption("Gemini chat (right panel) will use cached context only; it won't rerun summarization.")

# -------------------------
# Gemini setup (chat-only)
# -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
GEMINI_MODEL = None
if GENAI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Try to pick a modern gemini model (best-effort)
        try:
            GEMINI_MODEL = genai.GenerativeModel("models/gemini-2.5-pro")
        except Exception:
            # fallback: choose first gemini in list
            try:
                models = [m.name for m in genai.list_models()]
                pick = None
                for m in models:
                    if "gemini-2.5" in m or "gemini-pro" in m or "gemini-flash" in m:
                        pick = m
                        break
                if pick:
                    GEMINI_MODEL = genai.GenerativeModel(pick)
            except Exception:
                GEMINI_MODEL = None
    except Exception:
        GEMINI_MODEL = None

# -------------------------
# Cached loaders & models
# -------------------------
@st.cache_resource
def load_whisper_model(name: str):
    if not WHISPER_AVAILABLE:
        return None
    try:
        return whisper.load_model(name)
    except Exception:
        return None

@st.cache_resource
def load_sentence_model(name: str = "all-MiniLM-L6-v2"):
    if not SENTS_AVAILABLE:
        return None
    try:
        return SentenceTransformer(name)
    except Exception:
        return None

@st.cache_resource
def load_blip_models():
    if not BLIP_AVAILABLE:
        return (None, None)
    try:
        proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        mdl = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return proc, mdl
    except Exception:
        return (None, None)

@st.cache_resource
def load_basic_summarizer():
    try:
        return hf_pipeline_basic("summarization", model="facebook/bart-large-cnn")
    except Exception:
        return None

# instantiate
WHISPER_MODEL = load_whisper_model(whisper_choice) if WHISPER_AVAILABLE else None
SENT_MODEL = load_sentence_model() if SENTS_AVAILABLE else None
BLIP_PROC, BLIP_MODEL = load_blip_models()
BASIC_SUMMARIZER = load_basic_summarizer()

MBART_MODEL = None
MBART_TOKENIZER = None
if MBART_AVAILABLE:
    try:
        MBART_MODEL = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        MBART_TOKENIZER = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    except Exception:
        MBART_MODEL = None
        MBART_TOKENIZER = None

# -------------------------
# Helper functions (same logic + small hardening)
# -------------------------
def transcribe_with_whisper(audio_path: str):
    if not WHISPER_AVAILABLE or WHISPER_MODEL is None:
        return "", None, None
    try:
        r = WHISPER_MODEL.transcribe(audio_path, language=None, task="transcribe")
        return r.get("text", ""), r.get("segments", []), r.get("language", None)
    except Exception as e:
        st.warning(f"Whisper error: {e}")
        return "", None, None

def sample_frames_uniform(video_path: str, max_frames: int = 12):
    imgs, times = [], []
    if not CV2_AVAILABLE or Image is None:
        return imgs, times
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return imgs, times
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if total <= 0:
            cap.release()
            return imgs, times
        step = max(1, total // max_frames)
        idx = grabbed = 0
        while grabbed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imgs.append(Image.fromarray(rgb))
                times.append(idx / fps)
                grabbed += 1
            idx += 1
        cap.release()
    except Exception as e:
        st.warning(f"Frame sampling failed: {e}")
    return imgs, times

def caption_frame_with_blip(pil_image):
    if not BLIP_AVAILABLE or BLIP_PROC is None or BLIP_MODEL is None:
        return ""
    try:
        inputs = BLIP_PROC(pil_image, return_tensors="pt")
        out = BLIP_MODEL.generate(**inputs, max_new_tokens=40)
        return BLIP_PROC.decode(out[0], skip_special_tokens=True)
    except Exception:
        return ""

def build_segment_texts(whisper_segments, frame_captions, frame_times):
    segments = []
    if whisper_segments:
        for seg in whisper_segments:
            s = {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
            attached = []
            for cap_txt, ft in zip(frame_captions, frame_times):
                if ft >= seg["start"] - 0.5 and ft <= seg["end"] + 0.5:
                    if cap_txt:
                        attached.append(cap_txt)
            if attached:
                s["text"] = s["text"] + " " + " ".join(attached)
            segments.append(s)
    else:
        for i, ft in enumerate(frame_times):
            start = max(0.0, ft - 1.0)
            end = ft + 1.0
            segments.append({"start": start, "end": end, "text": frame_captions[i] if i < len(frame_captions) else ""})
    return segments

def embed_texts(texts: List[str]):
    if not SENT_MODEL:
        return np.array([[len(t)] for t in texts], dtype=float)
    return SENT_MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def rank_segments_by_query(segments: List[Dict[str,Any]], query: str, top_k: int = 3):
    if not segments or not query:
        return segments[:top_k]
    texts = [s.get("text","") for s in segments]
    emb_texts = embed_texts(texts)
    query_emb = embed_texts([query])[0:1]
    if SENTS_AVAILABLE:
        sims = st_util.cos_sim(query_emb, emb_texts)[0].cpu().numpy()
    else:
        def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))
        sims = np.array([cos(query_emb[0], e) for e in emb_texts])
    idx_sorted = np.argsort(-sims)
    return [segments[i] for i in idx_sorted[:min(top_k, len(segments))]]

def stitch_segments_to_video(input_video: str, selected_segments: List[Dict[str,Any]], output_path: str, add_subtitles: bool = True):
    if not MOVIEPY_AVAILABLE:
        return None
    try:
        clip = VideoFileClip(input_video)
    except Exception as e:
        st.warning(f"Cannot open video for stitching: {e}")
        return None
    subclips = []
    for seg in selected_segments:
        s = max(0, seg["start"] - 0.3)
        e = min(seg["end"] + 0.3, clip.duration)
        try:
            sub = clip.subclip(s, e)
        except Exception:
            continue
        if add_subtitles and seg.get("text"):
            subtitle_text = seg["text"].strip()
            if len(subtitle_text) > 200:
                subtitle_text = subtitle_text[:200].rsplit(" ",1)[0] + "..."
            try:
                txtclip = TextClip(subtitle_text, fontsize=22, color='white', bg_color='black', size=(clip.w, None))
                txtclip = txtclip.set_position(("center", "bottom")).set_start(0).set_duration(sub.duration)
                sub = CompositeVideoClip([sub, txtclip])
            except Exception:
                pass
        subclips.append(sub)
    if not subclips:
        return None
    final = concatenate_videoclips(subclips, method="compose")
    try:
        final.write_videofile(output_path, codec="libx264", audio_codec="aac", threads=0, preset="medium", verbose=False, logger=None)
    except Exception:
        try:
            final.write_videofile(output_path, codec="libx264", audio_codec="aac")
        except Exception as e:
            st.warning(f"Final video write failed: {e}")
            return None
    return output_path

def summarize_text_multilingual(text: str, detected_lang: str = None, max_len: int = 500, min_len: int = 150):
    if not text or not text.strip():
        return ""
    if MBART_MODEL and MBART_TOKENIZER:
        try:
            code = detected_lang if detected_lang else (langdetect.detect(text) if LANGDETECT_AVAILABLE else None)
            mapping = {"en": "en_XX", "hi": "hi_IN", "mr": "mr_IN", "es": "es_XX", "fr": "fr_XX", "de": "de_DE", "zh": "zh_CN", "ru": "ru_RU"}
            tgt = mapping.get(code, "en_XX")
            MBART_TOKENIZER.src_lang = tgt
            inputs = MBART_TOKENIZER(text, return_tensors="pt", truncation=True, max_length=1024)
            ids = MBART_MODEL.generate(inputs.input_ids, max_length=max_len, min_length=min_len, num_beams=4)
            return MBART_TOKENIZER.batch_decode(ids, skip_special_tokens=True)[0]
        except Exception:
            pass
    if BASIC_SUMMARIZER:
        try:
            out = BASIC_SUMMARIZER(text, max_length=max_len, min_length=min_len)
            return out[0].get("summary_text", "")
        except Exception:
            pass
    return text.strip()[:max_len*2] + ("..." if len(text) > max_len*2 else "")

# -------------------------
# App state init: sessions & chat
# -------------------------
if "video_sessions" not in st.session_state:
    st.session_state["video_sessions"] = []  # list of dicts {id, name, tmp_dir, video_path, transcript, summary, segments, summarized_video_path}
if "current_session" not in st.session_state:
    st.session_state["current_session"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# -------------------------
# Left column: uploader + processing controls
# -------------------------
col1, col2, col3 = st.columns([2, 3, 1])

with col1:
    st.header("Upload & Process")
    uploaded = st.file_uploader("Upload an .mp4 video", type=["mp4", "mkv", "mov"])
    if uploaded:
        # create a new session and process if not already done for this file (unique by name+size)
        tmp_dir = Path(tempfile.mkdtemp())
        video_path = tmp_dir / uploaded.name
        with open(video_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved to {video_path}")

        # check if we already processed same file (name + size)
        match = None
        for s in st.session_state["video_sessions"]:
            if s.get("orig_name") == uploaded.name and s.get("orig_size") == uploaded.size:
                match = s
                break

        if match:
            st.info("This video was already processed — switching to existing session.")
            st.session_state["current_session"] = match["id"]
        else:
            # process pipeline and store session (single-run)
            sess_id = f"session_{len(st.session_state['video_sessions'])+1}"
            session_obj = {"id": sess_id, "orig_name": uploaded.name, "orig_size": uploaded.size, "tmp_dir": str(tmp_dir), "video_path": str(video_path)}
            st.session_state["current_session"] = sess_id
            st.session_state["video_sessions"].append(session_obj)

            # process: extract audio if possible
            audio_tmp = tmp_dir / "extracted_audio.wav"
            audio_ok = False
            duration = None
            if MOVIEPY_AVAILABLE:
                try:
                    clip = VideoFileClip(str(video_path))
                    duration = clip.duration
                    if clip.audio:
                        clip.audio.write_audiofile(str(audio_tmp), verbose=False, logger=None)
                        audio_ok = True
                    else:
                        audio_ok = False
                except Exception as e:
                    st.warning(f"MoviePy audio extract error: {e}")
                    audio_ok = False
            else:
                st.warning("MoviePy not available — audio extraction skipped.")

            # load (or reload) chosen whisper model
            if WHISPER_AVAILABLE:
                WHISPER_MODEL = load_whisper_model(whisper_choice)
            else:
                WHISPER_MODEL = None

            # transcribe
            transcript_text, whisper_segments, detected_lang = "", None, None
            if audio_ok and WHISPER_AVAILABLE and WHISPER_MODEL:
                with st.spinner("Transcribing (Whisper)..."):
                    transcript_text, whisper_segments, detected_lang = transcribe_with_whisper(str(audio_tmp))
            else:
                st.info("Skipping transcription (audio or Whisper not available).")

            # sample frames + captions
            frames, frame_times = sample_frames_uniform(str(video_path), max_frames=frame_sample_count)
            frame_captions = []
            if frames and BLIP_AVAILABLE:
                with st.spinner("Generating frame captions (BLIP)..."):
                    for f in frames:
                        try:
                            frame_captions.append(caption_frame_with_blip(f))
                        except Exception:
                            frame_captions.append("")
            else:
                frame_captions = [""] * len(frames)

            # build segments
            combined_segments = build_segment_texts(whisper_segments, frame_captions, frame_times)
            if not combined_segments and frames:
                combined_segments = [{"start": max(0.0, ft - 1.0), "end": ft + 1.0, "text": frame_captions[i] if i < len(frame_captions) else ""} for i, ft in enumerate(frame_times)]

            # select segments to reach ~target_ratio*duration
            selected = []
            if combined_segments:
                total_dur = duration if duration else sum(max(0, s["end"] - s["start"]) for s in combined_segments)
                target_seconds = target_ratio * total_dur
                # No query by default — pick by text-length relevance greedy
                segs_sorted = sorted(combined_segments, key=lambda s: len(s.get("text","")), reverse=True)
                curr = 0.0
                for seg in segs_sorted:
                    seg_len = max(0.1, seg["end"] - seg["start"])
                    if curr > 0 and (curr + seg_len) > target_seconds:
                        break
                    selected.append(seg)
                    curr += seg_len
            # stitch summarized video
            summarized_path = None
            if MOVIEPY_AVAILABLE and selected:
                summarized_path = str(tmp_dir / "summarized.mp4")
                with st.spinner("Creating summarized video..."):
                    outp = stitch_segments_to_video(str(video_path), selected, summarized_path, add_subtitles=add_subtitles)
                    if outp:
                        summarized_path = outp
                    else:
                        summarized_path = None

            # text summary (aim longer)
            combined_text = ""
            if transcript_text:
                combined_text += transcript_text + "\n\n"
            combined_text += "\n".join([s.get("text","") for s in selected])
            summary_text = summarize_text_multilingual(combined_text, detected_lang=detected_lang, max_len=500, min_len=150)

            # store results in session object
            session_obj.update({
                "duration": duration,
                "transcript": transcript_text,
                "segments": combined_segments,
                "selected": selected,
                "summary": summary_text,
                "summarized_video": summarized_path
            })
            # replace session in list
            for i, s in enumerate(st.session_state["video_sessions"]):
                if s["id"] == sess_id:
                    st.session_state["video_sessions"][i] = session_obj
                    break

# -------------------------
# Middle column: show selected session results and downloads
# -------------------------
with col2:
    st.header("Results")
    sessions = st.session_state["video_sessions"]
    if sessions:
        # session chooser
        sess_map = {s["id"]: s for s in sessions}
        cur_id = st.session_state.get("current_session") or sessions[-1]["id"]
        chosen = st.selectbox("Choose processed session", options=[s["id"] for s in sessions], index=[s["id"] for s in sessions].index(cur_id))
        st.session_state["current_session"] = chosen
        sess = sess_map[chosen]

        st.subheader(sess.get("orig_name", "Processed Video"))
        if sess.get("duration"):
            st.caption(f"Original duration: {sess['duration']:.1f}s — target summarized ≈ {target_ratio * sess['duration']:.1f}s")
        if sess.get("summarized_video"):
            st.video(sess["summarized_video"])
            st.download_button("Download summarized video", data=open(sess["summarized_video"], "rb"), file_name=f"summarized_{sess['orig_name']}")
        if sess.get("summary"):
            st.subheader("Text summary (detailed)")
            st.markdown(sess["summary"])
            st.download_button("Download summary (text)", data=sess["summary"], file_name=f"summary_{sess['orig_name']}.txt")
        else:
            st.info("No summary available for this session.")
    else:
        st.info("No processed sessions yet. Upload a video to begin.")

# -------------------------
# Right column: Chat (Gemini) — isolated
# -------------------------
with col3:
    st.header("Chat")
    st.caption("Ask questions about the selected processed video. Chat uses cached transcript/summary; it won't re-run processing.")
    if "current_session" in st.session_state and st.session_state["current_session"] is not None:
        ctx = next((s for s in st.session_state["video_sessions"] if s["id"] == st.session_state["current_session"]), None)
    else:
        ctx = None

    # show chat history UI
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for m in st.session_state["chat_history"]:
        role = m.get("role", "user")
        txt = m.get("text", "")
        if role == "user":
            st.chat_message("user").write(txt)
        else:
            st.chat_message("assistant").write(txt)

    user_q = st.chat_input("Ask anything about the video (use natural language)...")

    if user_q:
        st.session_state["chat_history"].append({"role": "user", "text": user_q})
        st.chat_message("user").write(user_q)

        # build concise context from current session
        if ctx:
            context_parts = []
            if ctx.get("transcript"):
                context_parts.append("Transcript snippet:\n" + ctx["transcript"][:3000])
            if ctx.get("summary"):
                context_parts.append("Summary:\n" + ctx["summary"][:2000])
            if ctx.get("selected"):
                preview = "\n".join([f"[{s['start']:.1f}-{s['end']:.1f}] {s.get('text','')[:180]}" for s in ctx["selected"][:6]])
                context_parts.append("Selected segments preview:\n" + preview)
            context_for_model = "\n\n".join(context_parts)
        else:
            context_for_model = "No processed video context available. Please upload and process a video first."

        prompt = f"""You are an assistant answering questions about a video's content. Use the provided context from the transcript, the detailed summary, and selected segment previews to answer precisely.

Context:
{context_for_model}

User question: {user_q}

Answer (concise, but explain if the user asks to teach or elaborate):"""

        answer = None
        if GENAI_AVAILABLE and GEMINI_MODEL is not None:
            try:
                resp = GEMINI_MODEL.generate_content(prompt)
                if hasattr(resp, "text"):
                    answer = resp.text
                elif isinstance(resp, dict) and "candidates" in resp:
                    answer = resp["candidates"][0]["output"]
                else:
                    answer = str(resp)
            except Exception as e:
                answer = f"Gemini call failed: {e}"
        else:
            # fallback: local sentence-transformer retrieval from summary
            if SENTS_AVAILABLE and ctx and ctx.get("summary"):
                sentences = [s.strip() for s in ctx["summary"].split(". ") if s.strip()]
                if sentences:
                    q_emb = SENT_MODEL.encode([user_q], convert_to_numpy=True)
                    s_embs = SENT_MODEL.encode(sentences, convert_to_numpy=True)
                    sims = np.dot(s_embs, q_emb.T).squeeze()
                    top_idx = list(np.argsort(-sims)[:3])
                    sel = [sentences[i] for i in top_idx if i < len(sentences)]
                    answer = " ".join(sel) if sel else "I couldn't find a relevant passage in the summary."
                else:
                    answer = "No summary sentences available to answer from."
            else:
                answer = "Gemini not configured and no local embeddings available. Set GEMINI_API_KEY or install sentence-transformers."

        st.session_state["chat_history"].append({"role": "assistant", "text": answer})
        st.chat_message("assistant").write(answer)

    # chat controls
    if st.button("Reset Chat"):
        st.session_state["chat_history"] = []
        st.success("Chat cleared.")

    if st.button("Clear processed sessions (delete temp files)"):
        # remove tmp dirs
        for s in st.session_state["video_sessions"]:
            td = s.get("tmp_dir")
            try:
                if td:
                    shutil.rmtree(td)
            except Exception:
                pass
        st.session_state["video_sessions"] = []
        st.session_state["current_session"] = None
        st.success("All sessions cleared and temp files removed.")

# Footer: availability
st.markdown("---")
st.markdown(
    f"""
**Availability:**  
- Whisper installed: {'✅' if WHISPER_AVAILABLE else '❌'}  
- Selected Whisper model: `{whisper_choice}` {'(loaded)' if (WHISPER_AVAILABLE and WHISPER_MODEL is not None) else ''}  
- BLIP available: {'✅' if BLIP_AVAILABLE else '❌'}  
- sentence-transformers: {'✅' if SENTS_AVAILABLE else '❌'}  
- MoviePy: {'✅' if MOVIEPY_AVAILABLE else '❌'}  
- Gemini chat: {'✅' if (GENI_AVAILABLE := (GENAI_AVAILABLE and GEMINI_MODEL is not None)) else '❌ (set GEMINI_API_KEY)'}
"""
)
