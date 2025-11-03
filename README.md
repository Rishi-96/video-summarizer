# 🎬 AI Video Summarizer — Context-Aware Multimodal Summarization

An advanced **AI-powered video summarizer** that condenses lengthy videos into short, context-rich summaries — both **textual** and **visual**.  
This project intelligently identifies key video segments, generates a summarized version (~30% of the original length), and provides a **detailed 10+ line summary**.  
It also includes an interactive **Gemini-powered chatbox** to ask content-related queries **without re-running the summarization process**.

---

## 🚀 Project Overview

The goal is to help users quickly grasp the essence of any video without watching it entirely.  
It integrates **speech-to-text**, **semantic understanding**, **summarization**, and **video editing** pipelines using state-of-the-art AI models.

### 🔹 Capabilities
- Multilingual **speech transcription**
- Context-based **text & video summarization**
- Frame-level **visual understanding**
- Interactive **Gemini chatbot**
- Modern **Streamlit UI**

---

## 🧩 How It Works

🎞 Upload Video (.mp4)
   │
   ▼
🎧 Audio Extraction (MoviePy)
   │
   ▼
🗣 Speech-to-Text (Whisper)
   │
   ▼
🖼 Frame Sampling + Captions (BLIP)
   │
   ▼
🧠 Semantic Ranking (SentenceTransformers)
   │
   ▼
📝 Text Summary Generation (mBART / BART)
   │
   ▼
🎬 Video Stitching (MoviePy)
   │
   ▼
💬 Gemini Chat for Q&A

---

## ⚙️ Key Features

| Feature | Description |
|----------|-------------|
| 🧠 **Context-aware summarization** | Extracts meaningful portions of video using sentence embeddings |
| 🎧 **Whisper transcription** | Converts multilingual audio into text |
| 📝 **mBART / BART summarization** | Produces accurate 10+ line text summary |
| 🎥 **Video summarization** | Merges relevant segments into a short video (~30%) |
| 💬 **Gemini chatbot** | Answers queries about video content without repeating the process |
| 🌍 **Multilingual support** | Works with English, Hindi, Marathi, etc. |
| 🖥️ **Streamlit UI** | Easy-to-use modern interface with dark theme |

---

## 🧠 Models Used

| Model | Purpose | Key Algorithm | Why Chosen |
|--------|----------|----------------|-------------|
| **Whisper (OpenAI)** | Speech → Text | Transformer encoder-decoder | High accuracy multilingual transcription |
| **SentenceTransformer (MiniLM)** | Segment ranking | Siamese BERT | Efficient semantic similarity computation |
| **mBART (Facebook)** | Text summarization | Transformer Seq2Seq | Supports 50+ languages |
| **BART (Facebook)** | English fallback summarizer | Denoising autoencoder | High-quality abstractive summaries |
| **BLIP (Salesforce)** | Frame captioning | Vision Transformer + GPT2 | Adds visual context |
| **Gemini (Google)** | Chat answering | Multimodal transformer | Contextual and conversational AI |
| **MoviePy** | Video editing | ffmpeg backend | Script-based video merging and subtitle support |

---

## 🧰 Libraries & Tech Stack

| Category | Tools |
|-----------|-------|
| **UI** | Streamlit |
| **Audio & Video** | ffmpeg, MoviePy, OpenCV |
| **NLP & ML** | Transformers, SentenceTransformers, Whisper, mBART |
| **Chat Integration** | Google Gemini API |
| **Language Detection** | langdetect |
| **Environment** | Python 3.10+ |

---

## 🧱 Project Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/video-summarizer.git
cd video-summarizer
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # For Windows
# OR
source venv/bin/activate  # For Mac/Linux
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up Gemini API Key
```bash
set GEMINI_API_KEY=your_api_key_here        # Windows PowerShell
# OR
export GEMINI_API_KEY=your_api_key_here     # Mac/Linux
```

### 5️⃣ Run the App
```bash
streamlit run src/frontend.py
```

---

## 🧩 Output Example


| Type | Example Output |
|-------|----------------|
| **Original Video** | 10-min tutorial |
| **Summarized Video** | 3-min video (context-preserving) |
| **Text Summary** | 10+ line coherent summary |
| **Gemini Chat** | "Explain the backend process" → Instant, accurate answer |

---

## 🧠 How Each Module Works

### 🔹 Whisper
- Converts speech to text.
- Handles noise, accents, and multilingual speech.
- Trained on 680K hours of diverse audio using a transformer architecture.

### 🔹 SentenceTransformer
- Generates embeddings for each transcript segment.
- Finds segments most related to the user query.
- Uses contrastive Siamese BERT architecture for similarity ranking.

### 🔹 mBART / BART
- Generates a natural, coherent textual summary.
- mBART handles multilingual data; BART is fallback for English-only videos.
- Works on encoder-decoder transformer architecture.

### 🔹 MoviePy
- Extracts, trims, and merges relevant video segments.
- Adds subtitles from the summarized text.
- Output: smooth and context-preserving summarized video.

### 🔹 Gemini Chat
- Responds to user queries about the summarized content.
- Uses stored summary + transcript context (no reprocessing).
- Multilingual responses supported.

---

## 💡 Performance Tips

- Use **Whisper small** or **tiny** for faster CPU processing.  
- GPU significantly improves transcription & summarization speed.  
- Adjust `top_k` and summary ratio for desired length.  
- Avoid re-uploading the same video; cached models speed up re-runs.

---

## 🎯 Future Improvements

- [ ] Add option to control summary ratio (e.g., 20–50%)  
- [ ] Integrate Whisper-large for GPU environments  
- [ ] Multi-user session support  
- [ ] Export subtitles in `.srt`  
- [ ] Add Hindi summarization translation toggle  

---

## 👨‍💻 Author

**Rishi Jain**  
📍 Pune, India  
💼 Data Analyst & AI Developer  
📧 rishij6388@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/rishi-jain-b9b6b4259/)

---

## 🧾 License

This project is released under the **MIT License**.  
You may freely use, modify, and distribute it with attribution.

---

## ⭐ Acknowledgments

Thanks to:
- **OpenAI Whisper** — Speech-to-text  
- **Meta AI (mBART/BART)** — Text summarization  
- **Google Gemini** — Chat interface  
- **Hugging Face** — Model hosting  
- **Streamlit** — User interface framework  

---

⭐ *If this project helps you, please give it a star on GitHub!*
