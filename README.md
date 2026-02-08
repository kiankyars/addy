# Adible

Weave ads into podcasts: upload a recording → transcribe → AI places and generates ad copy → Cartesia TTS → stitched audio.

## Parts

| Part | Role |
|------|------|
| **server/recorded/** | Backend: Whisper (transcription), Anthropic (ad placement + copy + voice choice), Cartesia (TTS). FastAPI on port 4001. |
| **podcast/** | Next.js UI: upload audio, view generated ads, pick one, download stitched podcast. |

## Run

**Backend** (from repo root):

```bash
cd server/recorded
pip install -r requirements.txt   # or uv sync if you add pyproject.toml
cp sample.env .env                # set OPENAI_API_KEY, ANTHROPIC_API_KEY, CARTESIA_API_KEY
uvicorn app:app --host 0.0.0.0 --port 4001
```

**Frontend**:

```bash
cd podcast
pnpm install && pnpm dev
```

Then open the app and upload an audio file; it will call `http://localhost:4001` for upload and processing.
