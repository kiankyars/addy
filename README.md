## How it works

1. **Input** — You paste a **YouTube URL or video ID** in the UI.
2. **Transcript** — Backend uses **youtube_transcript_api** to fetch the captions.
3. **Ad placement** — **Anthropic Claude** reads the transcript and **sponsors** (`config/sponsors.json`) and picks where each ad fits.
4. **Ad copy** — For each placement, Claude writes **segue**, **content**, and **exit** in the tone of the show.
5. **TTS** — **Cartesia Sonic** turns each ad into audio (default Dwarkesh voice).
6. **Stitch** — You pick one generated ad; the backend inserts it into the original (audio from **yt-dlp**); you download the stitched file.

**Stack:** One app. FastAPI serves the API and the static UI (Next.js export). No SQL; sponsors from JSON.

---

## Run

### 1. Backend deps + env

```bash
cd server/recorded
cp sample.env .env
# Set ANTHROPIC_API_KEY and CARTESIA_API_KEY
uv pip install -r requirements.txt
```

### 2. Build the UI (once)

```bash
cd server/recorded/web
pnpm install
pnpm build
```

### 3. Start the app (API + UI on one port)

```bash
cd server/recorded
uvicorn app:app --host 0.0.0.0 --port 4001
```

Open http://localhost:4001 — same origin for UI and API.

Optional: edit `config/sponsors.json` (or pass `sponsors_config` in `POST /process` body).

---

## Repo layout

| Path | Purpose |
|------|--------|
| **server/recorded/** | FastAPI app: API routes + static serve of `web/out`. Pipeline, config, voices. |
| **server/recorded/web/** | Next.js UI (YouTube input, poll, pick ad, stitch, download). Build with `pnpm build` → `web/out`. |
