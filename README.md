## How it works

1. **Input** — You paste a **YouTube URL or video ID** in the podcast UI.
2. **Transcript** — Backend uses **youtube_transcript_api** to fetch the captions (no upload, no STT).
3. **Ad placement** — **Anthropic Claude** reads the transcript and a list of **sponsors** (from `config/sponsors.json`) and picks where each ad fits.
4. **Ad copy** — For each placement, Claude writes a **segue**, **content**, and **exit** in the tone of the show.
5. **TTS** — **Cartesia Sonic** turns each ad into audio with the default Dwarkesh voice.
6. **Stitch** — You pick one generated ad in the UI; the backend inserts it into the original (audio from **yt-dlp**) and you download the stitched file.

**Stack:** No SQL. FastAPI (in-memory job store), Next.js (podcast UI). Sponsors are loaded from a JSON config file.

---

## Step-by-step

### 1. Backend

```bash
cd server/recorded
cp sample.env .env
```

Edit `.env`: set `ANTHROPIC_API_KEY` and `CARTESIA_API_KEY`.

Install and run:

```bash
uv pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 4001
```

Optional: edit `config/sponsors.json` to change sponsors (array of `id`, `url`, `title`, `content`, `tags`). You can pass a different path via `POST /process` body: `{ "video_id": "...", "sponsors_config": "/path/to/sponsors.json" }`.

### 2. Frontend

```bash
cd podcast
pnpm install
pnpm dev
```

Open the URL (e.g. http://localhost:3000).

### 3. Flow

1. Enter a YouTube URL or video ID and click Process. The backend fetches the transcript, downloads audio with yt-dlp, runs placement + copy + TTS in the background.
2. When processing is complete you’re taken to the generated-ads view. Listen to each option and pick one.
3. Click to stitch; then download the stitched audio from the result page.

---

## Repo layout

| Path | Purpose |
|------|--------|
| **server/recorded/** | FastAPI app, YouTube transcript + yt-dlp, sponsors JSON, Anthropic + Cartesia TTS, in-memory jobs, stitching. |
| **podcast/** | Next.js UI: YouTube input, poll job, pick ad, stitch, download. |

API base: `http://localhost:4001` (see `podcast/src/app/uploader.tsx` and related fetch calls).
