# Adible

Weave ads into podcasts: upload a recording, get AI-placed sponsor reads with natural segues, then download a stitched version.

---

## How it works

1. **Upload** — You upload an audio file (e.g. podcast episode) via the podcast UI.
2. **Transcribe** — Backend uses **OpenAI Whisper** to get timestamped segments.
3. **Ad placement** — **Anthropic Claude** reads the transcript and a list of ads (from DB or Google Sheet) and picks where each ad fits best.
4. **Ad copy** — For each placement, Claude writes a **segue** (lead-in), **content** (sponsor read), and **exit** (back to show), in the tone of the surrounding speech.
5. **Voice choice** — Claude picks which **Cartesia** voice best matches the segment (or you use a single voice).
6. **TTS** — **Cartesia Sonic** turns each ad text into audio (MP3).
7. **Stitch** — Backend inserts the ad audio after the chosen segment; you pick one generated ad in the UI and download the full stitched file.

**Stack:** SQLite (audio metadata, ads, generated ads), FastAPI (backend), Next.js (podcast UI). Ads can be seeded from `migrations/seed_db_tables.sql` or synced from a Google Sheet (optional; see `server/recorded/utils.py` and `SHEET_ID` / `credentials.json`).

---

## Step-by-step

### 1. Backend (API + processing)

```bash
cd server/recorded
```

Create `.env` from the sample and set API keys:

```bash
cp sample.env .env
```

Edit `.env`:

- `OPENAI_API_KEY` — for Whisper transcription
- `ANTHROPIC_API_KEY` — for ad placement, copy, and voice choice
- `CARTESIA_API_KEY` — for TTS

Optional: `SHEET_ID` and `credentials.json` if you want to pull ads from a Google Sheet instead of (or in addition to) the seeded DB ads.

Install dependencies and run the API:

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 4001
```

Create the DB and seed ads (once, from `server/recorded`):

```bash
sqlite3 recorded.db < migrations/seed_db_tables.sql
```

### 2. Frontend (podcast UI)

In a new terminal:

```bash
cd podcast
pnpm install
pnpm dev
```

Open the URL shown (e.g. http://localhost:3000).

### 3. Use the flow

1. Upload an audio file (e.g. MP3) on the podcast page. The backend will transcribe it, pick ad placements, generate copy, and synthesize ad audio.
2. When processing is done, open the upload’s generated-ads view. Listen to the options (each is a short clip: context + ad + exit).
3. Choose one generated ad and request “stitch.” The backend inserts that ad into the original at the right timestamp.
4. Download the stitched audio from the UI when it’s ready.

---

## Repo layout

| Path | Purpose |
|------|--------|
| **server/recorded/** | Backend: FastAPI app, Whisper + Anthropic + Cartesia, DB, ad logic, stitching. |
| **podcast/** | Next.js app: upload, list uploads, view/pick generated ads, trigger stitch, download. |

The podcast app talks to the backend at `http://localhost:4001` (see `podcast/src/app/uploader.tsx` and related fetch calls).
