# Addy

Generate podcast-style ad reads from any YouTube video; Addy ingests a transcript, clones the voice, lets claude choose natural ad breaks, writes the ad copy, and voices it with Cartesia TTS.

## Features
- Transcript to ads in one command: run `addy` and get finished MP3s in your output folder.
- Dual LLM support: Claude (agentic placement with Exa research) or Gemini for placement/copy.
- Voice cloning ready: uses a Cartesia voice ID; optional emotion + speed controls per ad.
- Multi-sponsor aware: loads sponsors from config, places each at most once, skips if no good fit.
- XML-safe prompting/parsing with fallbacks to handle messy model output.

## Requirements
- Python deps: `uv sync` installs everything.
- System: `ffmpeg` (brew install ffmpeg).
- Env: create `.env` (see `sample.env` if present) with `ANTHROPIC_API_KEY`, `CARTESIA_API_KEY`, `GOOGLE_API_KEY`, optional `EXA_API_KEY`.

## Quickstart
```bash
uv sync
source .venv/bin/activate
addy  # reads dwarkesh.json in CWD, then project root
```

## Configuration (`dwarkesh.json` example)
```json
{
  "video": "https://www.youtube.com/watch?v=BYXbuik3dgA",
  "output": "output",
  "model": "gemini",        // "claude" or "gemini"
  "voice": "<cartesia-voice-id>",
  "sponsors": [
    {"id": "acme", "url": "https://acme.com", "title": "Acme", "content": "Short ad blurb", "tags": ["tag1", "tag2"]}
  ]
}
```
- `video`: full YouTube URL or 11-char ID.
- `output`: directory for generated MP3s.
- `model`: LLM choice for placement + copy.
- `voice`: Cartesia voice ID used for all ads.
- `sponsors`: array of sponsors; each may include `tags` for better placement.

## Pipeline
1) **Transcript** — `youtube_transcript_api` fetches timestamped segments.
2) **Ad placement** — Claude (agentic with optional Exa web research) or Gemini selects segment breaks.
3) **Ad copy** — LLM writes segue/content/exit + emotion/speed for TTS; picks best of three variations.
4) **TTS** — Cartesia Sonic 3 renders MP3s using the configured voice; files saved to `output`.

## Code map
- `addy/cli.py` — console entrypoint (`addy`); loads config, kicks off the pipeline, writes MP3s.
- `pipeline.py` — orchestrates transcript → placement → copy → TTS, tracks job state.
- `interface.py` — all external calls (LLMs, YouTube, Cartesia) and XML prompt/parse helpers.

## End-to-end

1. Load configuration.
2. Fetch YouTube transcript.
3. Use LLM to select ad placement(s).
4. LLM generates 3 ad copy variations per placement.
5. Cartesia TTS renders MP3s.
6. MP3s and ads.json are written to the output directory.

