# Addy

Generate podcast-style ad reads from a YouTube video: transcript → placement → copy → TTS. Outputs **one MP3 per sponsor** in your config (e.g. three sponsors → three audio files).

## How it works

1. **Input** — YouTube URL or video ID.
2. **Transcript** — Fetched via `youtube_transcript_api`.
3. **Ad placement** — LLM (Claude or Gemini) picks where each sponsor fits in the transcript.
4. **Ad copy** — For each placement, the LLM writes segue, content, and exit in the show’s tone.
5. **TTS** — Cartesia Sonic turns each ad into an MP3 (default Dwarkesh voice).

You get **N ad audio files** (N = number of sponsors in your config). No stitching, no frontend.

## Run (CLI)

```bash
# .env: ANTHROPIC_API_KEY, CARTESIA_API_KEY; for Gemini add GOOGLE_API_KEY
uv sync
addy "https://www.youtube.com/watch?v=BYXbuik3dgA"
```

Options: `-o DIR` (output), `--sponsors PATH`, `--model claude|gemini` (default: claude).

Examples:

```bash
addy BYXbuik3dgA -o ./ads
addy BYXbuik3dgA --model gemini
addy BYXbuik3dgA --sponsors config/sponsors.json
```

## Config

`config/sponsors.json` — array of `id`, `url`, `title`, `content`, `tags`.

## Stack

Python 3.11+, uv. LLM: Claude (default) or Gemini 2.0 Flash (`--model gemini`). Cartesia Sonic (TTS).
