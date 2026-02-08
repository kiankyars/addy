# Addy

Generate podcast-style ad reads from a YouTube video: transcript → placement → copy → TTS. Outputs **one MP3 per sponsor** in your config.

## How it works

1. **Input** — YouTube URL or video ID (from config).
2. **Transcript** — Fetched via `youtube_transcript_api`.
3. **Ad placement** — LLM (Claude or Gemini) picks where each sponsor fits.
4. **Ad copy** — For each placement, the LLM writes segue, content, and exit.
5. **TTS** — Cartesia Sonic turns each ad into an MP3 using the voice ID in config.

You get **N ad audio files** (N = number of sponsors). No CLI arguments — everything is read from **dwarkesh.json** at runtime.

## Run

```bash
# .env: ANTHROPIC_API_KEY, CARTESIA_API_KEY; for Gemini add GOOGLE_API_KEY
uv sync
addy
```

Looks for **dwarkesh.json** in the current directory, then the project root. Writes MP3s to the `output` directory specified in the config.

## Config (dwarkesh.json)

All settings in one file:

- **video** — YouTube URL or video ID.
- **output** — Directory for ad MP3s (default: `addy_output`).
- **model** — `claude` or `gemini`.
- **voice** — Cartesia voice ID for TTS.
- **sponsors** — Array of `id`, `url`, `title`, `content`, `tags`.

## Stack

Python 3.11+, uv. LLM: Claude or Gemini 2.0 Flash. Cartesia Sonic (TTS).
