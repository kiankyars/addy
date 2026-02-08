# Addy

Generate podcast-style ad reads from a YouTube video: transcript → placement → copy → TTS. Outputs **one MP3 per sponsor** in your config (e.g. three sponsors → three audio files).

## How it works

1. **Input** — YouTube URL or video ID.
2. **Transcript** — Fetched via `youtube_transcript_api`.
3. **Ad placement** — Claude picks where each sponsor fits in the transcript.
4. **Ad copy** — For each placement, Claude writes segue, content, and exit in the show’s tone.
5. **TTS** — Cartesia Sonic turns each ad into an MP3 (default Dwarkesh voice).

You get **N ad audio files** (N = number of sponsors in your config). No stitching, no frontend.

## Run (CLI)

```bash
# .env with ANTHROPIC_API_KEY and CARTESIA_API_KEY
uv sync
addy "https://www.youtube.com/watch?v=BYXbuik3dgA"
```

Ads are written to `addy_output/` by default. Options:

- `-o DIR` — output directory (default: `addy_output`)
- `--sponsors PATH` — sponsors JSON (default: `config/sponsors.json`)

Example:

```bash
addy BYXbuik3dgA -o ./ads --sponsors config/sponsors.json
```

## Config

`config/sponsors.json` — array of `id`, `url`, `title`, `content`, `tags`. Default three: Mercury, Jane Street, Labelbox.

## Stack

Python 3.11+, uv. Anthropic (placement + copy), Cartesia Sonic (TTS). No DB, no server.
