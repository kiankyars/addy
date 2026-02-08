# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Addy generates podcast-style ad reads from a YouTube video. Pipeline: transcript → LLM ad placement → LLM ad copy → Cartesia TTS. Outputs one MP3 per sponsor.

## Commands

```bash
uv sync          # install dependencies
addy             # run the pipeline (reads dwarkesh.json from cwd, then project root)
```

No CLI arguments — all config comes from **dwarkesh.json** (video URL, output dir, model choice, voice ID, sponsors array).

## Environment

Requires a `.env` file (see `sample.env`):
- `ANTHROPIC_API_KEY` — for Claude model
- `CARTESIA_API_KEY` — for TTS
- `GOOGLE_API_KEY` — for Gemini model

System dependency: `brew install ffmpeg`

## Architecture

Three layers, all at repo root:

- **`addy/cli.py`** — Entry point (`addy` command). Loads config, extracts video ID, runs pipeline, writes MP3s to output dir. Uses Rich for console UI.
- **`pipeline.py`** — Job orchestration. Creates a `Job`, calls interface functions in sequence (transcript → placement → copy → TTS), collects `GeneratedAd` results with audio bytes.
- **`interface.py`** — All external API calls and LLM logic. Contains:
  - `_llm_completion()` — unified wrapper for Claude (`claude-sonnet-4-20250514`) and Gemini (`gemini-3-flash-preview`)
  - `get_youtube_transcript()` — fetches via `youtube_transcript_api`
  - `determine_ad_placement()` — LLM picks which sponsor fits after which transcript segment (XML prompt/response format)
  - `generate_advertisements()` — LLM writes segue/content/exit for each placement (3 variations), using ±2 surrounding segments for context
  - `generate_advertisement_audio()` — Cartesia Sonic TTS, returns MP3 file path

The `addy/` package is a thin wrapper so `uv sync` can build a console script; core logic lives in `interface.py` and `pipeline.py` at the repo root. `cli.py` uses `sys.path.insert` to import from root.

## Key data models (Pydantic, in interface.py)

- `Advertisement` — sponsor info (id, url, title, content, tags)
- `TranscriptionSegment` — timestamped transcript chunk (no, start, end, text)
- `AdvertisementPlacement` — pairs a segment with an ad
- `GeneratedAdvertisementText` — LLM output (segue, content, exit)

## LLM prompts

Both LLM calls use XML-formatted prompts and expect XML responses parsed with `xml.etree.ElementTree`. Stop sequence is `</response>`.
