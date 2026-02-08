# Dwarkesh Podcast Voice Agent

Cartesia x Anthropic hackathon — voice agent as a thoughtful long-form podcast host (Dwarkesh style). Uses **Cartesia Line** + **Anthropic Claude**; supports **voice cloning** via Cartesia.

## Setup

1. [Cartesia account](https://play.cartesia.ai) + [API key](https://play.cartesia.ai/keys)
2. [Anthropic API key](https://platform.anthropic.com/settings/keys)
3. (Optional) [Clone a voice](https://play.cartesia.ai/voices/create/clone) and set `CARTESIA_VOICE_ID`

## Run locally

```bash
uv sync
ANTHROPIC_API_KEY=your-key PORT=8000 uv run python main.py
```

In another terminal:

```bash
cartesia chat 8000   # text test
# or call your deployed agent by phone
```

## Deploy (Cartesia)

```bash
cartesia auth login
cartesia init   # link repo to your Cartesia agent
git push        # auto-deploys main
cartesia env set ANTHROPIC_API_KEY=your-key
cartesia env set CARTESIA_VOICE_ID=your-cloned-voice-id   # optional
```

Then call the agent from your phone (number in Cartesia dashboard).

## License

Apache-2.0
