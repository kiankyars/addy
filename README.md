<p align="center">
    <img src="./misc/headphone.png" alt="Banner" height="300px" width="300px"/>
</p>

<h1 align="center">Adible</h1>
<h3 align="center">Weave ads seamlessly into conversational AI, podcasts and more.</h3>

## Stack (Cartesia x Anthropic)

- **voice-agent/** — Dwarkesh-style podcast voice agent (Cartesia Line + Anthropic Claude, optional voice cloning).
- **conversational-agent/** — Web landing + call CTA for the voice agent.
- **server/recorded/** — Podcast ad pipeline: Whisper (OpenAI) for transcription, Anthropic for ad placement/copy/voice choice, Cartesia for TTS.
- **podcast/** & **web/** — Upload and results UIs.

### Quick start (voice agent)

```bash
cd voice-agent && uv sync
ANTHROPIC_API_KEY=... PORT=8000 uv run python main.py
# In another terminal: cartesia chat 8000
```

See `voice-agent/README.md` for deploy and voice cloning.
