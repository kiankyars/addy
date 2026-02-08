## Dwarkesh Podcast Voice Agent (Web)

Landing page for the Cartesia + Anthropic voice agent. The agent runs in `../voice-agent` (Line SDK).

### Setup

```bash
cp .env.example .env
# Optional: set NEXT_PUBLIC_AGENT_PHONE after deploying voice-agent to Cartesia
pnpm install && pnpm dev
```

### Voice agent (separate)

See `../voice-agent/README.md` for running and deploying the agent. Call by phone or `cartesia chat 8000`.
