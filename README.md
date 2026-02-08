# Addy — Cartesia x Anthropic voice agent

Voice agent built with [Cartesia Line SDK](https://github.com/cartesia-ai/line) and [Anthropic Claude](https://anthropic.com), for the Cartesia x Anthropic hackathon.

- **Stack:** Line (voice agent framework), Cartesia Sonic (TTS), Anthropic Claude Haiku 4.5 (reasoning).
- **Tools:** `end_call`, `web_search` (situationally aware).

## Setup

1. [Cartesia account](https://play.cartesia.ai) and [Anthropic API key](https://platform.claude.com/settings/keys).
2. Install [Cartesia CLI](https://docs.cartesia.ai/line/start-building/quickstart) and [uv](https://docs.astral.sh/uv/):

   ```bash
   curl -fsSL https://cartesia.sh | sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. From repo root:

   ```bash
   cd voice-agent
   cp .env.example .env   # set ANTHROPIC_API_KEY
   uv sync
   ANTHROPIC_API_KEY=your-key PORT=8000 uv run python main.py
   ```

4. In another terminal: `cartesia chat 8000` (text) or use [Cartesia Playground](https://play.cartesia.ai/agents) to deploy and call.

## License

Apache 2.0.
