"""
Dwarkesh podcast voice agent — Cartesia Line + Anthropic.
Use a cloned voice at play.cartesia.ai/voices/create/clone and set CARTESIA_VOICE_ID.
"""
import os
from dotenv import load_dotenv

from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp
from line.llm_agent import LlmAgent, LlmConfig, end_call

load_dotenv()

DWARKESH_SYSTEM = """You are the host of a thoughtful long-form podcast in the style of Dwarkesh Patel.
You ask deep, specific questions and follow up on interesting points. You're curious about technology,
AI, startups, and big ideas. You let the guest (the caller) lead at times but steer toward substance.
Keep responses concise for voice — a few sentences at a time. Be warm and intellectually curious."""

INTRO = "Hey, welcome to the show. What's on your mind today?"


async def get_agent(env: AgentEnv, call_request: CallRequest):
    config = LlmConfig.from_call_request(call_request)
    if config.system_prompt is None or config.system_prompt.strip() == "":
        config = config.model_copy(update={"system_prompt": DWARKESH_SYSTEM})
    if config.introduction is None or config.introduction.strip() == "":
        config = config.model_copy(update={"introduction": INTRO})

    return LlmAgent(
        model="anthropic/claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        tools=[end_call],
        config=config,
    )


# Dwarkesh cloned voice for the demo
DWARKESH_VOICE_ID = "17523c4f-8ab7-43e1-9159-bc52487d77ad"


def pre_call_handler(call_request: CallRequest):
    from line.voice_agent_app import PreCallResult
    voice_id = os.getenv("CARTESIA_VOICE_ID") or DWARKESH_VOICE_ID
    return PreCallResult(config={"tts": {"voice": voice_id, "model": "sonic-2", "language": "en"}})


app = VoiceAgentApp(get_agent=get_agent, pre_call_handler=pre_call_handler)

if __name__ == "__main__":
    app.run()
