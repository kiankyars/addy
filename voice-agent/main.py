import os

from loguru import logger

from line.llm_agent import LlmAgent, LlmConfig, end_call, web_search
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

SYSTEM_PROMPT = """You are a friendly voice assistant built with Cartesia and Claude.

Be warm, concise, and natural. Keep replies to 1–2 sentences. Use contractions.
Never use lists or bullet points—speak in prose.
Use web_search when you need current information. Say a brief goodbye then use end_call when the conversation is clearly over."""

INTRODUCTION = "Hey! What would you like to talk about?"


async def get_agent(env: AgentEnv, call_request: CallRequest):
    logger.info(
        f"Starting call {call_request.call_id}. "
        f"Prompt: {call_request.agent.system_prompt or 'default'}, "
        f"Intro: {call_request.agent.introduction or 'default'}"
    )
    return LlmAgent(
        model="anthropic/claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        tools=[end_call, web_search],
        config=LlmConfig.from_call_request(
            call_request,
            fallback_system_prompt=SYSTEM_PROMPT,
            fallback_introduction=INTRODUCTION,
        ),
    )


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    print("Starting voice agent")
    app.run()
