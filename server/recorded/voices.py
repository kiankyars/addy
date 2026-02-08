from pydantic import BaseModel


class Voice(BaseModel):
    id: str
    description: str
    language: str


# Cartesia Sonic voices (preset IDs); use play.cartesia.ai/voices for more or clone your own
AVAILABLE_VOICES = [
    Voice(id="a0e99841-438c-4a64-b679-ae501e7d6091", description="Young conversational male, podcast/casual.", language="English"),
    Voice(id="f9836c6e-a0bd-460e-9d3c-f7299fa60f94", description="Sharp, modern, fast-paced.", language="English"),
    Voice(id="694f9389-aac1-45b6-b726-9d9369183238", description="Smooth assertive female, news/conversational.", language="English"),
]
