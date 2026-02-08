import os
from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv
from anthropic import Anthropic
from cartesia import Cartesia
from pydantic import BaseModel
import json
import xml.etree.ElementTree as ET
import uuid

from youtube_transcript_api import YouTubeTranscriptApi
from voices import DEFAULT_VOICE_ID

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
cartesia_client = Cartesia(api_key=CARTESIA_API_KEY)

LLM_MODEL = Literal["claude", "gemini"]
_gemini_model = None


def _get_gemini_model():
    global _gemini_model
    if _gemini_model is None:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        _gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    return _gemini_model


def _llm_completion(prompt: str, stop_sequences: list[str], model: LLM_MODEL) -> str:
    if model == "claude":
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
            stop_sequences=stop_sequences,
        )
        return (response.content[0].text or "").rstrip()
    if model == "gemini":
        genai = _get_gemini_model()
        from google.generativeai.types import GenerationConfig
        config = GenerationConfig(stop_sequences=stop_sequences, max_output_tokens=2048)
        response = genai.generate_content(prompt, generation_config=config)
        return (response.text or "").rstrip()
    raise ValueError(f"Unknown model: {model}")

DEFAULT_SPONSORS_PATH = Path(__file__).resolve().parent / "config" / "sponsors.json"


class Advertisement(BaseModel):
    id: str
    url: str
    title: str
    content: str
    tags: list[str]


class TranscriptionSegment(BaseModel):
    no: int
    start: float
    end: float
    text: str


class AdvertisementPlacement(BaseModel):
    transcription_segment: TranscriptionSegment
    determined_advertisement: Advertisement


class GeneratedAdvertisementText(BaseModel):
    segue: str
    content: str
    exit: str


def get_youtube_transcript(video_id: str) -> list[TranscriptionSegment]:
    """Fetch timestamped transcript from YouTube. video_id is the 11-char ID from the URL."""
    transcript = YouTubeTranscriptApi().fetch(video_id)
    raw = transcript.to_raw_data()
    return [
        TranscriptionSegment(
            no=i,
            start=float(entry["start"]),
            end=float(entry["start"]) + float(entry["duration"]),
            text=entry["text"].strip(),
        )
        for i, entry in enumerate(raw)
    ]


def load_sponsors(config_path: str | Path | None = None) -> list[Advertisement]:
    path = Path(config_path or DEFAULT_SPONSORS_PATH)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    sponsors = data.get("sponsors", data) if isinstance(data, dict) else data
    return [
        Advertisement(
            id=str(s.get("id", i)),
            url=str(s.get("url", "")),
            title=str(s.get("title", "")),
            content=str(s.get("content", "")),
            tags=[str(t) for t in s.get("tags", [])],
        )
        for i, s in enumerate(sponsors)
    ]

def _determine_ad_placement(
    transcription_segments: list[TranscriptionSegment],
    available_ads: list[Advertisement],
    model: LLM_MODEL,
) -> List[AdvertisementPlacement]:

    transcription_segment_xml = "\n".join([
        f"<transcription_segment no='{segment.no}'>"
        f"<start>{segment.start}</start>"
        f"<end>{segment.end}</end>"
        f"<text>{segment.text}</text>"
        f"</transcription_segment>"
    for segment in transcription_segments])

    available_ad_xml = "\n".join([
        f"<advertisement id='{ad.id}'>"
        f"<url>{ad.url}</url>"
        f"<title>{ad.title}</title>"
        f"<content>{ad.content}</content>"
        f"<tags>{','.join(ad.tags)}</tags>"
        f"</advertisement>"
    for ad in available_ads])

    prompt = f"""You are a natural advertisment placement expert. You will be working with the transcript of a audio recording which might be a podcast, youtube video, or any other recorded audio.
    You will be given the transcript of a audio recording and a list of advertisements that can be placed in the audio recording.
    Your job is to determine the transcription segments along with the advertisement that should be placed in the audio recording.

    Please respond in the format provided between the <example></example> tags.
    <example>
    <response>
    <possible_ad_placements>
    <placement>
    <transcription_segment no='segment number'/> <!-- The segment after which the advertisement should be placed -->
    <advertisement id='advertisement id'/> <!-- The advertisement that should be placed -->
    </placement>
    </possible_ad_placements>
    </response>
    </example>

    Please focus on following the following rules:
    1. You need to determine what ad can be placed in after a segment such that it is a part of the conversation.
    2. If no natural placement is found, then return an empty <possible_ad_placements> tag.
    3. Do not restrict to a single language in a multiple language audio recording.


    Here is the transcription of the audio recording:
    <transcription_segments>
    {transcription_segment_xml}
    </transcription_segments>

    Here is the list of advertisements that can be placed in the audio recording:
    <advertisements>
    {available_ad_xml}
    </advertisements>
    """

    raw = _llm_completion(prompt, ["</response>"], model)
    xml_response = "<response>" + raw + "</response>"
    root = ET.fromstring(xml_response)

    ad_placements = []
    for placement in root.findall('.//placement'):
        segment_no = int(placement.find('transcription_segment').get('no'))
        ad_id = placement.find('advertisement').get('id')

        transcription_segment = transcription_segments[segment_no]
        determined_advertisement = next(ad for ad in available_ads if ad.id == ad_id)

        ad_placements.append(AdvertisementPlacement(
            transcription_segment=transcription_segment,
            determined_advertisement=determined_advertisement
        ))

    return ad_placements


def determine_ad_placement(
    transcription_segments: list[TranscriptionSegment],
    available_ads: list[Advertisement],
    model: LLM_MODEL = "claude",
) -> List[AdvertisementPlacement]:
    return _determine_ad_placement(transcription_segments, available_ads, model)

def parse_advertisement_xml(xml_content: str) -> List[GeneratedAdvertisementText]:
    root = ET.fromstring(xml_content)
    advertisements = []

    for ad in root.findall('.//advertisement'):
        segue = ad.find('segue').text
        content = ad.find('content').text
        exit = ad.find('exit').text
        advertisements.append(GeneratedAdvertisementText(segue=segue, content=content, exit=exit))

    return advertisements

def _generate_advertisement_text(
    ad_placement: AdvertisementPlacement,
    surrounding_segments: list[TranscriptionSegment],
    model: LLM_MODEL,
) -> List[GeneratedAdvertisementText]:

    ad_placement_xml = f"""
<advertisement_placement>
    <transcription_segment no='{ad_placement.transcription_segment.no}'>
        <text>{ad_placement.transcription_segment.text}</text>
    </transcription_segment>
    <advertisement>
        <title>{ad_placement.determined_advertisement.title}</title>
        <content>{ad_placement.determined_advertisement.content}</content>
    </advertisement>
</advertisement_placement>
"""
    
    surrounding_segments_xml = "\n".join([
        f"<transcription_segment no='{segment.no}'>"
        f"<text>{segment.text}</text>"
        f"</transcription_segment>"
    for segment in surrounding_segments])


    prompt = f"""
You are a natural performance marketing expert. You know you way around the ad placment space. The finesse you have with ad placement is such that you can fit any advertisement into a segment of a podcast, youtube video, or any other recorded audio.
You will be given the advertisement placement and the surrounding segments of the audio recording.
Your job is to finnse the advertisement placement such that it is a part of the conversation and is not intrusive.

Here are some examples of how you can finnse the advertisement placement:
<examples>
<advertisement>
<segue>
Considering how much everything else has gone up over the last few years<break time="0.1s" /> like haircuts I guess we shouldn't be too surprised even if it is a tough pill to swallow like today's segue to our sponsor,<break time="0.1s" /> The Big Thunder Game.. (advertisement content here)
</segue>
<content>
The Big Thunder Game brings an exciting new way to play MMORPGs with friends regardless of their ability to play.
</content>
<exit>
(add some context here that continues to the next segment),<break time="0.2s" /> now back to the show.
</exit>
</advertisement>
<advertisement>
<segue>
if you're planning your next PC build<break time="0.1s" /> then consider checking out our sponsor VIP SCD key
</segue>
<content>
Their Windows 10 and 11 OEM Keys sell for a fraction of retail <break time="0.1s" /> and will unlock the full potential of your OS it'll also remove.. (more content here)
</content>
<exit>
use VIP SCD key on your next PC build <break time="0.2s" /> and now lets get back to this PC.
</exit>
</advertisement>
<advertisement>
<segue>
क्या आप रचनात्मकता और मज़े की तलाश में हैं? <break time="0.1s" /> तो आइए जानते हैं हमारे प्रायोजक LEGO के बारे में
</segue>
<content>
LEGO के नए साहसिक सेट्स के साथ अपनी कल्पना को उड़ान दें। <break time="0.1s" /> हर सेट में उच्च गुणवत्ता वाले ब्रिक्स, विस्तृत निर्देश पुस्तिका, और असीमित रचनात्मक संभावनाएं हैं। चाहे आप शुरुआती बिल्डर हों या अनुभवी LEGO प्रेमी, हमारे पास हर कौशल स्तर के लिए सेट हैं। इस महीने के विशेष ऑफर में सभी LEGO सिटी और LEGO टेकनिक सेट्स पर 20% की छूट पाएं। साथ ही, <break time="0.1s" /> VIP सदस्यों के लिए अतिरिक्त बोनस प्वाइंट्स!
</content>
<exit>
LEGO के साथ अपनी रचनात्मक यात्रा शुरू करें,<break time="0.1s" /> और अब वापस लौटते हैं हमारी मुख्य कहानी की ओर।
</exit>
</advertisement>
</examples>

Here is the advertisement placement:
<advertisement_placement>
    {ad_placement_xml}
</advertisement_placement>

Here are the surrounding segments of the audio recording:
<surrounding_segments>
{surrounding_segments_xml}
</surrounding_segments>

Please follow the below rules when generating the advertisements:
1. The advertisement should be a part of the conversation and not intrusive.
2. The advertisement content should be very short and to the point.
3. The ending of the advertisement should be a segue back to the show.
4. Always provide three variations of the same advertisement.
5. Try to match the language used by the segments.

Please respond in the format provided between the <example></example> tags.
<example>
<response>
<advertisements>
<advertisement>
<segue>
finnse your way here
</segue>
<content>
keep it short and concise
</content>
<exit>
the exit of the content
</exit>
</advertisement>
</advertisements>
</response>
</example>
"""
    
    raw = _llm_completion(prompt, ["</response>"], model)
    return parse_advertisement_xml("<response>" + raw + "</response>")


def generate_advertisements(
    ad_placement: AdvertisementPlacement,
    transcription_segments: list[TranscriptionSegment],
    model: LLM_MODEL = "claude",
) -> List[GeneratedAdvertisementText]:  
    surrounding_segments = []
    segment_nos = {segment.no: segment for segment in transcription_segments}

    for offset in [-2, -1, 1, 2]:
        target_no = ad_placement.transcription_segment.no + offset
        if target_no in segment_nos:
            surrounding_segments.append(segment_nos[target_no])

    return _generate_advertisement_text(ad_placement, surrounding_segments, model)

def generate_advertisement_audio(advertisement_text: str, voice_id: str = None, file_path: str = None) -> str:
    voice_id = voice_id or os.getenv("CARTESIA_VOICE_ID") or DEFAULT_VOICE_ID
    chunks = list(
        cartesia_client.tts.bytes(
            model_id="sonic-2",
            transcript=advertisement_text,
            voice={"id": voice_id},
            language="en",
            output_format={"container": "mp3", "sample_rate": 44100, "bit_rate": 128000},
        )
    )
    audio_bytes = b"".join(chunks)
    save_file_path = file_path or f"{uuid.uuid4()}.mp3"
    with open(save_file_path, "wb") as f:
        f.write(audio_bytes)
    return save_file_path

