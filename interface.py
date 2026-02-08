import os
from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv
from anthropic import Anthropic
from cartesia import Cartesia
from pydantic import BaseModel
import json
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape as xml_escape
import uuid

from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
cartesia_client = Cartesia(api_key=CARTESIA_API_KEY)

LLM_MODEL = Literal["claude", "gemini"]
_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _gemini_client


def _llm_completion(prompt: str, stop_sequences: list[str], model: LLM_MODEL) -> str:
    if model == "claude":
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=32000,
            messages=[{"role": "user", "content": prompt}],
            stop_sequences=stop_sequences,
        )
        return (response.content[0].text or "").rstrip()
    if model == "gemini":
        from google.genai import types
        client = _get_gemini_client()
        config = types.GenerateContentConfig(
            stop_sequences=stop_sequences,
            max_output_tokens=4096,
        )
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=config,
        )
        return (response.text or "").rstrip()
    raise ValueError(f"Unknown model: {model}")


import re as _re


def _extract_xml_block(raw: str, tag: str) -> str | None:
    """Extract a specific XML block from LLM output that may contain commentary."""
    pattern = _re.compile(rf"<{tag}>(.*?)</{tag}>", _re.DOTALL)
    m = pattern.search(raw)
    if m:
        return f"<{tag}>{m.group(1)}</{tag}>"
    return None


def _wrap_llm_xml(raw: str, inner_tag: str | None = None) -> str:
    """Clean up LLM XML output: extract relevant XML block, strip preamble."""
    text = raw.strip()
    # If we know the inner tag, extract just that block (handles mixed commentary)
    if inner_tag:
        block = _extract_xml_block(text, inner_tag)
        if block:
            return f"<response>{block}</response>"
    # Strip markdown fences if present
    text = _re.sub(r"```xml\s*", "", text)
    text = _re.sub(r"\s*```", "", text)
    # Strip anything before the first XML tag
    m = _re.search(r"<", text)
    if m:
        text = text[m.start():]
    # Remove outer <response> tags if the LLM already included them
    if text.startswith("<response>"):
        text = text[len("<response>"):]
    if text.endswith("</response>"):
        text = text[:-len("</response>")]
    return "<response>" + text + "</response>"


def _extract_placements_fallback(raw: str) -> ET.Element:
    """Extract complete <placement> blocks via regex when full XML parse fails."""
    blocks = _re.findall(r"<placement>.*?</placement>", raw, _re.DOTALL)
    xml = "<response><possible_ad_placements>" + "".join(blocks) + "</possible_ad_placements></response>"
    return ET.fromstring(xml)


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sponsors_from_config(config: dict) -> "list[Advertisement]":
    sponsors = config.get("sponsors", [])
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


def _determine_ad_placement(
    transcription_segments: list[TranscriptionSegment],
    available_ads: list[Advertisement],
    model: LLM_MODEL,
) -> List[AdvertisementPlacement]:

    transcription_segment_xml = "\n".join([
        f"<transcription_segment no='{segment.no}'>"
        f"<start>{segment.start}</start>"
        f"<end>{segment.end}</end>"
        f"<text>{xml_escape(segment.text)}</text>"
        f"</transcription_segment>"
    for segment in transcription_segments])

    available_ad_xml = "\n".join([
        f"<advertisement id='{xml_escape(ad.id)}'>"
        f"<url>{xml_escape(ad.url)}</url>"
        f"<title>{xml_escape(ad.title)}</title>"
        f"<content>{xml_escape(ad.content)}</content>"
        f"<tags>{xml_escape(','.join(ad.tags))}</tags>"
        f"</advertisement>"
    for ad in available_ads])

    prompt = f"""You are a natural advertisment placement expert. You will be working with the transcript of a audio recording which might be a podcast, youtube video, or any other recorded audio.
    You will be given the transcript of a audio recording and a list of advertisements that can be placed in the audio recording.
    Your job is to determine the transcription segments along with the advertisement that should be placed in the audio recording.

    Please respond in the format provided between the <example></example> tags. Do NOT include XML comments in your response.
    <example>
    <response>
    <possible_ad_placements>
    <placement>
    <transcription_segment no='segment number'/>
    <advertisement id='advertisement id'/>
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
    # Strip XML comments that some models add (break parsing if truncated)
    cleaned = _re.sub(r"<!--.*?-->", "", raw, flags=_re.DOTALL)
    xml_response = _wrap_llm_xml(cleaned, inner_tag="possible_ad_placements")
    try:
        root = ET.fromstring(xml_response)
    except ET.ParseError:
        # Fallback: extract complete <placement> blocks via regex
        root = _extract_placements_fallback(cleaned)

    segment_map = {seg.no: seg for seg in transcription_segments}
    ad_map = {ad.id: ad for ad in available_ads}
    ad_placements = []
    for placement in root.findall('.//placement'):
        seg_el = placement.find('transcription_segment')
        ad_el = placement.find('advertisement')
        if seg_el is None or ad_el is None:
            continue
        try:
            segment_no = int(seg_el.get('no'))
        except (TypeError, ValueError):
            continue
        ad_id = ad_el.get('id')
        if segment_no not in segment_map or ad_id not in ad_map:
            continue
        ad_placements.append(AdvertisementPlacement(
            transcription_segment=segment_map[segment_no],
            determined_advertisement=ad_map[ad_id],
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
        <text>{xml_escape(ad_placement.transcription_segment.text)}</text>
    </transcription_segment>
    <advertisement>
        <title>{xml_escape(ad_placement.determined_advertisement.title)}</title>
        <content>{xml_escape(ad_placement.determined_advertisement.content)}</content>
    </advertisement>
</advertisement_placement>
"""

    surrounding_segments_xml = "\n".join([
        f"<transcription_segment no='{segment.no}'>"
        f"<text>{xml_escape(segment.text)}</text>"
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
    xml_response = _wrap_llm_xml(raw, inner_tag="advertisements")
    try:
        return parse_advertisement_xml(xml_response)
    except ET.ParseError as e:
        raise RuntimeError(f"Ad text XML parse error: {e}\nRaw LLM response:\n{raw[:2000]}") from e


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

def generate_advertisement_audio(
    advertisement_text: str,
    voice_id: str,
    file_path: str | None = None,
) -> str:
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

