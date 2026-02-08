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

EXA_API_KEY = os.getenv("EXA_API_KEY")

LLM_MODEL = Literal["claude", "gemini"]
_gemini_client = None
_exa_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _gemini_client


def _get_exa_client():
    global _exa_client
    if _exa_client is None:
        from exa_py import Exa
        _exa_client = Exa(api_key=EXA_API_KEY)
    return _exa_client


def _llm_completion(prompt: str, stop_sequences: list[str], model: LLM_MODEL) -> str:
    if model == "claude":
        response = anthropic_client.messages.create(
            model="claude-haiku-4.5",
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
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )
        return (response.text or "").rstrip()


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
    emotion: str | None = None
    speed: float | None = None


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


_TOOL_SEARCH_SPONSOR = {
    "name": "search_sponsor_info",
    "description": "Search the web for information about a sponsor/product to write better ad placements. Use this to learn about what the sponsor does, their key selling points, and recent news.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query about the sponsor or product",
            }
        },
        "required": ["query"],
    },
}

_TOOL_SELECT_PLACEMENTS = {
    "name": "select_ad_placements",
    "description": "Commit your final ad placement decisions. Each placement pairs a transcript segment number with a sponsor ad ID. Call this once with all placements.",
    "input_schema": {
        "type": "object",
        "properties": {
            "placements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "segment_no": {"type": "integer", "description": "Transcript segment number to place the ad after"},
                        "ad_id": {"type": "string", "description": "Advertisement ID"},
                        "reasoning": {"type": "string", "description": "Brief reason this segment is a natural fit"},
                    },
                    "required": ["segment_no", "ad_id", "reasoning"],
                },
            }
        },
        "required": ["placements"],
    },
}


def _execute_exa_search(query: str) -> str:
    """Run an Exa search and return a text summary of results."""
    exa = _get_exa_client()
    result = exa.search_and_contents(query, num_results=3, text=True)
    parts = []
    for r in result.results:
        text_snippet = (r.text or "")
        parts.append(f"- {r.title}: {text_snippet}")
    return "\n".join(parts) if parts else "No results found."


def _determine_ad_placement_agentic(
    transcription_segments: list[TranscriptionSegment],
    available_ads: list[Advertisement],
) -> List[AdvertisementPlacement]:
    """Claude tool-use placement: optionally researches sponsors via Exa, then commits placements."""

    segment_lines = "\n".join(
        f"[{seg.no}] ({seg.start:.1f}s-{seg.end:.1f}s) {seg.text}"
        for seg in transcription_segments
    )
    ad_lines = "\n".join(
        f"- id={ad.id} | {ad.title}: {ad.content} (tags: {', '.join(ad.tags)})"
        for ad in available_ads
    )

    system_prompt = (
        "You are an expert podcast ad-placement agent. You will be given a transcript and a list of sponsors. "
        "Your goal: pick the best transcript segments to insert each sponsor's ad after, so ads feel like a natural part of the conversation.\n\n"
        "You have two tools:\n"
        "1. search_sponsor_info — research a sponsor online to understand their product better (optional, use if sponsor info is sparse)\n"
        "2. select_ad_placements — commit your final placement decisions (required)\n\n"
        "Rules:\n"
        "- Place each sponsor at most once.\n"
        "- Only place ads where there is a natural topic transition.\n"
        "- If no good placement exists for a sponsor, omit it.\n"
        "- Call select_ad_placements exactly once with all your decisions."
    )

    messages = [
        {
            "role": "user",
            "content": f"Here is the transcript:\n\n{segment_lines}\n\nHere are the sponsors:\n\n{ad_lines}\n\nResearch any sponsors if needed, then select placements.",
        }
    ]

    tools = [_TOOL_SEARCH_SPONSOR, _TOOL_SELECT_PLACEMENTS]

    # First call
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
        tools=tools,
    )

    # Process tool calls — may need a second round if searches were requested
    tool_results = []
    placements_input = None

    for block in response.content:
        if block.type == "tool_use":
            if block.name == "search_sponsor_info":
                result_text = _execute_exa_search(block.input["query"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })
            elif block.name == "select_ad_placements":
                placements_input = block.input["placements"]
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Placements recorded.",
                })

    # If we got searches but no placements yet, do a second call
    if tool_results and placements_input is None:
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        response2 = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            tools=tools,
        )

        for block in response2.content:
            if block.type == "tool_use" and block.name == "select_ad_placements":
                placements_input = block.input["placements"]
                break

    if not placements_input:
        return []

    # Convert to AdvertisementPlacement objects
    segment_map = {seg.no: seg for seg in transcription_segments}
    ad_map = {ad.id: ad for ad in available_ads}
    results = []
    for p in placements_input:
        seg_no = p["segment_no"]
        ad_id = p["ad_id"]
        if seg_no in segment_map and ad_id in ad_map:
            results.append(AdvertisementPlacement(
                transcription_segment=segment_map[seg_no],
                determined_advertisement=ad_map[ad_id],
            ))
    return results


def determine_ad_placement(
    transcription_segments: list[TranscriptionSegment],
    available_ads: list[Advertisement],
    model: LLM_MODEL = "claude",
) -> List[AdvertisementPlacement]:
    if model == "claude":
        placements = _determine_ad_placement_agentic(transcription_segments, available_ads)
    else:
        placements = _determine_ad_placement(transcription_segments, available_ads, model)

    # Enforce "each sponsor at most once" even if the model repeats sponsors.
    seen_ad_ids: set[str] = set()
    deduped: list[AdvertisementPlacement] = []
    for placement in placements:
        ad_id = placement.determined_advertisement.id
        if ad_id in seen_ad_ids:
            continue
        seen_ad_ids.add(ad_id)
        deduped.append(placement)

    # Ensure every sponsor gets exactly one placement by filling missing ads.
    missing_ads = [ad for ad in available_ads if ad.id not in seen_ad_ids]
    if missing_ads:
        used_segments = {p.transcription_segment.no for p in deduped}
        for i, ad in enumerate(missing_ads):
            seg = _choose_segment_for_ad(
                transcription_segments,
                ad,
                used_segments,
                fallback_rank=i,
                fallback_total=len(missing_ads),
            )
            used_segments.add(seg.no)
            deduped.append(
                AdvertisementPlacement(
                    transcription_segment=seg,
                    determined_advertisement=ad,
                )
            )
    return deduped


def _choose_segment_for_ad(
    transcription_segments: list[TranscriptionSegment],
    ad: Advertisement,
    used_segment_nos: set[int],
    fallback_rank: int,
    fallback_total: int,
) -> TranscriptionSegment:
    keywords = _ad_keywords(ad)
    best_seg = None
    best_score = -1.0
    for seg in transcription_segments:
        if seg.no in used_segment_nos:
            continue
        score = _score_segment(seg, keywords)
        if score > best_score:
            best_score = score
            best_seg = seg
    if best_seg and best_score > 0:
        return best_seg
    # Fallback: pick a roughly even spacing if no keyword hits.
    target_idx = int((fallback_rank + 1) / (fallback_total + 1) * len(transcription_segments))
    return _nearest_unused_segment(transcription_segments, target_idx, used_segment_nos)


def _nearest_unused_segment(
    transcription_segments: list[TranscriptionSegment],
    target_idx: int,
    used_segment_nos: set[int],
) -> TranscriptionSegment:
    if 0 <= target_idx < len(transcription_segments):
        seg = transcription_segments[target_idx]
        if seg.no not in used_segment_nos:
            return seg
    for offset in range(1, len(transcription_segments)):
        left = target_idx - offset
        right = target_idx + offset
        if left >= 0:
            seg = transcription_segments[left]
            if seg.no not in used_segment_nos:
                return seg
        if right < len(transcription_segments):
            seg = transcription_segments[right]
            if seg.no not in used_segment_nos:
                return seg
    return transcription_segments[0]


def _ad_keywords(ad: Advertisement) -> list[str]:
    words = []
    for tag in ad.tags:
        words.extend(_re.split(r"[^a-zA-Z0-9]+", tag.lower()))
    words.extend(_re.split(r"[^a-zA-Z0-9]+", ad.title.lower()))
    return [w for w in words if len(w) >= 3]


def _score_segment(seg: TranscriptionSegment, keywords: list[str]) -> float:
    text = seg.text.lower()
    score = 0.0
    for kw in keywords:
        if kw and kw in text:
            score += 1.0
    if text.strip().endswith((".", "?", "!", "…")):
        score += 0.2
    return score

def parse_advertisement_xml(xml_content: str) -> List[GeneratedAdvertisementText]:
    root = ET.fromstring(xml_content)
    advertisements = []

    for ad in root.findall('.//advertisement'):
        segue = ad.find('segue').text
        content = ad.find('content').text
        exit = ad.find('exit').text
        emotion_el = ad.find('emotion')
        speed_el = ad.find('speed')
        emotion = emotion_el.text.strip() if emotion_el is not None and emotion_el.text else None
        speed = None
        if speed_el is not None and speed_el.text:
            try:
                speed = float(speed_el.text.strip())
            except ValueError:
                pass
        advertisements.append(GeneratedAdvertisementText(
            segue=segue, content=content, exit=exit,
            emotion=emotion, speed=speed,
        ))

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
6. For each advertisement, include an <emotion> tag describing the vocal emotion for TTS (e.g. "curiosity", "excitement", "warmth", "confidence"). Pick the emotion that best fits the sponsor and conversation tone.
7. For each advertisement, include a <speed> tag with a speaking speed multiplier (float between 0.7 and 1.3). Use slightly faster for energetic ads, slightly slower for premium/serious sponsors. Default is 1.0.

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
<emotion>excitement</emotion>
<speed>1.1</speed>
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
    surrounding_segments = _collect_surrounding_segments(ad_placement, transcription_segments)
    return _generate_advertisement_text(ad_placement, surrounding_segments, model)


def _collect_surrounding_segments(
    ad_placement: AdvertisementPlacement,
    transcription_segments: list[TranscriptionSegment],
) -> list[TranscriptionSegment]:
    surrounding_segments = []
    segment_nos = {segment.no: segment for segment in transcription_segments}
    for offset in [-2, -1, 1, 2]:
        target_no = ad_placement.transcription_segment.no + offset
        if target_no in segment_nos:
            surrounding_segments.append(segment_nos[target_no])
    return surrounding_segments


def select_best_advertisement(
    ad_placement: AdvertisementPlacement,
    transcription_segments: list[TranscriptionSegment],
    candidates: List[GeneratedAdvertisementText],
    model: LLM_MODEL = "claude",
) -> int:
    """Return the 0-based index of the best ad candidate. Falls back to 0 on parse failure."""
    surrounding_segments = _collect_surrounding_segments(ad_placement, transcription_segments)
    segment_context = "\n".join(
        f"- {seg.text}" for seg in [ad_placement.transcription_segment, *surrounding_segments]
    )
    ads_text = "\n\n".join(
        [
            f"Ad {i+1}:\nSegue: {c.segue}\nContent: {c.content}\nExit: {c.exit}"
            for i, c in enumerate(candidates)
        ]
    )
    prompt = (
        "You are selecting the single best ad read for a podcast. "
        "Pick the ad that best fits the conversation flow, sounds natural, and is concise.\n\n"
        f"Conversation context:\n{segment_context}\n\n"
        f"Candidates:\n{ads_text}\n\n"
        "Respond with only the number 1, 2, or 3."
    )
    raw = _llm_completion(prompt, ["\n"], model)
    match = _re.search(r"\b([1-3])\b", raw.strip())
    if not match:
        return 0
    return int(match.group(1)) - 1

def generate_advertisement_audio(
    advertisement_text: str,
    voice_id: str,
    file_path: str | None = None,
    emotion: str | None = None,
    speed: float | None = None,
) -> str:
    tts_kwargs = dict(
        model_id="sonic-3",
        transcript=advertisement_text,
        voice={"id": voice_id},
        language="en",
        output_format={"container": "mp3", "sample_rate": 44100, "bit_rate": 128000},
    )
    if emotion or speed:
        gen_config = {}
        if emotion:
            gen_config["emotion"] = emotion
        if speed:
            gen_config["speed"] = speed
        tts_kwargs["generation_config"] = gen_config
    chunks = list(cartesia_client.tts.bytes(**tts_kwargs))
    audio_bytes = b"".join(chunks)
    save_file_path = file_path or f"{uuid.uuid4()}.mp3"
    with open(save_file_path, "wb") as f:
        f.write(audio_bytes)
    return save_file_path
