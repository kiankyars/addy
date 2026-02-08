"""
YouTube → transcript → ad placement → copy → TTS. Outputs N ad audio files (one per placement).
"""
import os
import re
from pathlib import Path
from uuid import uuid4
from dataclasses import dataclass, field

from interface import (
    get_youtube_transcript,
    sponsors_from_config,
    determine_ad_placement,
    generate_advertisements,
    select_best_advertisement,
    generate_advertisement_audio,
    TranscriptionSegment,
    Advertisement,
)


@dataclass
class GeneratedAd:
    id: str
    segue: str
    content: str
    exit: str
    audio_bytes: bytes
    segment_no: int
    segment_start: float
    segment_end: float
    segment_text: str
    advertisement: Advertisement


@dataclass
class Job:
    id: str
    video_id: str
    status: str  # "processing" | "complete" | "failed"
    transcript: list[TranscriptionSegment] = field(default_factory=list)
    generated_ads: list[GeneratedAd] = field(default_factory=list)
    error: str | None = None


_jobs: dict[str, Job] = {}


def start_job(video_id: str, sponsors_path: str | Path | None = None) -> str:
    job_id = str(uuid4())
    _jobs[job_id] = Job(id=job_id, video_id=video_id, status="processing")
    return job_id


def process_job(job_id: str, config: dict, on_status=None) -> None:
    """Run the full pipeline. on_status(step, detail, progress, total) is called at each stage."""
    def _status(step: str, detail: str = "", progress: int = 0, total: int = 0):
        if on_status:
            on_status(step, detail, progress, total)
    def _clean_tts_text(text: str) -> str:
        # Strip any XML/SSML-like tags and collapse whitespace for TTS safety.
        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    job = _jobs.get(job_id)
    if not job:
        return
    try:
        _status("transcript", "Fetching YouTube transcript")
        transcript = get_youtube_transcript(job.video_id)
        job.transcript = transcript
        if not transcript:
            job.status = "failed"
            job.error = "No transcript"
            return
        _status("transcript", f"Fetched {len(transcript)} segments")

        _status("sponsors", "Loading sponsors from config")
        ads = sponsors_from_config(config)
        if not ads:
            job.status = "failed"
            job.error = "No sponsors in config"
            return
        _status("sponsors", f"Loaded {len(ads)} sponsor(s)")

        model = config.get("model", "claude")
        voice_id = config["voice"]

        _status("placement", f"Determining ad placements")
        ad_placements = determine_ad_placement(transcript, ads, model=model)
        _status("placement", f"Found {len(ad_placements)} placement(s)")

        total_placements = len(ad_placements)
        for pi, ad_placement in enumerate(ad_placements, 1):
            sponsor = ad_placement.determined_advertisement.title
            _status("copy", f"Writing copy for {sponsor}", pi, total_placements)
            ad_texts = generate_advertisements(ad_placement, transcript, model=model)
            best_idx = select_best_advertisement(ad_placement, transcript, ad_texts, model=model)
            _status(
                "copy",
                f"Generated {len(ad_texts)} variation(s) for {sponsor}, selected #{best_idx + 1}",
                pi,
                total_placements,
            )

            # Try judge pick first, then fall back to other variants if TTS text is empty/punct-only.
            candidate_order = [best_idx] + [i for i in range(len(ad_texts)) if i != best_idx]
            gen_text = None
            base = ""
            for idx in candidate_order:
                candidate = ad_texts[idx]
                raw = candidate.segue + " " + candidate.content + " " + candidate.exit
                cleaned = _clean_tts_text(raw)
                if re.search(r"[A-Za-z0-9]", cleaned):
                    gen_text = candidate
                    base = cleaned
                    break
            if gen_text is None:
                raise RuntimeError("All ad variations were empty or punctuation-only after cleaning.")
            _status("tts", f"Cartesia TTS for {sponsor}", pi, total_placements)
            apath = generate_advertisement_audio(
                base,
                voice_id=voice_id,
                emotion=getattr(gen_text, "emotion", None),
                speed=getattr(gen_text, "speed", None),
            )
            with open(apath, "rb") as f:
                ad_bytes = f.read()
            os.remove(apath)
            seg = ad_placement.transcription_segment
            job.generated_ads.append(
                GeneratedAd(
                    id=str(uuid4()),
                    segue=gen_text.segue,
                    content=gen_text.content,
                    exit=gen_text.exit,
                    audio_bytes=ad_bytes,
                    segment_no=seg.no,
                    segment_start=seg.start,
                    segment_end=seg.end,
                    segment_text=seg.text,
                    advertisement=ad_placement.determined_advertisement,
                )
            )

        _status("done", f"Pipeline complete — {len(job.generated_ads)} ad(s) generated")
        job.status = "complete"
    except Exception as e:
        job.status = "failed"
        job.error = str(e)


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)
