"""
YouTube → transcript → ad placement → copy → TTS. Outputs N ad audio files (one per placement).
"""
import os
from pathlib import Path
from uuid import uuid4
from dataclasses import dataclass, field

from interface import (
    get_youtube_transcript,
    load_sponsors,
    determine_ad_placement,
    generate_advertisements,
    generate_advertisement_audio,
    TranscriptionSegment,
    Advertisement,
)

DEFAULT_SPONSORS_PATH = Path(__file__).resolve().parent / "config" / "sponsors.json"


@dataclass
class GeneratedAd:
    id: str
    segue: str
    content: str
    exit: str
    audio_bytes: bytes
    segment_no: int
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


def process_job(job_id: str, sponsors_path: str | Path | None = None) -> None:
    job = _jobs.get(job_id)
    if not job:
        return
    try:
        transcript = get_youtube_transcript(job.video_id)
        job.transcript = transcript
        if not transcript:
            job.status = "failed"
            job.error = "No transcript"
            return

        path = sponsors_path or DEFAULT_SPONSORS_PATH
        ads = load_sponsors(path)
        if not ads:
            job.status = "failed"
            job.error = "No sponsors in config"
            return

        ad_placements = determine_ad_placement(transcript, ads)
        for ad_placement in ad_placements:
            for gen_text in generate_advertisements(ad_placement, transcript):
                base = gen_text.segue + " " + gen_text.content + " " + gen_text.exit
                apath = generate_advertisement_audio(base)
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
                        advertisement=ad_placement.determined_advertisement,
                    )
                )
        job.status = "complete"
    except Exception as e:
        job.status = "failed"
        job.error = str(e)


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)
