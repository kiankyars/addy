"""
YouTube-based pipeline: transcript from youtube_transcript_api, sponsors from config JSON.
No SQL — in-memory job store.
"""
import os
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4
from dataclasses import dataclass, field

from interface import (
    get_youtube_transcript,
    load_sponsors,
    determine_ad_placement,
    generate_advertisements,
    generate_advertisement_audio,
    insert_advertisement_audio,
    generate_ad_audio_with_nearby_audio,
    TranscriptionSegment,
    Advertisement,
)
from pydub import AudioSegment
import io

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
    audio_path: str | None = None
    audio_bytes: bytes | None = None
    generated_ads: list[GeneratedAd] = field(default_factory=list)
    error: str | None = None


_jobs: dict[str, Job] = {}
_stitched: dict[str, bytes] = {}


def _download_youtube_audio(video_id: str) -> tuple[str, bytes]:
    """Download audio from YouTube with yt-dlp; return (temp_path, bytes)."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    subprocess.run(
        [
            "yt-dlp",
            "-f", "bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--extract-audio",
            "--audio-format", "mp3",
            "-o", path,
            "--no-playlist",
            url,
        ],
        check=True,
        capture_output=True,
    )
    with open(path, "rb") as f:
        data = f.read()
    return path, data


def start_job(video_id: str, sponsors_path: str | Path | None = None) -> str:
    """Create a processing job and return job_id. Call process_job(job_id) to run (e.g. in background)."""
    job_id = str(uuid4())
    _jobs[job_id] = Job(id=job_id, video_id=video_id, status="processing")
    return job_id


def process_job(job_id: str, sponsors_path: str | Path | None = None) -> None:
    """Fetch transcript, download audio, run placement + copy + TTS. Updates job in place."""
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

        audio_path, audio_bytes = _download_youtube_audio(job.video_id)
        job.audio_path = audio_path
        job.audio_bytes = audio_bytes

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


def get_generated_ad(ad_id: str) -> GeneratedAd | None:
    for job in _jobs.values():
        for ad in job.generated_ads:
            if ad.id == ad_id:
                return ad
    return None


def get_job_for_ad(ad_id: str) -> Job | None:
    for job in _jobs.values():
        for ad in job.generated_ads:
            if ad.id == ad_id:
                return job
    return None


def produce_preview_bytes(ad: GeneratedAd, job: Job) -> bytes:
    """Ad audio with 5s before + 10s after context."""
    if not job.audio_bytes:
        return ad.audio_bytes
    seg = next(s for s in job.transcript if s.no == ad.segment_no)
    return generate_ad_audio_with_nearby_audio(ad.audio_bytes, seg, job.audio_bytes)


def stitch(job_id: str, generated_ad_id: str) -> str:
    """Stitch chosen ad into full audio. Returns stitched_audio_id (for GET /stitched-audio/{id}/bytes)."""
    job = get_job(job_id)
    if not job or not job.audio_bytes:
        raise ValueError("Job or audio not found")
    ad = get_generated_ad(generated_ad_id)
    if not ad or ad not in job.generated_ads:
        raise ValueError("Generated ad not found")
    seg = next(s for s in job.transcript if s.no == ad.segment_no)
    path = insert_advertisement_audio(job.audio_bytes, ad.audio_bytes, seg)
    with open(path, "rb") as f:
        stitched_bytes = f.read()
    os.remove(path)
    stitched_id = str(uuid4())
    _stitched[stitched_id] = stitched_bytes
    return stitched_id


def get_stitched_bytes(stitched_id: str) -> bytes | None:
    return _stitched.get(stitched_id)
