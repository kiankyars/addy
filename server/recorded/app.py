from uuid import uuid4
from fastapi import FastAPI, Response, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from pipeline import (
    start_job,
    process_job,
    get_job,
    get_generated_ad,
    get_job_for_ad,
    produce_preview_bytes,
    stitch as pipeline_stitch,
    get_stitched_bytes,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    video_id: str
    sponsors_config: str | None = None


@app.post("/process", status_code=201)
async def process_youtube(background_tasks: BackgroundTasks, body: ProcessRequest):
    """Submit a YouTube video ID. Transcript from youtube_transcript_api, sponsors from config JSON."""
    job_id = start_job(body.video_id, body.sponsors_config)
    background_tasks.add_task(process_job, job_id, body.sponsors_config)
    return {"id": job_id}


@app.get("/audio_files/{file_id}")
async def get_audio_file(file_id: str):
    job = get_job(file_id)
    if not job:
        return Response(status_code=404, content="Job not found")
    return {
        "id": job.id,
        "file_name": job.video_id,
        "processing_status": job.status,
        "error": job.error,
        "generated_ads": [
            {
                "id": ad.id,
                "segue": ad.segue,
                "content": ad.content,
                "exit": ad.exit,
                "transcription_segment_id": str(ad.segment_no),
                "advertisement": {
                    "id": ad.advertisement.id,
                    "url": ad.advertisement.url,
                    "title": ad.advertisement.title,
                    "content": ad.advertisement.content,
                    "tags": ad.advertisement.tags,
                },
            }
            for ad in job.generated_ads
        ],
    }


@app.get("/audio_files")
async def get_all_audio_files():
    from pipeline import _jobs
    return [
        {
            "id": j.id,
            "file_name": j.video_id,
            "processing_status": j.status,
            "generated_ads": [
                {
                    "id": ad.id,
                    "segue": ad.segue,
                    "content": ad.content,
                    "exit": ad.exit,
                    "transcription_segment_id": str(ad.segment_no),
                    "advertisement": {"id": ad.advertisement.id, "url": ad.advertisement.url, "title": ad.advertisement.title, "content": ad.advertisement.content, "tags": ad.advertisement.tags},
                }
                for ad in j.generated_ads
            ],
        }
        for j in _jobs.values()
    ]


@app.get("/generated-ad/{ad_id}")
def get_generated_ad_audio(ad_id: str, range: str = Header(None)):
    ad = get_generated_ad(ad_id)
    if not ad:
        return Response(status_code=404, content="Generated ad not found")
    job = get_job_for_ad(ad_id)
    content = produce_preview_bytes(ad, job) if job else ad.audio_bytes
    content_length = len(content)
    if range:
        start, end = range.replace("bytes=", "").split("-")
        start = int(start)
        end = int(end) if end else content_length - 1
        content = content[start : end + 1]
        return Response(
            content=content,
            media_type="audio/mpeg",
            headers={
                "Content-Range": f"bytes {start}-{end}/{content_length}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(content)),
            },
            status_code=206,
        )
    return Response(
        content=content,
        media_type="audio/mpeg",
        headers={"Accept-Ranges": "bytes", "Content-Length": str(content_length)},
    )


class StitchRequest(BaseModel):
    audio_file_id: str  # job_id
    generated_ad_id: str


@app.post("/insert-advertisement-audio", status_code=201)
async def insert_advertisement_audio_endpoint(body: StitchRequest):
    stitched_id = pipeline_stitch(body.audio_file_id, body.generated_ad_id)
    return {"id": stitched_id}


@app.get("/stitched-audio/{stitched_audio_id}")
async def get_stitched_audio_meta(stitched_audio_id: str):
    if get_stitched_bytes(stitched_audio_id) is None:
        return Response(status_code=404, content="Stitched audio not found")
    return {"id": stitched_audio_id, "processing_status": "complete"}


@app.get("/stitched-audio/{stitched_audio_id}/bytes")
def get_stitched_audio_bytes(stitched_audio_id: str, range: str = Header(None)):
    content = get_stitched_bytes(stitched_audio_id)
    if content is None:
        return Response(status_code=404, content="Stitched audio not found")
    content_length = len(content)
    if range:
        start, end = range.replace("bytes=", "").split("-")
        start = int(start)
        end = int(end) if end else content_length - 1
        content = content[start : end + 1]
        return Response(
            content=content,
            media_type="audio/mpeg",
            headers={
                "Content-Range": f"bytes {start}-{end}/{content_length}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(content)),
            },
            status_code=206,
        )
    return Response(
        content=content,
        media_type="audio/mpeg",
        headers={"Accept-Ranges": "bytes", "Content-Length": str(content_length)},
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=4001, reload=True)
