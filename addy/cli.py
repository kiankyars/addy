"""
Addy: YouTube → transcript → placement → copy → TTS → N ad MP3s.
All settings from dwarkesh.json (video, output, model, voice, sponsors).
"""
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interface import load_config
from pipeline import start_job, process_job, get_job

console = Console()

VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|embed/)|youtu\.be/)([a-zA-Z0-9_-]{11})"
)


def extract_video_id(value: str) -> str | None:
    value = value.strip()
    if re.match(r"^[a-zA-Z0-9_-]{11}$", value):
        return value
    m = VIDEO_ID_RE.search(value)
    return m.group(1) if m else None


def main() -> None:
    config_path = Path.cwd() / "dwarkesh.json"
    if not config_path.exists():
        config_path = Path(__file__).resolve().parent.parent / "dwarkesh.json"
    if not config_path.exists():
        console.print("[red]Error:[/] dwarkesh.json not found in current directory or project root.")
        sys.exit(1)

    config = load_config(config_path)
    video = config.get("video")
    if not video:
        console.print("[red]Error:[/] dwarkesh.json must contain \"video\" (URL or ID).")
        sys.exit(1)

    video_id = extract_video_id(video)
    if not video_id:
        console.print("[red]Error:[/] Invalid video URL or ID in dwarkesh.json.")
        sys.exit(1)

    output_dir = Path(config.get("output", "addy_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        "[bold cyan]Addy[/] — transcript → placement → copy → TTS",
        border_style="cyan",
    ))
    console.print(f"  [dim]Video[/]   {video_id}")
    console.print(f"  [dim]Output[/]  {output_dir.resolve()}")
    console.print(f"  [dim]Model[/]   {config.get('model', 'claude')}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        t = progress.add_task("Running pipeline…", total=None)
        job_id = start_job(video_id)
        process_job(job_id, config)
        progress.update(t, completed=1)

    job = get_job(job_id)
    if not job:
        console.print("[red]Error:[/] Job not found.")
        sys.exit(1)
    if job.status == "failed":
        console.print(f"[red]Failed:[/] {job.error}")
        sys.exit(1)

    written = []
    for i, ad in enumerate(job.generated_ads):
        safe_title = re.sub(r"[^\w\-]", "_", ad.advertisement.title)[:40]
        out_path = output_dir / f"ad_{i+1}_{safe_title}.mp3"
        out_path.write_bytes(ad.audio_bytes)
        written.append((out_path, ad.advertisement.title))

    table = Table(title="Generated ads", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim")
    table.add_column("Sponsor")
    table.add_column("File", style="green")
    for i, (path, title) in enumerate(written, 1):
        table.add_row(str(i), title, str(path))
    console.print(table)
    console.print(f"\n[bold green]Done.[/] {len(written)} ad(s) written to [green]{output_dir}[/]")


if __name__ == "__main__":
    main()
