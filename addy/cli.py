"""
Addy CLI: YouTube URL → transcript → placement → copy → TTS → N ad MP3s.
"""
import re
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import start_job, process_job, get_job

console = Console()
app = typer.Typer(
    name="addy",
    help="Generate podcast ad reads from a YouTube video. Outputs one MP3 per sponsor.",
    no_args_is_help=True,
)

VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|embed/)|youtu\.be/)([a-zA-Z0-9_-]{11})"
)


def extract_video_id(value: str) -> str | None:
    value = value.strip()
    if re.match(r"^[a-zA-Z0-9_-]{11}$", value):
        return value
    m = VIDEO_ID_RE.search(value)
    return m.group(1) if m else None


@app.callback(invoke_without_command=True)
def main(
    video: str = typer.Argument(..., help="YouTube video URL or 11-character video ID"),
    output: Path = typer.Option(
        Path("addy_output"),
        "--output", "-o",
        path_type=Path,
        help="Directory to write ad MP3s",
    ),
    sponsors: Path | None = typer.Option(
        None,
        "--sponsors",
        path_type=Path,
        help="Path to sponsors JSON (default: config/sponsors.json)",
    ),
    model: str = typer.Option(
        "claude",
        "--model", "-m",
        help="LLM for placement and ad copy: claude or gemini",
    ),
) -> None:
    if model not in ("claude", "gemini"):
        console.print("[red]Error:[/] --model must be 'claude' or 'gemini'.")
        raise typer.Exit(1)

    video_id = extract_video_id(video)
    if not video_id:
        console.print("[red]Error:[/] Invalid YouTube URL or video ID.")
        raise typer.Exit(1)

    output.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        "[bold cyan]Addy[/] — transcript → placement → copy → TTS",
        border_style="cyan",
    ))
    console.print(f"  [dim]Video ID[/] {video_id}")
    console.print(f"  [dim]Output[/]  {output.resolve()}")
    console.print(f"  [dim]LLM[/]     {model}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        t = progress.add_task("Running pipeline…", total=None)
        job_id = start_job(video_id, sponsors)
        process_job(job_id, sponsors, model=model)
        progress.update(t, completed=1)

    job = get_job(job_id)
    if not job:
        console.print("[red]Error:[/] Job not found.")
        raise typer.Exit(1)
    if job.status == "failed":
        console.print(f"[red]Failed:[/] {job.error}")
        raise typer.Exit(1)

    written = []
    for i, ad in enumerate(job.generated_ads):
        safe_title = re.sub(r"[^\w\-]", "_", ad.advertisement.title)[:40]
        out_path = output / f"ad_{i+1}_{safe_title}.mp3"
        out_path.write_bytes(ad.audio_bytes)
        written.append((out_path, ad.advertisement.title))

    table = Table(title="Generated ads", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim")
    table.add_column("Sponsor")
    table.add_column("File", style="green")
    for i, (path, title) in enumerate(written, 1):
        table.add_row(str(i), title, str(path))
    console.print(table)
    console.print(f"\n[bold green]Done.[/] {len(written)} ad(s) written to [green]{output}[/]")


if __name__ == "__main__":
    app()
