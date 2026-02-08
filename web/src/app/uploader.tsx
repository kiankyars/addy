"use client";

import { useLocalStorage } from "usehooks-ts";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { GeneratedAd } from "@/lib/types";
import { sentenceCase } from "change-case";

const VIDEO_ID_RE =
  /(?:youtube\.com\/(?:watch\?v=|embed\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})/;

function extractVideoId(input: string): string | null {
  const trimmed = input.trim();
  if (/^[a-zA-Z0-9_-]{11}$/.test(trimmed)) return trimmed;
  const m = trimmed.match(VIDEO_ID_RE);
  return m ? m[1] : null;
}

const Uploader = () => {
  const [urlOrId, setUrlOrId] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const videoId = extractVideoId(urlOrId);
    if (!videoId) {
      return;
    }
    setIsSubmitting(true);
    try {
      const res = await fetch("/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: videoId }),
      });
      if (!res.ok) throw new Error("Failed to start process");
      const data = await res.json();
      setJobId(data.id);
    } catch (err) {
      console.error(err);
      setIsSubmitting(false);
    }
  };

  return (
    <div className="w-full space-y-4">
      {!jobId ? (
        <form onSubmit={handleSubmit} className="flex flex-col gap-2">
          <input
            type="text"
            value={urlOrId}
            onChange={(e) => setUrlOrId(e.target.value)}
            placeholder="YouTube URL or video ID"
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
          />
          <button
            type="submit"
            disabled={isSubmitting || !extractVideoId(urlOrId)}
            className="rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground disabled:opacity-50"
          >
            {isSubmitting ? "Starting…" : "Process"}
          </button>
        </form>
      ) : (
        <Process jobId={jobId} onReset={() => setJobId(null)} />
      )}
    </div>
  );
};

const Process = ({
  jobId,
  onReset,
}: {
  jobId: string;
  onReset: () => void;
}) => {
  const [status, setStatus] = useState("processing");
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  const [, setValue] = useLocalStorage<GeneratedAd[] | undefined>(
    "generated_ads",
    undefined
  );

  useEffect(() => {
    const interval = setInterval(async () => {
      const res = await fetch(`/audio_files/${jobId}`);
      if (!res.ok) return;
      const data = await res.json();
      setStatus(data.processing_status);
      if (data.error) setError(data.error);
      if (data.processing_status === "complete") {
        clearInterval(interval);
        setValue(data.generated_ads);
        router.push(`/generated?audioId=${jobId}`);
      }
      if (data.processing_status === "failed") {
        clearInterval(interval);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [jobId, router, setValue]);

  return (
    <div className="space-y-2">
      <p>
        Status: {sentenceCase(status)}
        {error && <span className="text-destructive"> — {error}</span>}
      </p>
      {status === "failed" && (
        <button
          type="button"
          onClick={onReset}
          className="rounded-md border border-border px-3 py-1 text-sm"
        >
          Try again
        </button>
      )}
    </div>
  );
};

export default Uploader;
