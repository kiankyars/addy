"use client";

import { sentenceCase } from "change-case";
import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";

export const Poll = () => {
  const id = useSearchParams().get("id");

  const [processingStatus, setProcessingStatus] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;

    const interval = setInterval(async () => {
      try {
        const response = await fetch(
          `/stitched-audio/${id}`
        );
        const data = await response.json();
        setProcessingStatus(data.processing_status);

        if (data.processing_status === "complete") {
          clearInterval(interval);
        }
      } catch (error) {
        console.error("Error fetching processing status:", error);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [id]);

  return (
    <div className="flex flex-col items-center justify-center w-full">
      {processingStatus !== "complete" ? (
        <div className="flex items-center justify-center h-full">
          <p className="text-2xl">
            {sentenceCase(processingStatus ?? "Processing")}
          </p>
        </div>
      ) : (
        <audio
          className="w-full"
          controls
          src={`/stitched-audio/${id}/bytes`}
        />
      )}
    </div>
  );
};
