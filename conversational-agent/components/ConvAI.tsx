"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const AGENT_PHONE = process.env.NEXT_PUBLIC_AGENT_PHONE || "";

export function ConvAI() {
  return (
    <div className="flex justify-center items-center gap-x-4">
      <Card className="rounded-3xl">
        <CardContent>
          <CardHeader>
            <CardTitle className="text-center">
              Dwarkesh Podcast Voice Agent
            </CardTitle>
          </CardHeader>
          <p className="text-center text-muted-foreground text-sm mb-4">
            Talk to a thoughtful long-form podcast host (Anthropic + Cartesia).
            Use a cloned voice at{" "}
            <a
              href="https://play.cartesia.ai/voices/create/clone"
              target="_blank"
              rel="noopener noreferrer"
              className="underline"
            >
              play.cartesia.ai
            </a>
            .
          </p>
          <div className="flex flex-col gap-y-4 text-center">
            <div className={cn("orb my-16 mx-12 orb-inactive")}></div>
            {AGENT_PHONE ? (
              <Button
                variant="outline"
                className="rounded-full"
                size="lg"
                asChild
              >
                <a href={`tel:${AGENT_PHONE}`}>Call the agent</a>
              </Button>
            ) : (
              <p className="text-sm text-muted-foreground">
                Deploy the voice-agent and set NEXT_PUBLIC_AGENT_PHONE, or run
                locally: <code className="bg-muted px-1 rounded">cartesia chat 8000</code>
              </p>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
