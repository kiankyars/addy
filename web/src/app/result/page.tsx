import Image from "next/image";
import { Suspense } from "react";
import { Poll } from "./poll";

export default function Result() {
  return (
    <div className="flex flex-col py-12 h-screen">
      <h1 className="flex items-center gap-x-2 text-3xl font-black mb-8 -ml-2">
        <Image src="/headphone.png" width={64} height={64} alt="Addy logo" />
        Addy Podcasts
      </h1>
      <div className="flex flex-col w-full items-center justify-center">
        <Suspense fallback={<p>Loading…</p>}>
          <Poll />
        </Suspense>
      </div>
    </div>
  );
}
