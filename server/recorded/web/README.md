# Podcast UI

Next.js app for the Adible podcast ad pipeline. Upload audio → see generated ads → pick one → download stitched file.

Start the backend first (see root [README](../README.md)):

```bash
# server/recorded on port 4001
cd ../server/recorded && uvicorn app:app --port 4001
```

Then:

```bash
pnpm install && pnpm dev
```
