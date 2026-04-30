import express from "express";
import cors from "cors";
import { join } from "path";
import { existsSync, mkdirSync, rmSync } from "fs";
import multer from "multer";
import { v4 as uuidv4 } from "uuid";
import { processClaim } from "./generic-processor";

const app = express();
app.use(cors());
app.use(express.json());

const UPLOAD_ROOT = "/tmp/medical-claims";

const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => {
      const claimId = req.body.claimId || (req as any).generatedId || uuidv4();
      if (!req.body.claimId) (req as any).generatedId = claimId;
      
      const requestId = (req as any).requestId || uuidv4();
      (req as any).requestId = requestId;

      const dest = join(UPLOAD_ROOT, `${claimId}_${requestId}`);
      if (!existsSync(dest)) {
        mkdirSync(dest, { recursive: true });
      }
      cb(null, dest);
    },
    filename: (req, file, cb) => {
      cb(null, file.originalname);
    },
  }),
});

const validateRequest = (req: express.Request, res: express.Response) => {
  const claimId = req.body.claimId;
  if (!claimId) {
    res.status(400).json({ error: "Missing 'claimId' in request body" });
    return false;
  }
  return true;
};

app.post("/api/report", upload.array("files"), async (req, res) => {
  console.log(`\n[API] POST /api/report - Received request`);
  if (!validateRequest(req, res)) return;
  const { claimId } = req.body;
  const requestId = (req as any).requestId;
  const files = req.files as Express.Multer.File[];
  console.log(`[API] ClaimID: ${claimId} (Req: ${requestId}) - Files received: ${files?.length ?? 0}`);

  const claimDir = join(UPLOAD_ROOT, `${claimId}_${requestId}`);
  try {
    console.log(`[API] Starting processClaim (skipSummary=true)...`);
    const result = await processClaim(claimId, claimDir, true);
    console.log(`[API] Processing complete for ${claimId}`);
    res.json(result.records);
  } catch (error) {
    console.error(`[API] Error processing ${claimId}:`, error);
    res.status(500).json({ error: "Processing failed", details: (error as Error).message });
  } finally {
    try {
      if (requestId) {
        console.log(`[API] Cleaning up ${claimDir}...`);
        rmSync(claimDir, { recursive: true, force: true });
      }
    } catch (err) {
      console.error(`[API] Cleanup failed for ${claimDir}:`, err);
    }
  }
});

app.post("/api/summary", upload.array("files"), async (req, res) => {
  console.log(`\n[API] POST /api/summary - Received request`);
  if (!validateRequest(req, res)) return;
  const { claimId } = req.body;
  const requestId = (req as any).requestId;
  const files = req.files as Express.Multer.File[];
  console.log(`[API] ClaimID: ${claimId} (Req: ${requestId}) - Files received: ${files?.length ?? 0}`);

  const claimDir = join(UPLOAD_ROOT, `${claimId}_${requestId}`);
  try {
    console.log(`[API] Starting processClaim (skipSummary=false)...`);
    const result = await processClaim(claimId, claimDir, false);
    console.log(`[API] Summary generation complete for ${claimId}`);
    res.json(result.summary);
  } catch (error) {
    console.error(`[API] Error generating summary for ${claimId}:`, error);
    res.status(500).json({ error: "Processing failed", details: (error as Error).message });
  } finally {
    try {
      if (requestId) {
        console.log(`[API] Cleaning up ${claimDir}...`);
        rmSync(claimDir, { recursive: true, force: true });
      }
    } catch (err) {
      console.error(`[API] Cleanup failed for ${claimDir}:`, err);
    }
  }
});

app.get("/", (req, res) => {
  res.json({ status: "ok", message: "Medical Reports API is running" });
});

const PORT = process.env.PORT || 3000;
if (process.env.NODE_ENV !== "production") {
  app.listen(PORT, () => {
    console.log(`Server listening on port ${PORT}`);
  });
}

export default app;
