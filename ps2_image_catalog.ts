#!/usr/bin/env bun
/**
 * PS2 Image Catalog — per-image OCR + summary for each claim folder.
 *
 * For each medical-document file in a claim folder:
 *   - JPG / PNG       → one record
 *   - PDF             → one record per page  (page is shown as "<file.pdf> page N/M")
 *
 * For every record the model produces:
 *   - image_type         (X-ray / USG / CT / Angiogram / Typed report / Stamp / …)
 *   - body_region        (Abdomen / Chest / Coronary tree / Right kidney …)
 *   - modality_view      (PA / Lateral / RAO-Caudal / Transverse / N/A)
 *   - ocr.verbatim       (every legible line, in original script)
 *   - ocr.language       (Hindi / Tamil / English / mixed / …)
 *   - ocr.english        (translated to English when source is non-English)
 *   - summary            (1-3 sentence description of what's in the image)
 *   - key_observations   (bullet-list of clinical observations)
 *   - dates_found        (every date string found, normalised DD/MM/YYYY)
 *   - image_quality      (good | fair | poor + reason)
 *
 * Local Ollama, qwen3-vl:8b-instruct (≤15B). One call per image so each gets
 * a focused read — small VL models lose detail when several images compete.
 *
 * Usage:
 *   bun ps2_image_catalog.ts <PACKAGE_FOLDER> <CLAIM_ID>   # one claim
 *   bun ps2_image_catalog.ts <PACKAGE_FOLDER>              # every claim in package
 *   bun ps2_image_catalog.ts <PACKAGE_FOLDER> <CLAIM_ID> --limit 3   # cap pages (debug)
 *
 *   Output → Claims/<PACKAGE_FOLDER>/<CLAIM_ID>/image_catalog.{json,md}
 *   (written next to the source PDFs/images inside the claim folder itself)
 */

import { readdirSync, statSync, readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { join, extname, resolve } from "node:path";
import { pdf }    from "pdf-to-img";
import sharp      from "sharp";
import { jsonrepair } from "jsonrepair";
import { summarizeClaim } from "./ps3_claim_summary";

// ── Project root + .env loader (must come BEFORE reading any Bun.env vars) ──
// Walk up from cwd looking for the Claims/ folder so the script works whether
// invoked from the project root (`bun code/ps2_image_catalog.ts ...`) or from
// inside code/ (`bun ps2_image_catalog.ts ...`).
function findProjectRoot(): string {
  let dir = process.cwd();
  for (let i = 0; i < 4; i++) {
    if (existsSync(join(dir, "Claims"))) return dir;
    const parent = resolve(dir, "..");
    if (parent === dir) break;
    dir = parent;
  }
  return process.cwd();
}
const PROJECT_ROOT  = findProjectRoot();
const CLAIMS_ROOT   = join(PROJECT_ROOT, "Claims");

// Bun's auto-.env loader is cwd-bound; load .env from the detected project
// root so secrets work from any cwd. Doesn't override anything already set.
function loadDotenvFromRoot(rootDir: string) {
  const p = join(rootDir, ".env");
  if (!existsSync(p)) return;
  for (const line of readFileSync(p, "utf-8").split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$/i);
    if (!m) continue;
    let val = m[2]!;
    if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
      val = val.slice(1, -1);
    }
    if (!process.env[m[1]!]) process.env[m[1]!] = val;
  }
}
loadDotenvFromRoot(PROJECT_ROOT);

// ── Config ──────────────────────────────────────────────────────────────
// PROVIDER picks the backend. Default = Fireworks AI (OpenAI-compat); needs
// FIREWORKS_API_KEY in .env. Set PROVIDER=ollama to fall back to local Ollama.
const PROVIDER: "ollama" | "fireworks" =
  (Bun.env.PROVIDER as any) ?? "fireworks";

const OLLAMA_URL = Bun.env.OLLAMA_URL ?? "http://localhost:11434/api/chat";

// Fireworks AI (OpenAI-compatible). Override the model via MODEL env var.
// The default points at the user's own Fireworks DEPLOYMENT; override with
//   MODEL=accounts/fireworks/models/<published-model>
// to use a serverless model instead.
const FIREWORKS_URL     = Bun.env.FIREWORKS_URL ?? "https://api.fireworks.ai/inference/v1/chat/completions";
const FIREWORKS_API_KEY = Bun.env.FIREWORKS_API_KEY;

const MODEL = Bun.env.MODEL ?? (
  PROVIDER === "fireworks"
    ? "accounts/ajeya-rao-k-eckusf6m/deployments/euufjyfd"  // qwen3-vl-8b-instruct on Fireworks (≤15B)
    : "qwen3-vl:8b-instruct"                                  // local Ollama default, ≤15B
);

if (PROVIDER === "fireworks" && !FIREWORKS_API_KEY) {
  throw new Error("PROVIDER=fireworks but FIREWORKS_API_KEY is not set in .env");
}

const MAX_EDGE     = 1280;
const JPEG_Q       = 90;
const PDF_SCALE    = 220 / 72;     // ~220 DPI — readable handwriting / Indic glyphs
const NUM_PREDICT  = 4096;         // one image; verdict fits comfortably
const TEMPERATURE  = 0.05;
const TOP_K        = 30;
const REPEAT_PEN   = 1.0;
const SUPPORTED_IMG = new Set([".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"]);

// ── Per-image system prompt ─────────────────────────────────────────────
const SYSTEM = `You are a medical document OCR + description assistant for India's
AB PM-JAY claims dataset. You receive ONE image at a time. The image may be:
  - A medical scan (X-ray, IVP, KUB, USG, CT, MRI, coronary angiogram still)
  - A typed clinical / radiology report (English or Indic-language)
  - A handwritten doctor's note or prescription
  - A doctor's stamp, signature block, hospital letterhead
  - A photograph of a cath-lab monitor

ROLE — STRICTLY DESCRIPTIVE
- Do not diagnose. Do not approve / reject. Do not recommend treatment.
- Describe what you SEE. Cite uncertainty when a feature is unclear.

OCR + LANGUAGE
- Transcribe every legible line VERBATIM in its original script for the
  "ocr.verbatim" field.
- Detect the language for "ocr.language": one of English, Hindi, Tamil, Telugu,
  Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Urdu,
  Assamese, mixed, or none (if no readable text).
- Translate the verbatim text into English for "ocr.english" (omit fillers,
  preserve clinical terms, drug names, doses, dates).
- When a date is in DD-MM-YYYY / DD.MM.YYYY / YYYY-MM-DD / Indic numerals,
  normalise to DD/MM/YYYY in "dates_found".

IMAGE_TYPE (pick ONE)
  "Chest X-Ray"   "Abdominal X-Ray (KUB)"   "IVP"   "Urethrogram"
  "Ultrasound"    "CT"   "MRI"   "Coronary Angiogram"
  "Typed report"  "Handwritten note"   "Stamp/Signature"
  "Hospital letterhead"   "Patient photo"   "Other"

For "Coronary Angiogram" specifically: cath-lab fluoroscopy frames have a
dark background, grey vessel structures, and on-screen UI overlays such as
"AXIOM-Artis", "RAO/LAO/CRAN/CAUD N°", "EE %", "DDO %", "WC", "WW", "CARD f/s",
or "73/F"-style patient header. ALL such monitor screenshots are
"Coronary Angiogram" — even when guidewires, balloons, or stents are present.

BODY_REGION (free text, anatomically precise)
  e.g. "Chest (PA)", "Abdomen — pelvicalyceal system", "Left kidney + ureter",
       "Coronary tree — left system", "Hepatobiliary system",
       "N/A (text document)"

MODALITY_VIEW (free text)
  e.g. "PA", "Lateral", "Oblique", "RAO 30° / Caudal 20°", "Transverse",
       "Sagittal", "Long-axis", "N/A"

OUTPUT — exactly ONE JSON object, no markdown, no code fence, no prose:

{
  "image_type":     "<from list above>",
  "body_region":    "<string or 'N/A'>",
  "modality_view":  "<string or 'N/A'>",
  "stage_of_care":  "pre-procedure" | "intra-procedure" | "post-procedure" | "uncertain" | "n/a",
  "stage_evidence": "<short clue: e.g. 'DJ stent visible', 'pre-contrast diagnostic injection', 'no devices in field'>",
  "ocr": {
    "language":  "<English | Hindi | Tamil | Telugu | Bengali | Marathi | Gujarati | Kannada | Malayalam | Punjabi | Odia | Urdu | Assamese | mixed | none>",
    "verbatim":  "<every legible line in original script — newline-separated>",
    "english":   "<English translation; same as verbatim if already English>"
  },
  "summary":          "<1-3 sentences: what this image shows>",
  "key_observations": ["<bullet 1>", "<bullet 2>", "<bullet 3>"],
  "dates_found":      ["<DD/MM/YYYY>", "..."],
  "image_quality":    {
    "rating":      "good" | "fair" | "poor",
    "limitations": ["<e.g. partial view, low contrast, photograph of monitor, glare, blur>"]
  }
}

FORBIDDEN
- Diagnosing.
- Inventing text not visible in the image.
- Approving / rejecting claims.
- Markdown, code fences, or prose around the JSON.`;

// ── PDF / image preprocessing ───────────────────────────────────────────
async function pdfToBuffers(p: string): Promise<Buffer[]> {
  const out: Buffer[] = [];
  const doc = await pdf(p, { scale: PDF_SCALE });
  for await (const page of doc) out.push(page);
  return out;
}

async function optimize(buf: Buffer): Promise<Buffer> {
  return sharp(buf)
    .rotate()
    .resize({ width: MAX_EDGE, height: MAX_EDGE, fit: "inside", withoutEnlargement: true })
    .jpeg({ quality: JPEG_Q, mozjpeg: true })
    .toBuffer();
}

const toBase64 = (b: Buffer) => b.toString("base64");

// ── Provider-agnostic model call (Ollama OR Fireworks) ──────────────────
// Returns a normalised shape: { content, prompt_tokens, completion_tokens }
// regardless of which provider answered.
const USER_INSTRUCTION = (label: string) =>
  `IMAGE: ${label}\n\nProduce the JSON object described in the system prompt. ` +
  `OCR every legible line verbatim. Translate non-English text to English. ` +
  `Describe what is visible — body region, modality, observations. ` +
  `ONE JSON object. NO PROSE.`;

async function callOllama(imageB64: string, label: string): Promise<Response> {
  return fetch(OLLAMA_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model:  MODEL,
      stream: false,
      format: "json",
      messages: [
        { role: "system", content: SYSTEM },
        { role: "user",   content: USER_INSTRUCTION(label), images: [imageB64] },
      ],
      options: {
        temperature:    TEMPERATURE,
        top_p:          1,
        top_k:          TOP_K,
        repeat_penalty: REPEAT_PEN,
        num_predict:    NUM_PREDICT,
      },
    }),
  });
}

async function callFireworks(imageB64: string, label: string): Promise<Response> {
  // OpenAI-compat: image goes in content[] as image_url with a data URL.
  return fetch(FIREWORKS_URL, {
    method: "POST",
    headers: {
      "Content-Type":  "application/json",
      "Authorization": `Bearer ${FIREWORKS_API_KEY}`,
    },
    body: JSON.stringify({
      model: MODEL,
      messages: [
        { role: "system", content: SYSTEM },
        {
          role: "user",
          content: [
            { type: "text",      text: USER_INSTRUCTION(label) },
            { type: "image_url", image_url: { url: `data:image/jpeg;base64,${imageB64}` } },
          ],
        },
      ],
      response_format: { type: "json_object" },
      temperature:     TEMPERATURE,
      top_p:           1,
      top_k:           TOP_K,
      max_tokens:      NUM_PREDICT,
    }),
  });
}

async function callModel(imageB64: string, label: string, retries = 3): Promise<{ content: string; prompt_eval_count?: number; eval_count?: number }> {
  let lastErr: any;
  for (let attempt = 0; attempt < retries; attempt++) {
    const res = PROVIDER === "fireworks"
      ? await callFireworks(imageB64, label)
      : await callOllama(imageB64, label);

    if (res.ok) {
      const j = await res.json() as any;
      // Normalise the two response shapes.
      if (PROVIDER === "fireworks") {
        return {
          content:           j?.choices?.[0]?.message?.content ?? "",
          prompt_eval_count: j?.usage?.prompt_tokens,
          eval_count:        j?.usage?.completion_tokens,
        };
      }
      return {
        content:           j?.message?.content ?? "",
        prompt_eval_count: j?.prompt_eval_count,
        eval_count:        j?.eval_count,
      };
    }
    const txt = await res.text();
    if ([429, 500, 502, 503, 504].includes(res.status) && attempt < retries - 1) {
      const wait = (2 ** attempt + 1) * 1000;
      console.warn(`    ⚠  HTTP ${res.status}; retry in ${wait / 1000}s`);
      await new Promise(r => setTimeout(r, wait));
      lastErr = new Error(`HTTP ${res.status}: ${txt}`);
      continue;
    }
    throw new Error(`HTTP ${res.status}: ${txt}`);
  }
  throw lastErr;
}

function parseJson(text: string): any {
  let s = text.trim();
  const fence = s.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fence?.[1]) s = fence[1].trim();
  const start = s.indexOf("{");
  const end   = s.lastIndexOf("}");
  if (start !== -1 && end > start) s = s.slice(start, end + 1);
  try { return JSON.parse(s); }
  catch { return JSON.parse(jsonrepair(s)); }
}

// ── Walk a claim folder → list of (file, page-buffer) records ───────────
type ImageRef = { source: string; page: number | null; pages_total: number | null; buf: Buffer };

async function listImagesInClaim(claimDir: string): Promise<ImageRef[]> {
  const files = readdirSync(claimDir)
    .filter(f => {
      const ext = extname(f).toLowerCase();
      return statSync(join(claimDir, f)).isFile() && (ext === ".pdf" || SUPPORTED_IMG.has(ext));
    })
    .sort();

  const out: ImageRef[] = [];
  for (const f of files) {
    const full = join(claimDir, f);
    const ext  = extname(f).toLowerCase();
    if (ext === ".pdf") {
      const pages = await pdfToBuffers(full);
      pages.forEach((buf, i) => out.push({ source: f, page: i + 1, pages_total: pages.length, buf }));
    } else {
      out.push({ source: f, page: null, pages_total: null, buf: readFileSync(full) });
    }
  }
  return out;
}

// ── Pipeline: one claim → catalog file ──────────────────────────────────
async function catalogClaim(packageFolder: string, claimId: string, limit?: number): Promise<void> {
  const claimDir = join(CLAIMS_ROOT, packageFolder, claimId);
  if (!statSync(claimDir).isDirectory()) throw new Error(`Not a folder: ${claimDir}`);

  console.log(`\n→ ${packageFolder}/${claimId}`);
  let images = await listImagesInClaim(claimDir);
  if (limit) images = images.slice(0, limit);
  if (!images.length) { console.warn(`  no images`); return; }

  console.log(`  ${images.length} image(s) total`);

  const records: any[] = [];
  for (const [idx, ref] of images.entries()) {
    const label = ref.page ? `${ref.source} page ${ref.page}/${ref.pages_total}` : ref.source;
    process.stdout.write(`  [${idx + 1}/${images.length}] ${label} … `);

    try {
      const optBuf = await optimize(ref.buf);
      const t0 = Date.now();
      const resp = await callModel(toBase64(optBuf), label);
      const dt   = Date.now() - t0;
      const obj  = parseJson(resp.content || "{}");

      records.push({
        image_index: idx,
        source:      ref.source,
        page:        ref.page,
        pages_total: ref.pages_total,
        ...obj,
        _usage: {
          prompt_tokens:     resp.prompt_eval_count,
          completion_tokens: resp.eval_count,
          total_duration_ms: dt,
        },
      });
      console.log(`${obj.image_type ?? "?"}  (${dt} ms)`);
    } catch (e: any) {
      console.log(`ERROR — ${e.message}`);
      records.push({ image_index: idx, source: ref.source, page: ref.page, error: e.message });
    }
  }

  // Write the catalog next to the source files inside the claim folder.
  const catalogDir = claimDir;
  mkdirSync(catalogDir, { recursive: true });

  const outFile = join(catalogDir, "image_catalog.json");
  writeFileSync(outFile, JSON.stringify({
    claim_id:  claimId,
    package:   packageFolder,
    model:     MODEL,
    n_images:  records.length,
    records,
  }, null, 2));
  console.log(`  ✓ ${resolve(outFile)}`);

  const mdFile = join(catalogDir, "image_catalog.md");
  writeFileSync(mdFile, renderMarkdown(claimId, packageFolder, records));
  console.log(`  ✓ ${resolve(mdFile)}`);

  // Roll the per-image records up into a whole-claim summary.
  try {
    await summarizeClaim(packageFolder, claimId);
  } catch (e: any) {
    console.warn(`  ⚠  claim summary failed: ${e.message}`);
  }
}

// ── Human-readable Markdown summary (sibling to image_catalog.json) ─────
function renderMarkdown(claimId: string, packageFolder: string, records: any[]): string {
  const lines: string[] = [];
  lines.push(`# Image Catalog — ${claimId}`);
  lines.push("");
  lines.push(`**Package:** ${packageFolder}  ·  **Model:** ${MODEL}  ·  **Images:** ${records.length}`);
  lines.push("");

  // ── Overview table — at-a-glance per image
  lines.push("## Overview");
  lines.push("");
  lines.push("| # | Source | Page | Type | Body region | Stage | Quality |");
  lines.push("|---|--------|------|------|-------------|-------|---------|");
  for (const r of records) {
    if (r.error) {
      lines.push(`| ${r.image_index} | ${r.source} | ${r.page ?? "—"} | _error_ | — | — | — |`);
      continue;
    }
    const pg = r.page ? `${r.page}/${r.pages_total}` : "—";
    lines.push(
      `| ${r.image_index} | ${r.source} | ${pg} | ${r.image_type ?? "?"} | ${r.body_region ?? "?"} | ${r.stage_of_care ?? "?"} | ${r.image_quality?.rating ?? "?"} |`,
    );
  }
  lines.push("");

  // ── Per-image deep section
  for (const r of records) {
    const pg = r.page ? ` — page ${r.page}/${r.pages_total}` : "";
    lines.push(`## [${r.image_index}] ${r.source}${pg}`);
    lines.push("");

    if (r.error) {
      lines.push(`> **Error:** ${r.error}`);
      lines.push("");
      continue;
    }

    const meta: string[] = [];
    if (r.image_type)    meta.push(`**Type:** ${r.image_type}`);
    if (r.body_region)   meta.push(`**Body region:** ${r.body_region}`);
    if (r.modality_view && r.modality_view !== "N/A") meta.push(`**View:** ${r.modality_view}`);
    if (r.stage_of_care) meta.push(`**Stage:** ${r.stage_of_care}${r.stage_evidence ? ` (${r.stage_evidence})` : ""}`);
    if (r.image_quality?.rating) {
      const lim = (r.image_quality.limitations ?? []).filter(Boolean);
      meta.push(`**Quality:** ${r.image_quality.rating}${lim.length ? ` — ${lim.join(", ")}` : ""}`);
    }
    if (meta.length) { lines.push(meta.join("  ·  ")); lines.push(""); }

    if (r.summary) {
      lines.push(`**Summary.** ${r.summary}`);
      lines.push("");
    }

    if (Array.isArray(r.key_observations) && r.key_observations.length) {
      lines.push("**Key observations:**");
      for (const obs of r.key_observations) lines.push(`- ${obs}`);
      lines.push("");
    }

    if (Array.isArray(r.dates_found) && r.dates_found.length) {
      lines.push(`**Dates found:** ${r.dates_found.join(", ")}`);
      lines.push("");
    }

    const lang = r.ocr?.language;
    const verbatim = (r.ocr?.verbatim ?? "").trim();
    const english  = (r.ocr?.english ?? "").trim();
    if (verbatim || english) {
      lines.push(`**OCR** _(language: ${lang ?? "?"})_`);
      if (verbatim) {
        lines.push("");
        lines.push("<details><summary>Verbatim (original script)</summary>");
        lines.push("");
        lines.push("```");
        lines.push(verbatim);
        lines.push("```");
        lines.push("");
        lines.push("</details>");
      }
      // Show the English version inline (most useful for review); skip if identical to verbatim already.
      if (english && english !== verbatim) {
        lines.push("");
        lines.push("**English:**");
        lines.push("");
        lines.push("```");
        lines.push(english);
        lines.push("```");
      }
      lines.push("");
    }

    lines.push("---");
    lines.push("");
  }

  return lines.join("\n");
}

// ── CLI ─────────────────────────────────────────────────────────────────
const args  = Bun.argv.slice(2);
const limit = (() => {
  const i = args.indexOf("--limit");
  return i >= 0 ? Number(args[i + 1]) : undefined;
})();

const pkg     = args[0];
const claimId = args[1] && args[1] !== "--limit" ? args[1] : undefined;

if (!pkg) {
  console.log("Usage:");
  console.log("  bun ps2_image_catalog.ts <PACKAGE_FOLDER> [CLAIM_ID] [--limit N]");
  console.log("");
  console.log(`Provider: ${PROVIDER}`);
  console.log(`Endpoint: ${PROVIDER === "fireworks" ? FIREWORKS_URL : OLLAMA_URL}`);
  console.log(`Model:    ${MODEL}`);
  console.log("");
  console.log("Switch provider:");
  console.log("  PROVIDER=ollama bun ps2_image_catalog.ts <PACKAGE> [CLAIM]    # local Ollama fallback");
  console.log("  MODEL=accounts/.../deployments/<id> bun ps2_image_catalog.ts ...   # different Fireworks deployment");
  process.exit(0);
}

const pkgDir = join(CLAIMS_ROOT, pkg);
if (!existsSync(pkgDir)) { console.error(`No folder: ${pkgDir}`); process.exit(1); }

if (claimId) {
  await catalogClaim(pkg, claimId, limit);
} else {
  const claims = readdirSync(pkgDir).filter(d => statSync(join(pkgDir, d)).isDirectory()).sort();
  for (const c of claims) await catalogClaim(pkg, c, limit);
}
