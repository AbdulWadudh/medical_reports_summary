#!/usr/bin/env bun
/**
 * PS3 Claim Summary — synthesise one whole-claim summary from the per-image
 * catalog produced by ps2_image_catalog.ts.
 *
 * Input  : Claims/<PACKAGE>/<CLAIM>/image_catalog.json
 * Output : Claims/<PACKAGE>/<CLAIM>/claim_summary.md
 *
 * Usage:
 *   bun ps3_claim_summary.ts <PACKAGE_FOLDER> <CLAIM_ID>   # one claim
 *   bun ps3_claim_summary.ts <PACKAGE_FOLDER>              # every claim in pkg
 *   bun ps3_claim_summary.ts --all                         # every catalog under Claims/
 *
 * Also exported as `summarizeClaim(pkg, claimId)` so ps2 can chain it.
 */

import { readdirSync, statSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { jsonrepair } from "jsonrepair";

// ── Project root + .env loader ──────────────────────────────────────────
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
const PROJECT_ROOT = findProjectRoot();
const CLAIMS_ROOT  = join(PROJECT_ROOT, "Claims");

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

// ── Provider config (mirrors ps2; defaults to Fireworks) ────────────────
const PROVIDER: "ollama" | "fireworks" =
  (Bun.env.PROVIDER as any) ?? "fireworks";

const OLLAMA_URL = Bun.env.OLLAMA_URL ?? "http://localhost:11434/api/chat";

const FIREWORKS_URL     = Bun.env.FIREWORKS_URL ?? "https://api.fireworks.ai/inference/v1/chat/completions";
const FIREWORKS_API_KEY = Bun.env.FIREWORKS_API_KEY;

const MODEL = Bun.env.SUMMARY_MODEL ?? Bun.env.MODEL ?? (
  PROVIDER === "fireworks"
    ? "accounts/ajeya-rao-k-eckusf6m/deployments/euufjyfd"
    : "qwen3-vl:8b-instruct"
);

if (PROVIDER === "fireworks" && !FIREWORKS_API_KEY) {
  throw new Error("PROVIDER=fireworks but FIREWORKS_API_KEY is not set in .env");
}

const TEMPERATURE = 0.1;
const NUM_PREDICT = 4096;

// ── System prompt ───────────────────────────────────────────────────────
const SYSTEM = `You are a medical-claim summarisation assistant for India's
AB PM-JAY claims dataset. You receive a JSON list of per-image OCR + description
records produced by a vision model for ONE claim folder. Your job is to fuse
those records into a single whole-claim summary.

ROLE — STRICTLY DESCRIPTIVE
- Do not diagnose. Do not approve / reject. Do not recommend treatment.
- Describe what the dossier shows, gaps included.
- When uncertain, say so. Never invent identifiers, dates, or findings.

INPUTS YOU CAN USE
- Each record has: source filename, page, image_type, body_region, modality_view,
  stage_of_care, summary, key_observations, dates_found, image_quality,
  and an OCR block (verbatim + english).
- Combine OCR text + image observations to identify patient name, age, sex,
  hospital, doctor, dates, procedure performed, and overall narrative.
- Treat OCR field 'english' as the primary text source. Use 'verbatim' to
  resolve disagreements or to capture proper nouns / IDs.

OUTPUT — exactly ONE JSON object, no markdown, no code fence, no prose:

{
  "patient": {
    "name":      "<string or 'unknown'>",
    "age":       "<e.g. '54 years' or 'unknown'>",
    "sex":       "<Male | Female | unknown>",
    "id_numbers":["<any patient/MRN/UHID/Aadhaar-style numbers found>"]
  },
  "hospital": {
    "name":     "<string or 'unknown'>",
    "location": "<city / state if visible, else 'unknown'>",
    "doctors":  ["<doctor names with degrees if visible>"]
  },
  "encounter": {
    "date_range":          "<earliest to latest DD/MM/YYYY found, or 'unknown'>",
    "all_dates":           ["<every distinct DD/MM/YYYY>"],
    "primary_procedure":   "<short phrase, e.g. 'Coronary angiography + PCI to LAD' or 'unknown'>",
    "package_code":        "<PMJAY / scheme code if visible, else 'unknown'>"
  },
  "image_inventory": {
    "total_images":        <int>,
    "by_type":             { "<image_type>": <count>, "...": <count> },
    "stages_present":      ["<pre-procedure | intra-procedure | post-procedure | uncertain | n/a>"],
    "languages_seen":      ["<English | Hindi | ...>"]
  },
  "clinical_narrative": "<2-4 sentence plain-English description of what the dossier portrays — patient context, what was done, what the imaging shows>",
  "key_findings":       ["<bullet — synthesised across images>", "..."],
  "completeness": {
    "has_pre_procedure_imaging":   <true|false>,
    "has_intra_procedure_imaging": <true|false>,
    "has_post_procedure_imaging":  <true|false>,
    "has_typed_report":            <true|false>,
    "has_handwritten_notes":       <true|false>,
    "has_signed_stamp":            <true|false>,
    "notes":                       "<any caveats, e.g. 'no post-procedure imaging present'>"
  },
  "concerns_or_gaps": ["<missing piece, illegible doc, contradiction between records, low-quality scan, etc.>"]
}

FORBIDDEN
- Diagnosing.
- Approving / rejecting the claim.
- Inventing names, IDs, or dates that do not appear in the records.
- Markdown, code fences, or prose around the JSON.`;

// ── Provider call ───────────────────────────────────────────────────────
async function callFireworks(userText: string): Promise<Response> {
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
        { role: "user",   content: userText },
      ],
      response_format:   { type: "json_object" },
      temperature:       TEMPERATURE,
      max_tokens:        NUM_PREDICT,
      frequency_penalty: 0.4,   // suppress n-gram loops triggered by repetitive OCR junk
      presence_penalty:  0.1,
    }),
  });
}

async function callOllama(userText: string): Promise<Response> {
  return fetch(OLLAMA_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model:  MODEL,
      stream: false,
      format: "json",
      messages: [
        { role: "system", content: SYSTEM },
        { role: "user",   content: userText },
      ],
      options: { temperature: TEMPERATURE, num_predict: NUM_PREDICT },
    }),
  });
}

async function callModel(userText: string, retries = 3): Promise<string> {
  let lastErr: any;
  for (let attempt = 0; attempt < retries; attempt++) {
    const res = PROVIDER === "fireworks"
      ? await callFireworks(userText)
      : await callOllama(userText);

    if (res.ok) {
      const j = await res.json() as any;
      return PROVIDER === "fireworks"
        ? (j?.choices?.[0]?.message?.content ?? "")
        : (j?.message?.content ?? "");
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

// Map common drift in model-generated keys back to our canonical schema.
function pickKey(obj: any, ...candidates: string[]): any {
  if (!obj || typeof obj !== "object") return undefined;
  for (const k of candidates) if (k in obj) return obj[k];
  // case-insensitive fallback
  const lower = Object.fromEntries(Object.keys(obj).map(k => [k.toLowerCase(), k]));
  for (const k of candidates) {
    const hit = lower[k.toLowerCase()];
    if (hit) return obj[hit];
  }
  return undefined;
}

function toBool(v: any): boolean | undefined {
  if (typeof v === "boolean") return v;
  if (typeof v === "number")  return v !== 0;
  if (typeof v === "string") {
    const s = v.trim().toLowerCase();
    if (["true", "yes", "y", "1", "present"].includes(s))   return true;
    if (["false", "no", "n", "0", "absent", "none"].includes(s)) return false;
  }
  return undefined;
}

function normalizeSummary(raw: any): any {
  const c = pickKey(raw, "completeness") ?? {};
  return {
    patient:             pickKey(raw, "patient") ?? {},
    hospital:            pickKey(raw, "hospital") ?? {},
    encounter:           pickKey(raw, "encounter") ?? {},
    image_inventory:     pickKey(raw, "image_inventory", "imageInventory") ?? {},
    clinical_narrative:  pickKey(raw, "clinical_narrative", "Clinical_narrative", "clinicalNarrative", "narrative") ?? "",
    key_findings:        pickKey(raw, "key_findings", "kay_findings", "keyFindings", "findings") ?? [],
    completeness: {
      has_pre_procedure_imaging:   toBool(pickKey(c, "has_pre_procedure_imaging",   "is_pre_procedure_imaging_present",   "pre_procedure_imaging")),
      has_intra_procedure_imaging: toBool(pickKey(c, "has_intra_procedure_imaging", "is_intra_procedure_imaging_present", "intra_procedure_imaging")),
      has_post_procedure_imaging:  toBool(pickKey(c, "has_post_procedure_imaging",  "is_post_procedure_imaging_present",  "post_procedure_imaging")),
      has_typed_report:            toBool(pickKey(c, "has_typed_report",            "is_typed_report_present",            "typed_report")),
      has_handwritten_notes:       toBool(pickKey(c, "has_handwritten_notes",       "is_handwritten_notes_present",       "handwritten_notes")),
      has_signed_stamp:            toBool(pickKey(c, "has_signed_stamp",            "is_signed_stamp_present",            "signed_stamp", "has_stamp")),
      notes:                       pickKey(c, "notes", "note") ?? "",
    },
    concerns_or_gaps:    pickKey(raw, "concerns_or_gaps", "gaps_or_concerns", "concerns", "gaps") ?? [],
  };
}

// ── Compact the per-image catalog to keep the prompt focused ────────────
// Drop _usage and stage_evidence; dedupe + cap noisy fields so repeated OCR
// junk doesn't trigger the model's n-gram loops.
const MAX_LIST_PER_RECORD = 12;
const MAX_OCR_CHARS       = 2000;

function dedupCap(arr: any, max = MAX_LIST_PER_RECORD): string[] {
  if (!Array.isArray(arr)) return [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const v of arr) {
    const s = String(v ?? "").trim();
    if (!s) continue;
    if (seen.has(s)) continue;
    seen.add(s);
    out.push(s);
    if (out.length >= max) break;
  }
  return out;
}

function trimText(s: any, max = MAX_OCR_CHARS): string {
  if (typeof s !== "string") return "";
  const collapsed = s.replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
  return collapsed.length > max ? collapsed.slice(0, max) + "…[truncated]" : collapsed;
}

function compactCatalog(catalog: any): any {
  const records = (catalog.records ?? []).map((r: any) => {
    if (r.error) return { image_index: r.image_index, source: r.source, page: r.page, error: r.error };
    return {
      image_index:      r.image_index,
      source:           r.source,
      page:             r.page,
      pages_total:      r.pages_total,
      image_type:       r.image_type,
      body_region:      r.body_region,
      modality_view:    r.modality_view,
      stage_of_care:    r.stage_of_care,
      summary:          trimText(r.summary, 600),
      key_observations: dedupCap(r.key_observations),
      dates_found:      dedupCap(r.dates_found),
      ocr: {
        language: r.ocr?.language,
        verbatim: trimText(r.ocr?.verbatim),
        english:  trimText(r.ocr?.english),
      },
      image_quality:    r.image_quality,
    };
  });
  return {
    claim_id: catalog.claim_id,
    package:  catalog.package,
    n_images: catalog.n_images ?? records.length,
    records,
  };
}

// ── Public API: summarise one claim from its image_catalog.json ─────────
export async function summarizeClaim(packageFolder: string, claimId: string): Promise<void> {
  const claimDir = join(CLAIMS_ROOT, packageFolder, claimId);
  const catalogPath = join(claimDir, "image_catalog.json");
  if (!existsSync(catalogPath)) {
    console.warn(`  ⚠  no image_catalog.json — skip summary (${catalogPath})`);
    return;
  }

  console.log(`\n→ summarise ${packageFolder}/${claimId}`);
  const catalog = JSON.parse(readFileSync(catalogPath, "utf-8"));
  const compact = compactCatalog(catalog);

  const userText =
    `Claim: ${packageFolder}/${claimId}\n` +
    `Per-image catalog (JSON below). Produce the whole-claim summary JSON object ` +
    `described in the system prompt. ONE JSON object. NO PROSE.\n\n` +
    JSON.stringify(compact, null, 2);

  const t0 = Date.now();
  const raw = await callModel(userText);
  const dt  = Date.now() - t0;
  const parsed = parseJson(raw || "{}");
  const obj = normalizeSummary(parsed);

  const out = {
    claim_id: catalog.claim_id ?? claimId,
    package:  catalog.package  ?? packageFolder,
    model:    MODEL,
    n_images: catalog.n_images ?? compact.records.length,
    generated_ms: dt,
    ...obj,
  };

  const mdPath = join(claimDir, "claim_summary.md");
  writeFileSync(mdPath, renderMarkdown(out));
  console.log(`  ✓ ${resolve(mdPath)}  (${dt} ms)`);
}

// ── Markdown rendering ──────────────────────────────────────────────────
function renderMarkdown(s: any): string {
  const lines: string[] = [];
  lines.push(`# Claim Summary — ${s.claim_id}`);
  lines.push("");
  lines.push(`**Package:** ${s.package}  ·  **Model:** ${s.model}  ·  **Images:** ${s.n_images}`);
  lines.push("");

  const p = s.patient ?? {};
  lines.push("## Patient");
  lines.push(`- **Name:** ${p.name ?? "unknown"}`);
  lines.push(`- **Age:** ${p.age ?? "unknown"}`);
  lines.push(`- **Sex:** ${p.sex ?? "unknown"}`);
  if (Array.isArray(p.id_numbers) && p.id_numbers.length) {
    lines.push(`- **IDs:** ${p.id_numbers.join(", ")}`);
  }
  lines.push("");

  const h = s.hospital ?? {};
  lines.push("## Hospital");
  lines.push(`- **Name:** ${h.name ?? "unknown"}`);
  lines.push(`- **Location:** ${h.location ?? "unknown"}`);
  if (Array.isArray(h.doctors) && h.doctors.length) {
    lines.push(`- **Doctors:** ${h.doctors.join("; ")}`);
  }
  lines.push("");

  const e = s.encounter ?? {};
  lines.push("## Encounter");
  lines.push(`- **Date range:** ${e.date_range ?? "unknown"}`);
  lines.push(`- **Primary procedure:** ${e.primary_procedure ?? "unknown"}`);
  lines.push(`- **Package code:** ${e.package_code ?? "unknown"}`);
  if (Array.isArray(e.all_dates) && e.all_dates.length) {
    lines.push(`- **All dates seen:** ${e.all_dates.join(", ")}`);
  }
  lines.push("");

  if (s.clinical_narrative) {
    lines.push("## Clinical narrative");
    lines.push("");
    lines.push(s.clinical_narrative);
    lines.push("");
  }

  if (Array.isArray(s.key_findings) && s.key_findings.length) {
    lines.push("## Key findings");
    for (const f of s.key_findings) lines.push(`- ${f}`);
    lines.push("");
  }

  const inv = s.image_inventory ?? {};
  lines.push("## Image inventory");
  lines.push(`- **Total images:** ${inv.total_images ?? s.n_images}`);
  if (inv.by_type && typeof inv.by_type === "object") {
    lines.push(`- **By type:**`);
    for (const [k, v] of Object.entries(inv.by_type)) lines.push(`  - ${k}: ${v}`);
  }
  if (Array.isArray(inv.stages_present) && inv.stages_present.length) {
    lines.push(`- **Stages present:** ${inv.stages_present.join(", ")}`);
  }
  if (Array.isArray(inv.languages_seen) && inv.languages_seen.length) {
    lines.push(`- **Languages:** ${inv.languages_seen.join(", ")}`);
  }
  lines.push("");

  const c = s.completeness ?? {};
  lines.push("## Completeness");
  lines.push(`- Pre-procedure imaging: ${c.has_pre_procedure_imaging ?? "?"}`);
  lines.push(`- Intra-procedure imaging: ${c.has_intra_procedure_imaging ?? "?"}`);
  lines.push(`- Post-procedure imaging: ${c.has_post_procedure_imaging ?? "?"}`);
  lines.push(`- Typed report: ${c.has_typed_report ?? "?"}`);
  lines.push(`- Handwritten notes: ${c.has_handwritten_notes ?? "?"}`);
  lines.push(`- Signed stamp: ${c.has_signed_stamp ?? "?"}`);
  if (c.notes) lines.push(`- **Notes:** ${c.notes}`);
  lines.push("");

  if (Array.isArray(s.concerns_or_gaps) && s.concerns_or_gaps.length) {
    lines.push("## Concerns / gaps");
    for (const g of s.concerns_or_gaps) lines.push(`- ${g}`);
    lines.push("");
  }

  return lines.join("\n");
}

// ── CLI ─────────────────────────────────────────────────────────────────
// Only run the CLI when invoked directly, not when imported by ps2.
if (import.meta.main) {
  const args = Bun.argv.slice(2);
  const all  = args.includes("--all");
  const pkg     = args[0] && args[0] !== "--all" ? args[0] : undefined;
  const claimId = args[1] && args[1] !== "--all"  ? args[1] : undefined;

  if (!all && !pkg) {
    console.log("Usage:");
    console.log("  bun ps3_claim_summary.ts <PACKAGE_FOLDER> [CLAIM_ID]");
    console.log("  bun ps3_claim_summary.ts --all");
    console.log("");
    console.log(`Provider: ${PROVIDER}`);
    console.log(`Model:    ${MODEL}`);
    process.exit(0);
  }

  if (all) {
    const pkgs = readdirSync(CLAIMS_ROOT).filter(d => statSync(join(CLAIMS_ROOT, d)).isDirectory()).sort();
    for (const p of pkgs) {
      const claims = readdirSync(join(CLAIMS_ROOT, p))
        .filter(d => statSync(join(CLAIMS_ROOT, p, d)).isDirectory())
        .sort();
      for (const c of claims) {
        if (existsSync(join(CLAIMS_ROOT, p, c, "image_catalog.json"))) {
          await summarizeClaim(p, c);
        }
      }
    }
  } else if (claimId) {
    await summarizeClaim(pkg!, claimId);
  } else {
    const claims = readdirSync(join(CLAIMS_ROOT, pkg!))
      .filter(d => statSync(join(CLAIMS_ROOT, pkg!, d)).isDirectory())
      .sort();
    for (const c of claims) {
      if (existsSync(join(CLAIMS_ROOT, pkg!, c, "image_catalog.json"))) {
        await summarizeClaim(pkg!, c);
      }
    }
  }
}
