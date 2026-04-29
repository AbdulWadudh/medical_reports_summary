#!/usr/bin/env bun
/**
 * PS3 Claim Summary — synthesise one whole-claim summary from the per-image
 * catalog produced by ps2_image_catalog.ts.
 *
 * Input  : Claims/<PACKAGE>/<CLAIM>/image_catalog.json
 * Output : Claims/<PACKAGE>/<CLAIM>/claim_summary.json   (dashboard data)
 *          Claims/<PACKAGE>/<CLAIM>/claim_summary.md     (human-readable preview)
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
const NUM_PREDICT = 16000;  // dashboard schema is large; truncation = empty fields

// ── System prompt — dashboard schema ────────────────────────────────────
const SYSTEM_DASHBOARD = `You are a medical-claim summarisation assistant for India's
AB PM-JAY claims dataset. You receive a JSON list of per-image OCR + description
records produced by a vision model for ONE claim folder. Fuse those records
into a SINGLE whole-claim summary in the schema below — a frontend reviewer
dashboard renders the result directly.

ROLE — STRICTLY DESCRIPTIVE
- Do not diagnose. Do not approve / reject. Do not recommend treatment.
- Describe what the dossier shows, gaps included.

NULL HANDLING — VERY IMPORTANT
- Use the JSON literal null (NOT the string "null", NOT "unknown", NOT "N/A")
  for any field whose value cannot be derived from the records.
- Booleans must be true / false literals. If unknown → null.
- Percentages are integers 0-100 with no % sign. If unknown → null.
- Dates: DD/MM/YYYY string only. If unknown → null.
- Empty array [] = "checked, none found". null = "cannot determine".
- Never invent identifiers, dates, or findings.

INPUTS YOU CAN USE
- Per-record fields: source, page, image_type, body_region, modality_view,
  stage_of_care, summary, key_observations, dates_found, image_quality, ocr.
- Treat ocr.english as the primary text source for "report" content.
- Image observations describe what the IMAGES show.
- "Report" content = OCR'd typed reports / handwritten notes inside the dossier.

OUTPUT — exactly ONE JSON object matching this schema. No markdown, no prose.

{
  "header": {
    "claim_id":     "<copy from input>",
    "patient_name": "<string|null>",
    "modality":     "<dominant modality e.g. 'X-Ray'|'CT'|'MRI'|'Ultrasound'|'Coronary Angiogram'|'Mixed'|null>",
    "body_part":    "<dominant body region e.g. 'Tibia'|'Coronary tree'|'Abdomen'|null>",
    "study_date":   "<earliest study date DD/MM/YYYY|null>",
    "reviewer":     null
  },
  "status": {
    "consistency":         "consistent|partial|mismatch|null",
    "confidence_pct":      <int 0-100|null>,
    "clinical_risk_score": "Low|Medium|High|null",
    "key_findings": [
      { "finding": "<short label>", "ai_detected": <bool|null>, "report_mentioned": <bool|null>, "note": "<string|null>" }
    ]
  },
  "scan_viewer": {
    "primary_image_source": "<best representative filename|null>",
    "detected_regions": [
      { "label": "<e.g. 'Fracture zone'>", "image_source": "<filename>", "page": <int|null> }
    ],
    "ai_overlays_available": false
  },
  "ai_clinical_findings": {
    "fracture":           { "present": <bool|null>, "confidence_pct": <int|null>, "evidence": "<string|null>" },
    "fluid_accumulation": { "present": <bool|null>, "confidence_pct": <int|null>, "evidence": "<string|null>" },
    "tumor_mass":         { "present": <bool|null>, "confidence_pct": <int|null>, "evidence": "<string|null>" },
    "infiltration":       { "severity": "None|Mild|Moderate|Severe|null", "confidence_pct": <int|null>, "evidence": "<string|null>" },
    "image_quality":      "good|fair|poor|null"
  },
  "report_nlp_extraction": {
    "reported_diagnosis":    "<string|null>",
    "reported_severity":     "Minor|Moderate|Severe|null",
    "reported_findings":     ["<string>"],
    "extraction_confidence": "High|Medium|Low|null"
  },
  "inconsistency_detection": {
    "possible_exaggerations": ["<string>"],
    "underreported_findings": ["<string>"],
    "hidden_findings":        ["<string>"]
  },
  "stg_alignment": {
    "claimed_package":          "<string|null>",
    "evidence_required":        [ { "item": "<string>", "present": <bool|null> } ],
    "stg_compliance_score_pct": <int|null>
  },
}

GUIDANCE FOR SPECIFIC SECTIONS

status.consistency: "consistent" if image findings and report broadly agree;
"partial" if some mismatch; "mismatch" if major conflicts.

ai_clinical_findings: report on what THE IMAGES (not the report) show. Set
.present=true only if the image observations describe the finding. .evidence
should cite the image source filename(s).

report_nlp_extraction: report on what the WRITTEN REPORTS / NOTES (OCR) say.
Independent of image findings.

inconsistency_detection:
  possible_exaggerations = report claims things images don't show.
  underreported_findings = images show things report didn't mention.
  hidden_findings        = anything obviously omitted from both.

stg_alignment: PMJAY package codes look like "MC011A", "SU007A". claimed_package
is the input package. evidence_required is a short generic checklist for that
procedure type. Mark .present per the dossier.

FORBIDDEN
- Diagnosing.
- Approving / rejecting the claim.
- Inventing names, IDs, or dates that do not appear in the records.
- Returning the string "null" instead of the literal null.
- Markdown, code fences, or prose around the JSON.`;

// ── System prompt — reference detail (patient/hospital/encounter/etc) ───
const SYSTEM_REFERENCE = `You are a medical-claim summarisation assistant for India's
AB PM-JAY claims dataset. You receive a JSON list of per-image OCR + description
records produced by a vision model for ONE claim folder. Extract patient,
hospital, encounter, image-inventory, narrative, completeness, and gap details.

ROLE — STRICTLY DESCRIPTIVE. No diagnoses. No claim approval / rejection.

NULL HANDLING
- Use literal null (not "unknown", not "N/A", not the string "null") for any
  field you cannot derive from the records.
- Booleans must be true/false literals. Unknown → null.
- Dates: DD/MM/YYYY only.
- Empty array [] = "checked, none found"; null = "cannot determine".
- Never invent.

OUTPUT — exactly ONE JSON object. No markdown. No prose.

{
  "patient": {
    "name":       "<string|null>",
    "age":        "<e.g. '54 years'|null>",
    "sex":        "Male|Female|null",
    "id_numbers": ["<MRN/UHID/Aadhaar etc.>"]
  },
  "hospital": {
    "name":     "<string|null>",
    "location": "<city/state|null>",
    "doctors":  ["<string with degrees if visible>"]
  },
  "encounter": {
    "date_range":        "<DD/MM/YYYY to DD/MM/YYYY|null>",
    "all_dates":         ["<every distinct DD/MM/YYYY>"],
    "primary_procedure": "<short phrase|null>",
    "package_code":      "<PMJAY scheme code|null>"
  },
  "image_inventory": {
    "total_images":   <int>,
    "by_type":        { "<image_type>": <count> },
    "stages_present": ["pre-procedure|intra-procedure|post-procedure|uncertain|n/a"],
    "languages_seen": ["English|Hindi|..."]
  },
  "clinical_narrative": "<2-4 sentence plain-English description|null>",
  "completeness": {
    "has_pre_procedure_imaging":   <bool|null>,
    "has_intra_procedure_imaging": <bool|null>,
    "has_post_procedure_imaging":  <bool|null>,
    "has_typed_report":            <bool|null>,
    "has_handwritten_notes":       <bool|null>,
    "has_signed_stamp":            <bool|null>,
    "notes":                       "<string|null>"
  },
  "concerns_or_gaps": ["<string>"]
}

FORBIDDEN
- Diagnosing / approving / rejecting.
- Inventing names, IDs, or dates.
- Markdown, code fences, or prose around the JSON.`;

// ── Provider call ───────────────────────────────────────────────────────
async function callFireworks(systemText: string, userText: string): Promise<Response> {
  return fetch(FIREWORKS_URL, {
    method: "POST",
    headers: {
      "Content-Type":  "application/json",
      "Authorization": `Bearer ${FIREWORKS_API_KEY}`,
    },
    body: JSON.stringify({
      model: MODEL,
      messages: [
        { role: "system", content: systemText },
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

async function callOllama(systemText: string, userText: string): Promise<Response> {
  return fetch(OLLAMA_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model:  MODEL,
      stream: false,
      format: "json",
      messages: [
        { role: "system", content: systemText },
        { role: "user",   content: userText },
      ],
      options: { temperature: TEMPERATURE, num_predict: NUM_PREDICT },
    }),
  });
}

async function callModel(systemText: string, userText: string, retries = 5): Promise<string> {
  let lastErr: any;
  for (let attempt = 0; attempt < retries; attempt++) {
    const res = PROVIDER === "fireworks"
      ? await callFireworks(systemText, userText)
      : await callOllama(systemText, userText);

    if (res.ok) {
      const j = await res.json() as any;
      return PROVIDER === "fireworks"
        ? (j?.choices?.[0]?.message?.content ?? "")
        : (j?.message?.content ?? "");
    }
    const txt = await res.text();
    // Fireworks deployments scale to zero when idle. The cold-start can take
    // 1-3 minutes, so use a longer backoff for 503 (scaling) than for transient errors.
    const isScalingUp = res.status === 503 && /scaling.up|scaled.to.zero/i.test(txt);
    if ([429, 500, 502, 503, 504].includes(res.status) && attempt < retries - 1) {
      const wait = isScalingUp ? 30_000 * (attempt + 1) : (2 ** attempt + 1) * 1000;
      console.warn(`    ⚠  HTTP ${res.status}${isScalingUp ? " (deployment scaling up)" : ""}; retry in ${wait / 1000}s`);
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

// ── Schema normalization ─────────────────────────────────────────────────
// Goal: enforce dashboard-ready schema with null (not "unknown"/"N/A"/"") for
// missing values, real booleans (not strings), int percentages, and arrays.

function pickKey(obj: any, ...candidates: string[]): any {
  if (!obj || typeof obj !== "object") return undefined;
  for (const k of candidates) if (k in obj) return obj[k];
  const lower = Object.fromEntries(Object.keys(obj).map(k => [k.toLowerCase(), k]));
  for (const k of candidates) {
    const hit = lower[k.toLowerCase()];
    if (hit) return obj[hit];
  }
  return undefined;
}

const NULLISH_STRINGS = new Set(["", "unknown", "n/a", "na", "none", "null", "?", "—", "-"]);

// Coerce missing-ish strings ("unknown", "N/A", "") to null.
function nzStr(v: any): string | null {
  if (v === null || v === undefined) return null;
  if (typeof v !== "string") return v == null ? null : String(v);
  const s = v.trim();
  if (NULLISH_STRINGS.has(s.toLowerCase())) return null;
  return s;
}

function nzBool(v: any): boolean | null {
  if (typeof v === "boolean") return v;
  if (typeof v === "number") return v !== 0;
  if (typeof v === "string") {
    const s = v.trim().toLowerCase();
    if (["true", "yes", "y", "1", "present"].includes(s)) return true;
    if (["false", "no", "n", "0", "absent"].includes(s)) return false;
  }
  return null;
}

function nzInt(v: any): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return Math.round(v);
  if (typeof v === "string") {
    const m = v.match(/-?\d+(\.\d+)?/);
    if (m) return Math.round(Number(m[0]));
  }
  return null;
}

function nzArr(v: any): any[] | null {
  if (!Array.isArray(v)) return null;
  return v;
}

function nzEnum(v: any, allowed: string[]): string | null {
  const s = nzStr(v);
  if (s == null) return null;
  const hit = allowed.find(a => a.toLowerCase() === s.toLowerCase());
  return hit ?? null;
}

// Walk an object/array tree and turn "unknown"/"N/A"/"null" strings into null.
function deepNullify(v: any): any {
  if (v === null || v === undefined) return null;
  if (typeof v === "string") {
    return NULLISH_STRINGS.has(v.trim().toLowerCase()) ? null : v;
  }
  if (Array.isArray(v)) return v.map(deepNullify);
  if (typeof v === "object") {
    const out: any = {};
    for (const [k, val] of Object.entries(v)) out[k] = deepNullify(val);
    return out;
  }
  return v;
}

function normalizeFinding(f: any): any {
  if (!f || typeof f !== "object") f = {};
  return {
    present:        nzBool(pickKey(f, "present", "detected", "is_present")),
    confidence_pct: nzInt(pickKey(f, "confidence_pct", "confidence", "confidence_percentage")),
    evidence:       nzStr(pickKey(f, "evidence", "source", "note")),
  };
}

function normalizeSummary(raw: any, packageFolder: string, claimId: string): any {
  raw = deepNullify(raw ?? {});

  const header     = pickKey(raw, "header") ?? {};
  const status     = pickKey(raw, "status") ?? {};
  const scan       = pickKey(raw, "scan_viewer", "scanViewer") ?? {};
  const aiF        = pickKey(raw, "ai_clinical_findings", "aiClinicalFindings", "ai_findings") ?? {};
  const multi      = pickKey(raw, "multi_image_analysis", "multiImageAnalysis") ?? {};
  const nlp        = pickKey(raw, "report_nlp_extraction", "reportNlpExtraction", "report_extraction") ?? {};
  const corr       = pickKey(raw, "finding_correlation", "findingCorrelation") ?? {};
  const inc        = pickKey(raw, "inconsistency_detection", "inconsistencyDetection") ?? {};
  const stg        = pickKey(raw, "stg_alignment", "stgAlignment") ?? {};
  const tl         = pickKey(raw, "radiology_timeline", "radiologyTimeline") ?? {};
  const patient    = pickKey(raw, "patient") ?? {};
  const hospital   = pickKey(raw, "hospital") ?? {};
  const encounter  = pickKey(raw, "encounter") ?? {};
  const inv        = pickKey(raw, "image_inventory", "imageInventory") ?? {};
  const c          = pickKey(raw, "completeness") ?? {};

  return {
    header: {
      claim_id:     claimId,
      patient_name: nzStr(pickKey(header, "patient_name", "patientName", "name")) ?? nzStr(pickKey(patient, "name")),
      modality:     nzStr(pickKey(header, "modality")),
      body_part:    nzStr(pickKey(header, "body_part", "bodyPart", "body_region")),
      study_date:   nzStr(pickKey(header, "study_date", "studyDate", "date")),
      reviewer:     null,
    },
    status: {
      consistency:         nzEnum(pickKey(status, "consistency"), ["consistent", "partial", "mismatch"]),
      confidence_pct:      nzInt(pickKey(status, "confidence_pct", "confidence")),
      clinical_risk_score: nzEnum(pickKey(status, "clinical_risk_score", "clinicalRiskScore", "risk"), ["Low", "Medium", "High"]),
      key_findings: (nzArr(pickKey(status, "key_findings", "keyFindings", "findings")) ?? []).map((f: any) => ({
        finding:          nzStr(pickKey(f, "finding", "label", "name")),
        ai_detected:      nzBool(pickKey(f, "ai_detected", "image_ai", "ai")),
        report_mentioned: nzBool(pickKey(f, "report_mentioned", "report", "in_report")),
        note:             nzStr(pickKey(f, "note", "evidence", "detail")),
      })),
    },
    scan_viewer: {
      primary_image_source: nzStr(pickKey(scan, "primary_image_source", "primary", "source")),
      detected_regions: (nzArr(pickKey(scan, "detected_regions", "regions")) ?? []).map((r: any) => ({
        label:        nzStr(pickKey(r, "label", "name")),
        image_source: nzStr(pickKey(r, "image_source", "source")),
        page:         nzInt(pickKey(r, "page")),
      })),
      ai_overlays_available: false,
    },
    ai_clinical_findings: {
      fracture:           normalizeFinding(pickKey(aiF, "fracture")),
      fluid_accumulation: normalizeFinding(pickKey(aiF, "fluid_accumulation", "fluidAccumulation", "fluid")),
      tumor_mass:         normalizeFinding(pickKey(aiF, "tumor_mass", "tumorMass", "mass", "tumor")),
      infiltration: {
        severity:       nzEnum(pickKey(pickKey(aiF, "infiltration") ?? {}, "severity"), ["None", "Mild", "Moderate", "Severe"]),
        confidence_pct: nzInt(pickKey(pickKey(aiF, "infiltration") ?? {}, "confidence_pct", "confidence")),
        evidence:       nzStr(pickKey(pickKey(aiF, "infiltration") ?? {}, "evidence")),
      },
      image_quality: nzEnum(pickKey(aiF, "image_quality", "imageQuality"), ["good", "fair", "poor"]),
    },
    multi_image_analysis: {
      entries: (nzArr(pickKey(multi, "entries", "items")) ?? []).map((e: any) => ({
        modality:  nzStr(pickKey(e, "modality")),
        day:       nzInt(pickKey(e, "day")),
        date:      nzStr(pickKey(e, "date")),
        finding:   nzStr(pickKey(e, "finding")),
        confirmed: nzBool(pickKey(e, "confirmed")),
      })),
      consistency_score_pct: nzInt(pickKey(multi, "consistency_score_pct", "consistency_score", "consistencyScore")),
    },
    report_nlp_extraction: {
      reported_diagnosis:    nzStr(pickKey(nlp, "reported_diagnosis", "diagnosis")),
      reported_severity:     nzEnum(pickKey(nlp, "reported_severity", "severity"), ["Minor", "Moderate", "Severe"]),
      reported_findings:     (nzArr(pickKey(nlp, "reported_findings", "findings")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
      extraction_confidence: nzEnum(pickKey(nlp, "extraction_confidence", "confidence"), ["High", "Medium", "Low"]),
    },
    finding_correlation: (() => {
      const rows = (nzArr(pickKey(corr, "rows", "items")) ?? []).map((r: any) => ({
        finding:  nzStr(pickKey(r, "finding", "label")),
        image_ai: nzBool(pickKey(r, "image_ai", "ai", "imageAi")),
        report:   nzBool(pickKey(r, "report", "in_report", "reportMentioned")),
        match:    nzBool(pickKey(r, "match")),
      }));
      // Compute match if missing; compute consistency_score from rows if model omitted.
      for (const r of rows) {
        if (r.match === null && r.image_ai !== null && r.report !== null) {
          r.match = r.image_ai === r.report;
        }
      }
      let score = nzInt(pickKey(corr, "consistency_score_pct", "consistency_score", "consistencyScore"));
      if (score === null && rows.length) {
        const decided = rows.filter(r => r.match !== null);
        if (decided.length) {
          score = Math.round(100 * decided.filter(r => r.match === true).length / decided.length);
        }
      }
      return { rows, consistency_score_pct: score };
    })(),
    inconsistency_detection: {
      possible_exaggerations: (nzArr(pickKey(inc, "possible_exaggerations", "exaggerations")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
      underreported_findings: (nzArr(pickKey(inc, "underreported_findings", "underreported")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
      hidden_findings:        (nzArr(pickKey(inc, "hidden_findings", "hidden")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
    },
    stg_alignment: {
      claimed_package: nzStr(pickKey(stg, "claimed_package", "claimedPackage", "package")) ?? packageFolder,
      evidence_required: (nzArr(pickKey(stg, "evidence_required", "evidence", "checklist")) ?? []).map((e: any) => ({
        item:    nzStr(pickKey(e, "item", "label", "name")),
        present: nzBool(pickKey(e, "present", "satisfied")),
      })),
      stg_compliance_score_pct: nzInt(pickKey(stg, "stg_compliance_score_pct", "compliance_score", "complianceScore")),
    },
    radiology_timeline: {
      events: (nzArr(pickKey(tl, "events", "items")) ?? []).map((ev: any) => ({
        day:   nzInt(pickKey(ev, "day")),
        date:  nzStr(pickKey(ev, "date")),
        event: nzStr(pickKey(ev, "event", "description", "label")),
      })),
      logical: nzBool(pickKey(tl, "logical")),
    },
    patient: {
      name:       nzStr(pickKey(patient, "name")),
      age:        nzStr(pickKey(patient, "age")),
      sex:        nzEnum(pickKey(patient, "sex"), ["Male", "Female"]),
      id_numbers: (nzArr(pickKey(patient, "id_numbers", "ids")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
    },
    hospital: {
      name:     nzStr(pickKey(hospital, "name")),
      location: nzStr(pickKey(hospital, "location")),
      doctors:  (nzArr(pickKey(hospital, "doctors")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
    },
    encounter: {
      date_range:        nzStr(pickKey(encounter, "date_range", "dateRange")),
      all_dates:         (nzArr(pickKey(encounter, "all_dates", "allDates", "dates")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
      primary_procedure: nzStr(pickKey(encounter, "primary_procedure", "procedure")),
      package_code:      nzStr(pickKey(encounter, "package_code", "packageCode")),
    },
    image_inventory: {
      total_images:   nzInt(pickKey(inv, "total_images", "totalImages")),
      by_type:        pickKey(inv, "by_type", "byType") ?? {},
      stages_present: (nzArr(pickKey(inv, "stages_present", "stagesPresent")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
      languages_seen: (nzArr(pickKey(inv, "languages_seen", "languagesSeen")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
    },
    clinical_narrative: nzStr(pickKey(raw, "clinical_narrative", "Clinical_narrative", "clinicalNarrative", "narrative")),
    completeness: {
      has_pre_procedure_imaging:   nzBool(pickKey(c, "has_pre_procedure_imaging",   "is_pre_procedure_imaging_present",   "pre_procedure_imaging")),
      has_intra_procedure_imaging: nzBool(pickKey(c, "has_intra_procedure_imaging", "is_intra_procedure_imaging_present", "intra_procedure_imaging")),
      has_post_procedure_imaging:  nzBool(pickKey(c, "has_post_procedure_imaging",  "is_post_procedure_imaging_present",  "post_procedure_imaging")),
      has_typed_report:            nzBool(pickKey(c, "has_typed_report",            "is_typed_report_present",            "typed_report")),
      has_handwritten_notes:       nzBool(pickKey(c, "has_handwritten_notes",       "is_handwritten_notes_present",       "handwritten_notes")),
      has_signed_stamp:            nzBool(pickKey(c, "has_signed_stamp",            "is_signed_stamp_present",            "signed_stamp", "has_stamp")),
      notes:                       nzStr(pickKey(c, "notes", "note")),
    },
    concerns_or_gaps: (nzArr(pickKey(raw, "concerns_or_gaps", "gaps_or_concerns", "concerns", "gaps")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
  };
}

// ── Compact the per-image catalog to keep the prompt focused ────────────
// Drop _usage and stage_evidence; dedupe + cap noisy fields so repeated OCR
// junk doesn't trigger the model's n-gram loops.
const MAX_LIST_PER_RECORD = 8;
const MAX_OCR_CHARS       = 1200;

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
  // Two parallel calls: dashboard schema + reference detail. The full schema
  // overflowed a single response (model returned null for half the fields),
  // so split it.
  const [rawDashboard, rawReference] = await Promise.all([
    callModel(SYSTEM_DASHBOARD, userText),
    callModel(SYSTEM_REFERENCE, userText),
  ]);
  const dt = Date.now() - t0;
  const merged = { ...parseJson(rawReference || "{}"), ...parseJson(rawDashboard || "{}") };
  const normalized = normalizeSummary(merged, packageFolder, claimId);
  const obj = computeDerived(normalized, catalog);

  const out = {
    claim_id:     catalog.claim_id ?? claimId,
    package:      catalog.package  ?? packageFolder,
    model:        MODEL,
    n_images:     catalog.n_images ?? compact.records.length,
    generated_ms: dt,
    ...obj,
  };

  const jsonPath = join(claimDir, "claim_summary.json");
  writeFileSync(jsonPath, JSON.stringify(out, null, 2));
  console.log(`  ✓ ${resolve(jsonPath)}  (${dt} ms)`);

  const mdPath = join(claimDir, "claim_summary.md");
  writeFileSync(mdPath, renderMarkdown(out));
  console.log(`  ✓ ${resolve(mdPath)}`);
}

// ── Deterministic derivations from the per-image catalog ────────────────
// The model is unreliable at filling these tabular sections, so we compute
// them in code from data we already have.

function parseDate(s: string | null | undefined): Date | null {
  if (!s) return null;
  const m = s.match(/(\d{1,2})\/(\d{1,2})\/(\d{4})/);
  if (!m) return null;
  const [_, dd, mm, yyyy] = m;
  const d = new Date(Number(yyyy), Number(mm) - 1, Number(dd));
  return Number.isNaN(d.getTime()) ? null : d;
}

function deriveFindingCorrelation(s: any): any {
  const kf = s?.status?.key_findings ?? [];
  const rows = kf.map((f: any) => ({
    finding:  f.finding ?? null,
    image_ai: f.ai_detected,
    report:   f.report_mentioned,
    match:    (f.ai_detected !== null && f.report_mentioned !== null)
              ? f.ai_detected === f.report_mentioned
              : null,
  }));
  let score: number | null = null;
  const decided = rows.filter((r: any) => r.match !== null);
  if (decided.length) {
    score = Math.round(100 * decided.filter((r: any) => r.match === true).length / decided.length);
  }
  return { rows, consistency_score_pct: score };
}

function deriveMultiImageAnalysis(catalog: any): any {
  const records: any[] = catalog.records ?? [];
  // Group by image_type. For each group, pick earliest date, dominant body region,
  // and a "confirmed" flag if at least one record's stage is non-null and not "uncertain".
  const groups = new Map<string, any[]>();
  for (const r of records) {
    if (r.error) continue;
    const t = r.image_type ?? "Other";
    if (!groups.has(t)) groups.set(t, []);
    groups.get(t)!.push(r);
  }

  const earliest = (rs: any[]): string | null => {
    const dates = rs.flatMap(r => r.dates_found ?? [])
      .map((d: string) => ({ s: d, d: parseDate(d) }))
      .filter((x: any) => x.d !== null) as any[];
    if (!dates.length) return null;
    dates.sort((a: any, b: any) => a.d - b.d);
    return dates[0].s;
  };

  const allDates = records.flatMap(r => (r.dates_found ?? []) as string[])
    .map(s => ({ s, d: parseDate(s) }))
    .filter(x => x.d !== null) as { s: string; d: Date }[];
  const minDate = allDates.length ? allDates.reduce((a, b) => a.d < b.d ? a : b).d : null;

  const entries = Array.from(groups.entries()).map(([modality, rs]) => {
    const date = earliest(rs);
    const day  = (() => {
      const d = parseDate(date);
      if (!d || !minDate) return null;
      return Math.round((d.getTime() - minDate.getTime()) / 86400000) + 1;
    })();
    const confirmed = rs.some(r => {
      const stage = r.stage_of_care;
      return stage && !["uncertain", "n/a", "none"].includes(String(stage).toLowerCase());
    });
    const finding = rs.map(r => r.body_region).filter(Boolean)[0] ?? null;
    return { modality, day, date, finding, confirmed };
  });

  return { entries, consistency_score_pct: null };
}

function deriveRadiologyTimeline(catalog: any): any {
  const records: any[] = catalog.records ?? [];
  // Build (date, modality) pairs, dedupe by (date, modality), sort by date.
  const pairs = new Map<string, { date: string; modality: string; date_obj: Date }>();
  let minDate: Date | null = null;
  for (const r of records) {
    if (r.error) continue;
    const modality = r.image_type ?? "Other";
    for (const ds of (r.dates_found ?? []) as string[]) {
      const d = parseDate(ds);
      if (!d) continue;
      if (!minDate || d < minDate) minDate = d;
      const key = `${ds}::${modality}`;
      if (!pairs.has(key)) pairs.set(key, { date: ds, modality, date_obj: d });
    }
  }

  const events = Array.from(pairs.values())
    .sort((a, b) => a.date_obj.getTime() - b.date_obj.getTime())
    .map(p => ({
      day:   minDate ? Math.round((p.date_obj.getTime() - minDate.getTime()) / 86400000) + 1 : null,
      date:  p.date,
      event: p.modality,
    }));

  // logical = ascending dates only (true if all sorted dates are non-decreasing — always true after sort)
  // Set null if no events; otherwise true (sorted by definition).
  return { events, logical: events.length ? true : null };
}

function deriveCompleteness(s: any, catalog: any): any {
  const records: any[] = catalog.records ?? [];
  const stages = new Set<string>(records.map(r => String(r.stage_of_care ?? "").toLowerCase()).filter(Boolean));
  const types  = new Set<string>(records.map(r => String(r.image_type ?? "").toLowerCase()).filter(Boolean));

  const hasStage = (needle: string) => records.length === 0 ? null
    : Array.from(stages).some(s => s.includes(needle));
  const hasType = (needle: string) => records.length === 0 ? null
    : Array.from(types).some(t => t.includes(needle));

  // Prefer model's value when it's a real bool; otherwise fall back to derivation.
  const c = s?.completeness ?? {};
  const merge = (modelVal: any, derived: any) => modelVal === true || modelVal === false ? modelVal : derived;

  return {
    has_pre_procedure_imaging:   merge(c.has_pre_procedure_imaging,   hasStage("pre")),
    has_intra_procedure_imaging: merge(c.has_intra_procedure_imaging, hasStage("intra")),
    has_post_procedure_imaging:  merge(c.has_post_procedure_imaging,  hasStage("post")),
    has_typed_report:            merge(c.has_typed_report,            hasType("typed")),
    has_handwritten_notes:       merge(c.has_handwritten_notes,       hasType("handwritten")),
    has_signed_stamp:            merge(c.has_signed_stamp,            hasType("stamp") || hasType("signature")),
    notes:                       c.notes ?? null,
  };
}

function computeDerived(s: any, catalog: any): any {
  return {
    ...s,
    finding_correlation:   deriveFindingCorrelation(s),
    multi_image_analysis:  deriveMultiImageAnalysis(catalog),
    radiology_timeline:    deriveRadiologyTimeline(catalog),
    completeness:          deriveCompleteness(s, catalog),
  };
}

// ── Markdown rendering — mirrors the dashboard layout ───────────────────
const STATUS_DOT: Record<string, string> = {
  consistent: "🟢", partial: "🟡", mismatch: "🔴",
};
const dash = (v: any) => v === null || v === undefined || v === "" ? "—" : String(v);
const tick = (v: any) => v === true ? "✔" : v === false ? "✖" : "—";

function renderMarkdown(s: any): string {
  const L: string[] = [];
  const h = s.header ?? {};
  const st = s.status ?? {};

  // Header bar
  L.push(`# Claim Summary — ${s.claim_id}`);
  L.push("");
  L.push(`**Package:** ${s.package}  ·  **Model:** ${s.model}  ·  **Images:** ${s.n_images}`);
  L.push("");
  L.push("| Claim ID | Patient | Modality | Body Part | Study Date | Reviewer |");
  L.push("|---|---|---|---|---|---|");
  L.push(`| ${dash(h.claim_id)} | ${dash(h.patient_name)} | ${dash(h.modality)} | ${dash(h.body_part)} | ${dash(h.study_date)} | ${dash(h.reviewer)} |`);
  L.push("");

  // Status banner
  const dot = STATUS_DOT[st.consistency ?? ""] ?? "⚪";
  const consistencyLabel = ({
    consistent: "IMAGE & REPORT CONSISTENT",
    partial:    "PARTIAL MATCH",
    mismatch:   "MISMATCH",
  } as Record<string, string>)[st.consistency ?? ""] ?? "STATUS UNKNOWN";
  L.push(`> ${dot} **${consistencyLabel}**  ·  Confidence: ${st.confidence_pct ?? "—"}%  ·  Clinical Risk Score: ${dash(st.clinical_risk_score)}`);
  L.push("");
  if (Array.isArray(st.key_findings) && st.key_findings.length) {
    L.push("**Key findings:**");
    for (const f of st.key_findings) {
      const ai = f.ai_detected === true ? "AI ✔" : f.ai_detected === false ? "AI ✖" : "AI —";
      const rp = f.report_mentioned === true ? "Report ✔" : f.report_mentioned === false ? "Report ✖" : "Report —";
      const note = f.note ? ` — ${f.note}` : "";
      L.push(`- ${dash(f.finding)} _(${ai} · ${rp})_${note}`);
    }
    L.push("");
  }

  // Left column
  L.push("## Interactive Scan Viewer");
  const sv = s.scan_viewer ?? {};
  L.push(`- **Primary image:** ${dash(sv.primary_image_source)}`);
  L.push(`- **AI overlays available:** ${sv.ai_overlays_available ? "yes" : "no"}`);
  if (Array.isArray(sv.detected_regions) && sv.detected_regions.length) {
    L.push(`- **Detected regions:**`);
    for (const r of sv.detected_regions) {
      L.push(`  - ${dash(r.label)} _(in ${dash(r.image_source)}${r.page ? ` p.${r.page}` : ""})_`);
    }
  }
  L.push("");

  L.push("## AI Clinical Findings");
  const ai = s.ai_clinical_findings ?? {};
  const f = (label: string, x: any) => {
    if (!x) { L.push(`- **${label}:** —`); return; }
    const conf = x.confidence_pct != null ? ` (${x.confidence_pct}%)` : "";
    const ev   = x.evidence ? ` — ${x.evidence}` : "";
    L.push(`- **${label}:** ${tick(x.present)}${conf}${ev}`);
  };
  f("Fracture",           ai.fracture);
  f("Fluid accumulation", ai.fluid_accumulation);
  f("Tumor / mass",       ai.tumor_mass);
  if (ai.infiltration) {
    const conf = ai.infiltration.confidence_pct != null ? ` (${ai.infiltration.confidence_pct}%)` : "";
    L.push(`- **Infiltration:** ${dash(ai.infiltration.severity)}${conf}${ai.infiltration.evidence ? ` — ${ai.infiltration.evidence}` : ""}`);
  }
  L.push(`- **Image quality:** ${dash(ai.image_quality)}`);
  L.push("");

  L.push("## Multi-image Analysis");
  const mi = s.multi_image_analysis ?? {};
  if (Array.isArray(mi.entries) && mi.entries.length) {
    L.push("| Modality | Day | Date | Finding | Confirmed |");
    L.push("|---|---|---|---|---|");
    for (const e of mi.entries) {
      L.push(`| ${dash(e.modality)} | ${dash(e.day)} | ${dash(e.date)} | ${dash(e.finding)} | ${tick(e.confirmed)} |`);
    }
  } else { L.push("_no multi-image entries_"); }
  L.push("");
  L.push(`**Consistency Score:** ${mi.consistency_score_pct ?? "—"}%`);
  L.push("");

  // Right column
  L.push("## Report NLP Extraction");
  const nlp = s.report_nlp_extraction ?? {};
  L.push(`- **Reported diagnosis:** ${dash(nlp.reported_diagnosis)}`);
  L.push(`- **Reported severity:** ${dash(nlp.reported_severity)}`);
  if (Array.isArray(nlp.reported_findings) && nlp.reported_findings.length) {
    L.push(`- **Reported findings:**`);
    for (const x of nlp.reported_findings) L.push(`  - ${x}`);
  } else { L.push(`- **Reported findings:** —`); }
  L.push(`- **Extraction confidence:** ${dash(nlp.extraction_confidence)}`);
  L.push("");

  L.push("## Finding Correlation");
  const fc = s.finding_correlation ?? {};
  if (Array.isArray(fc.rows) && fc.rows.length) {
    L.push("| Finding | Image AI | Report | Match |");
    L.push("|---|---|---|---|");
    for (const r of fc.rows) {
      L.push(`| ${dash(r.finding)} | ${tick(r.image_ai)} | ${tick(r.report)} | ${tick(r.match)} |`);
    }
  } else { L.push("_no correlation rows_"); }
  L.push("");
  L.push(`**Consistency Score:** ${fc.consistency_score_pct ?? "—"}%`);
  L.push("");

  L.push("## Inconsistency Detection");
  const inc = s.inconsistency_detection ?? {};
  const list = (label: string, arr: any) => {
    if (Array.isArray(arr) && arr.length) {
      L.push(`- **${label}:**`);
      for (const x of arr) L.push(`  - ⚠ ${x}`);
    } else {
      L.push(`- **${label}:** ✔ none detected`);
    }
  };
  list("Possible exaggerations", inc.possible_exaggerations);
  list("Underreported findings", inc.underreported_findings);
  list("Hidden findings",        inc.hidden_findings);
  L.push("");

  L.push("## STG Alignment");
  const stg = s.stg_alignment ?? {};
  L.push(`- **Claimed package:** ${dash(stg.claimed_package)}`);
  if (Array.isArray(stg.evidence_required) && stg.evidence_required.length) {
    L.push(`- **Evidence required:**`);
    for (const e of stg.evidence_required) L.push(`  - ${dash(e.item)} ${tick(e.present)}`);
  }
  L.push(`- **STG compliance score:** ${stg.stg_compliance_score_pct ?? "—"}%`);
  L.push("");

  L.push("## Radiology Timeline");
  const tl = s.radiology_timeline ?? {};
  if (Array.isArray(tl.events) && tl.events.length) {
    for (const e of tl.events) {
      const day = e.day != null ? `Day ${e.day} – ` : "";
      const date = e.date ? ` _(${e.date})_` : "";
      L.push(`- ${day}${dash(e.event)}${date}`);
    }
  } else { L.push("_no events_"); }
  L.push("");
  L.push(`**Timeline logical:** ${tick(tl.logical)}`);
  L.push("");

  // Reference info — patient/hospital/encounter/inventory/completeness/gaps
  L.push("---");
  L.push("");
  L.push("## Reference detail");
  L.push("");

  const p = s.patient ?? {};
  L.push("**Patient**");
  L.push(`- Name: ${dash(p.name)} · Age: ${dash(p.age)} · Sex: ${dash(p.sex)}`);
  if (Array.isArray(p.id_numbers) && p.id_numbers.length) L.push(`- IDs: ${p.id_numbers.join(", ")}`);
  L.push("");

  const ho = s.hospital ?? {};
  L.push("**Hospital**");
  L.push(`- Name: ${dash(ho.name)} · Location: ${dash(ho.location)}`);
  if (Array.isArray(ho.doctors) && ho.doctors.length) L.push(`- Doctors: ${ho.doctors.join("; ")}`);
  L.push("");

  const en = s.encounter ?? {};
  L.push("**Encounter**");
  L.push(`- Date range: ${dash(en.date_range)} · Procedure: ${dash(en.primary_procedure)} · Package code: ${dash(en.package_code)}`);
  if (Array.isArray(en.all_dates) && en.all_dates.length) L.push(`- All dates seen: ${en.all_dates.join(", ")}`);
  L.push("");

  const inv = s.image_inventory ?? {};
  L.push("**Image inventory**");
  L.push(`- Total: ${inv.total_images ?? s.n_images}`);
  if (inv.by_type && Object.keys(inv.by_type).length) {
    const types = Object.entries(inv.by_type).map(([k, v]) => `${k}: ${v}`).join(", ");
    L.push(`- By type: ${types}`);
  }
  if (Array.isArray(inv.stages_present) && inv.stages_present.length) L.push(`- Stages: ${inv.stages_present.join(", ")}`);
  if (Array.isArray(inv.languages_seen) && inv.languages_seen.length) L.push(`- Languages: ${inv.languages_seen.join(", ")}`);
  L.push("");

  if (s.clinical_narrative) {
    L.push("**Clinical narrative**");
    L.push("");
    L.push(s.clinical_narrative);
    L.push("");
  }

  const c = s.completeness ?? {};
  L.push("**Completeness**");
  L.push(`- Pre: ${tick(c.has_pre_procedure_imaging)} · Intra: ${tick(c.has_intra_procedure_imaging)} · Post: ${tick(c.has_post_procedure_imaging)}`);
  L.push(`- Typed report: ${tick(c.has_typed_report)} · Handwritten: ${tick(c.has_handwritten_notes)} · Signed stamp: ${tick(c.has_signed_stamp)}`);
  if (c.notes) L.push(`- Notes: ${c.notes}`);
  L.push("");

  if (Array.isArray(s.concerns_or_gaps) && s.concerns_or_gaps.length) {
    L.push("**Concerns / gaps**");
    for (const g of s.concerns_or_gaps) L.push(`- ${g}`);
    L.push("");
  }

  return L.join("\n");
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
