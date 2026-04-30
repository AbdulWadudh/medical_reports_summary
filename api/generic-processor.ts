import { readdirSync, statSync, readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { join, extname, resolve } from "node:path";
import { pdf } from "pdf-to-img";
import sharp from "sharp";
import { jsonrepair } from "jsonrepair";

// ── Project-Wide Configuration ───────────────────────────────────────────
const PROVIDER: "ollama" | "fireworks" = (process.env.PROVIDER as any) ?? "fireworks";
const OLLAMA_URL = process.env.OLLAMA_URL ?? "http://localhost:11434/api/chat";
const FIREWORKS_URL = process.env.FIREWORKS_URL ?? "https://api.fireworks.ai/inference/v1/chat/completions";
const FIREWORKS_API_KEY = process.env.FIREWORKS_API_KEY;

const MODEL = process.env.MODEL ?? (
  PROVIDER === "fireworks"
    ? "accounts/ajeya-rao-k-eckusf6m/deployments/euufjyfd"
    : "qwen3-vl:8b-instruct"
);
const SUMMARY_MODEL = process.env.SUMMARY_MODEL ?? MODEL;

const MAX_EDGE = 1280;
const JPEG_Q = 90;
const PDF_SCALE = 220 / 72;
const NUM_PREDICT = 4096;
const TEMPERATURE = 0.05;
const TOP_K = 30;
const REPEAT_PEN = 1.0;
const SUPPORTED_IMG = new Set([".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"]);

// ── Prompts from ps2_image_catalog.ts ────────────────────────────────────
const SYSTEM_OCR = `You are a medical document OCR + description assistant for India's
AB PM-JAY claims dataset. You receive ONE image at a time. The image may be:
  - A medical scan (X-ray, IVP, KUB, USG, CT, MRI, coronary angiogram still)
  - A typed clinical / radiology report (English or Indic-language)
  - A handwritten doctor's note or prescription
  - A doctor's stamp, signature block, hospital letterhead
  - A photograph of a cath-lab monitor
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

// ── Prompts from ps3_claim_summary.ts ────────────────────────────────────
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

FORBIDDEN
- Diagnosing.
- Approving / rejecting the claim.
- Inventing names, IDs, or dates that do not appear in the records.
- Returning the string "null" instead of the literal null.
- Markdown, code fences, or prose around the JSON.`;

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

// ── Utility Functions ───────────────────────────────────────────────────

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

function parseJson(text: string): any {
  let s = text.trim();
  const fence = s.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fence?.[1]) s = fence[1].trim();
  const start = s.indexOf("{");
  const end = s.lastIndexOf("}");
  if (start !== -1 && end > start) s = s.slice(start, end + 1);
  try { return JSON.parse(s); }
  catch { return JSON.parse(jsonrepair(s)); }
}

const USER_INSTRUCTION = (label: string) =>
  `IMAGE: ${label}\n\nProduce the JSON object described in the system prompt. ` +
  `OCR every legible line verbatim. Translate non-English text to English. ` +
  `Describe what is visible — body region, modality, observations. ` +
  `ONE JSON object. NO PROSE.`;

async function callModel(systemText: string, userText: string, imageB64?: string, model = MODEL, retries = 5): Promise<any> {
  console.log(`[Processor] callModel - Using model: ${model} (Retries: ${retries})`);
  let lastErr: any;
  for (let attempt = 0; attempt < retries; attempt++) {
    const isFireworks = PROVIDER === "fireworks";
    const url = isFireworks ? FIREWORKS_URL : OLLAMA_URL;

    let messages: any[] = [{ role: "system", content: systemText }];
    if (imageB64) {
      messages.push({
        role: "user",
        content: isFireworks
          ? [{ type: "text", text: userText }, { type: "image_url", image_url: { url: `data:image/jpeg;base64,${imageB64}` } }]
          : userText // Ollama takes images separately in its proprietary API but for simplicity we assume chat/compat here
      });
    } else {
      messages.push({ role: "user", content: userText });
    }

    const body: any = isFireworks
      ? { model, messages, response_format: { type: "json_object" }, temperature: TEMPERATURE, max_tokens: 16000 }
      : { model, messages, stream: false, format: "json", images: imageB64 ? [imageB64] : undefined, options: { temperature: TEMPERATURE, num_predict: 16000 } };

    const resp = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(isFireworks ? { "Authorization": `Bearer ${FIREWORKS_API_KEY}` } : {})
      },
      body: JSON.stringify(body)
    });

    if (resp.ok) {
      const j = await resp.json() as any;
      const content = isFireworks ? j?.choices?.[0]?.message?.content : j?.message?.content;
      return {
        content,
        prompt_eval_count: isFireworks ? j?.usage?.prompt_tokens : j?.prompt_eval_count,
        eval_count: isFireworks ? j?.usage?.completion_tokens : j?.eval_count
      };
    }

    const txt = await resp.text();
    if ([429, 500, 502, 503, 504].includes(resp.status) && attempt < retries - 1) {
      const wait = (2 ** attempt + 1) * 1000;
      await new Promise(r => setTimeout(r, wait));
      continue;
    }
    throw new Error(`HTTP ${resp.status}: ${txt}`);
  }
}

// ── Normalization Helpers from ps3 ───────────────────────────────────────

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

function deepNullify(v: any): any {
  if (v === null || v === undefined) return null;
  if (typeof v === "string") return NULLISH_STRINGS.has(v.trim().toLowerCase()) ? null : v;
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
    present: nzBool(pickKey(f, "present", "detected", "is_present")),
    confidence_pct: nzInt(pickKey(f, "confidence_pct", "confidence", "confidence_percentage")),
    evidence: nzStr(pickKey(f, "evidence", "source", "note")),
  };
}

function normalizeSummary(raw: any, claimId: string): any {
  raw = deepNullify(raw ?? {});
  const header = pickKey(raw, "header") ?? {};
  const status = pickKey(raw, "status") ?? {};
  const scan = pickKey(raw, "scan_viewer", "scanViewer") ?? {};
  const aiF = pickKey(raw, "ai_clinical_findings", "aiClinicalFindings", "ai_findings") ?? {};
  const multi = pickKey(raw, "multi_image_analysis", "multiImageAnalysis") ?? {};
  const nlp = pickKey(raw, "report_nlp_extraction", "reportNlpExtraction", "report_extraction") ?? {};
  const corr = pickKey(raw, "finding_correlation", "findingCorrelation") ?? {};
  const inc = pickKey(raw, "inconsistency_detection", "inconsistencyDetection") ?? {};
  const stg = pickKey(raw, "stg_alignment", "stgAlignment") ?? {};
  const tl = pickKey(raw, "radiology_timeline", "radiologyTimeline") ?? {};
  const patient = pickKey(raw, "patient") ?? {};
  const hospital = pickKey(raw, "hospital") ?? {};
  const encounter = pickKey(raw, "encounter") ?? {};
  const inv = pickKey(raw, "image_inventory", "imageInventory") ?? {};
  const c = pickKey(raw, "completeness") ?? {};

  return {
    header: {
      claim_id: claimId,
      patient_name: nzStr(pickKey(header, "patient_name", "patientName", "name")) ?? nzStr(pickKey(patient, "name")),
      modality: nzStr(pickKey(header, "modality")),
      body_part: nzStr(pickKey(header, "body_part", "bodyPart", "body_region")),
      study_date: nzStr(pickKey(header, "study_date", "studyDate", "date")),
      reviewer: null,
    },
    status: {
      consistency: nzEnum(pickKey(status, "consistency"), ["consistent", "partial", "mismatch"]),
      confidence_pct: nzInt(pickKey(status, "confidence_pct", "confidence")),
      clinical_risk_score: nzEnum(pickKey(status, "clinical_risk_score", "clinicalRiskScore", "risk"), ["Low", "Medium", "High"]),
      key_findings: (nzArr(pickKey(status, "key_findings", "keyFindings", "findings")) ?? []).map((f: any) => ({
        finding: nzStr(pickKey(f, "finding", "label", "name")),
        ai_detected: nzBool(pickKey(f, "ai_detected", "image_ai", "ai")),
        report_mentioned: nzBool(pickKey(f, "report_mentioned", "report", "in_report")),
        note: nzStr(pickKey(f, "note", "evidence", "detail")),
      })),
    },
    scan_viewer: {
      primary_image_source: nzStr(pickKey(scan, "primary_image_source", "primary", "source")),
      detected_regions: (nzArr(pickKey(scan, "detected_regions", "regions")) ?? []).map((r: any) => ({
        label: nzStr(pickKey(r, "label", "name")),
        image_source: nzStr(pickKey(r, "image_source", "source")),
        page: nzInt(pickKey(r, "page")),
      })),
      ai_overlays_available: false,
    },
    ai_clinical_findings: {
      fracture: normalizeFinding(pickKey(aiF, "fracture")),
      fluid_accumulation: normalizeFinding(pickKey(aiF, "fluid_accumulation", "fluidAccumulation", "fluid")),
      tumor_mass: normalizeFinding(pickKey(aiF, "tumor_mass", "tumorMass", "mass", "tumor")),
      infiltration: {
        severity: nzEnum(pickKey(pickKey(aiF, "infiltration") ?? {}, "severity"), ["None", "Mild", "Moderate", "Severe"]),
        confidence_pct: nzInt(pickKey(pickKey(aiF, "infiltration") ?? {}, "confidence_pct", "confidence")),
        evidence: nzStr(pickKey(pickKey(aiF, "infiltration") ?? {}, "evidence")),
      },
      image_quality: nzEnum(pickKey(aiF, "image_quality", "imageQuality"), ["good", "fair", "poor"]),
    },
    multi_image_analysis: {
      entries: (nzArr(pickKey(multi, "entries", "items")) ?? []).map((e: any) => ({
        modality: nzStr(pickKey(e, "modality")),
        day: nzInt(pickKey(e, "day")),
        date: nzStr(pickKey(e, "date")),
        finding: nzStr(pickKey(e, "finding")),
        confirmed: nzBool(pickKey(e, "confirmed")),
      })),
      consistency_score_pct: nzInt(pickKey(multi, "consistency_score_pct", "consistency_score", "consistencyScore")),
    },
    report_nlp_extraction: {
      reported_diagnosis: nzStr(pickKey(nlp, "reported_diagnosis", "diagnosis")),
      reported_severity: nzEnum(pickKey(nlp, "reported_severity", "severity"), ["Minor", "Moderate", "Severe"]),
      reported_findings: (nzArr(pickKey(nlp, "reported_findings", "findings")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
      extraction_confidence: nzEnum(pickKey(nlp, "extraction_confidence", "confidence"), ["High", "Medium", "Low"]),
    },
    finding_correlation: (() => {
      const rows = (nzArr(pickKey(corr, "rows", "items")) ?? []).map((r: any) => ({
        finding: nzStr(pickKey(r, "finding", "label")),
        image_ai: nzBool(pickKey(r, "image_ai", "ai", "imageAi")),
        report: nzBool(pickKey(r, "report", "in_report", "reportMentioned")),
        match: nzBool(pickKey(r, "match")),
      }));
      for (const r of rows) {
        if (r.match === null && r.image_ai !== null && r.report !== null) r.match = r.image_ai === r.report;
      }
      let score = nzInt(pickKey(corr, "consistency_score_pct", "consistency_score", "consistencyScore"));
      if (score === null && rows.length) {
        const decided = rows.filter(r => r.match !== null);
        if (decided.length) score = Math.round(100 * decided.filter(r => r.match === true).length / decided.length);
      }
      return { rows, consistency_score_pct: score };
    })(),
    inconsistency_detection: {
      possible_exaggerations: (nzArr(pickKey(inc, "possible_exaggerations", "exaggerations")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
      underreported_findings: (nzArr(pickKey(inc, "underreported_findings", "underreported")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
      hidden_findings: (nzArr(pickKey(inc, "hidden_findings", "hidden")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
    },
    stg_alignment: {
      claimed_package: nzStr(pickKey(stg, "claimed_package", "claimedPackage", "package")),
      evidence_required: (nzArr(pickKey(stg, "evidence_required", "evidence", "checklist")) ?? []).map((e: any) => ({
        item: nzStr(pickKey(e, "item", "label", "name")),
        present: nzBool(pickKey(e, "present", "satisfied")),
      })),
      stg_compliance_score_pct: nzInt(pickKey(stg, "stg_compliance_score_pct", "compliance_score", "complianceScore")),
    },
    radiology_timeline: {
      events: (nzArr(pickKey(tl, "events", "items")) ?? []).map((ev: any) => ({
        day: nzInt(pickKey(ev, "day")),
        date: nzStr(pickKey(ev, "date")),
        event: nzStr(pickKey(ev, "event", "description", "label")),
      })),
      logical: nzBool(pickKey(tl, "logical")),
    },
    patient: {
      name: nzStr(pickKey(patient, "name")),
      age: nzStr(pickKey(patient, "age")),
      sex: nzEnum(pickKey(patient, "sex"), ["Male", "Female"]),
      id_numbers: (nzArr(pickKey(patient, "id_numbers", "ids")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
    },
    hospital: {
      name: nzStr(pickKey(hospital, "name")),
      location: nzStr(pickKey(hospital, "location")),
      doctors: (nzArr(pickKey(hospital, "doctors")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
    },
    encounter: {
      date_range: nzStr(pickKey(encounter, "date_range", "dateRange")),
      all_dates: (nzArr(pickKey(encounter, "all_dates", "allDates", "dates")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
      primary_procedure: nzStr(pickKey(encounter, "primary_procedure", "procedure")),
      package_code: nzStr(pickKey(encounter, "package_code", "packageCode")),
    },
    image_inventory: {
      total_images: nzInt(pickKey(inv, "total_images", "totalImages")),
      by_type: pickKey(inv, "by_type", "byType") ?? {},
      stages_present: (nzArr(pickKey(inv, "stages_present", "stagesPresent")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
      languages_seen: (nzArr(pickKey(inv, "languages_seen", "languagesSeen")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
    },
    clinical_narrative: nzStr(pickKey(raw, "clinical_narrative", "Clinical_narrative", "clinicalNarrative", "narrative")),
    completeness: {
      has_pre_procedure_imaging: nzBool(pickKey(c, "has_pre_procedure_imaging", "is_pre_procedure_imaging_present", "pre_procedure_imaging")),
      has_intra_procedure_imaging: nzBool(pickKey(c, "has_intra_procedure_imaging", "is_intra_procedure_imaging_present", "intra_procedure_imaging")),
      has_post_procedure_imaging: nzBool(pickKey(c, "has_post_procedure_imaging", "is_post_procedure_imaging_present", "post_procedure_imaging")),
      has_typed_report: nzBool(pickKey(c, "has_typed_report", "is_typed_report_present", "typed_report")),
      has_handwritten_notes: nzBool(pickKey(c, "has_handwritten_notes", "is_handwritten_notes_present", "handwritten_notes")),
      has_signed_stamp: nzBool(pickKey(c, "has_signed_stamp", "is_signed_stamp_present", "signed_stamp", "has_stamp")),
      notes: nzStr(pickKey(c, "notes", "note")),
    },
    concerns_or_gaps: (nzArr(pickKey(raw, "concerns_or_gaps", "gaps_or_concerns", "concerns", "gaps")) ?? []).map((x: any) => nzStr(x)).filter(Boolean) as string[],
  };
}

// ── Deterministic Derivations from ps3 ───────────────────────────────────

function parseDate(s: string | null | undefined): Date | null {
  if (!s) return null;
  const m = s.match(/(\d{1,2})\/(\d{1,2})\/(\d{4})/);
  if (!m) return null;
  const d = new Date(Number(m[3]), Number(m[2]) - 1, Number(m[1]));
  return Number.isNaN(d.getTime()) ? null : d;
}

function deriveFindingCorrelation(s: any): any {
  const kf = s?.status?.key_findings ?? [];
  const rows = kf.map((f: any) => ({
    finding: f.finding ?? null,
    image_ai: f.ai_detected,
    report: f.report_mentioned,
    match: (f.ai_detected !== null && f.report_mentioned !== null) ? f.ai_detected === f.report_mentioned : null,
  }));
  let score: number | null = null;
  const decided = rows.filter((r: any) => r.match !== null);
  if (decided.length) score = Math.round(100 * decided.filter((r: any) => r.match === true).length / decided.length);
  return { rows, consistency_score_pct: score };
}

function deriveMultiImageAnalysis(catalog: any): any {
  const records: any[] = catalog.records ?? [];
  const groups = new Map<string, any[]>();
  for (const r of records) {
    if (r.error) continue;
    const t = r.image_type ?? "Other";
    if (!groups.has(t)) groups.set(t, []);
    groups.get(t)!.push(r);
  }
  const allDates = records.flatMap(r => (r.dates_found ?? []) as string[]).map(s => ({ s, d: parseDate(s) })).filter(x => x.d !== null) as { s: string; d: Date }[];
  const minDate = allDates.length ? allDates.reduce((a, b) => a.d < b.d ? a : b).d : null;
  const entries = Array.from(groups.entries()).map(([modality, rs]) => {
    const dates = rs.flatMap(r => r.dates_found ?? []).map(d => ({ s: d, d: parseDate(d) })).filter(x => x.d !== null) as any[];
    dates.sort((a, b) => a.d - b.d);
    const date = dates[0]?.s ?? null;
    const day = (() => {
      const d = parseDate(date);
      if (!d || !minDate) return null;
      return Math.round((d.getTime() - minDate.getTime()) / 86400000) + 1;
    })();
    const confirmed = rs.some(r => r.stage_of_care && !["uncertain", "n/a", "none"].includes(String(r.stage_of_care).toLowerCase()));
    return { modality, day, date, finding: rs.map(r => r.body_region).filter(Boolean)[0] ?? null, confirmed };
  });
  return { entries, consistency_score_pct: null };
}

function deriveRadiologyTimeline(catalog: any): any {
  const records: any[] = catalog.records ?? [];
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
  const events = Array.from(pairs.values()).sort((a, b) => a.date_obj.getTime() - b.date_obj.getTime()).map(p => ({
    day: minDate ? Math.round((p.date_obj.getTime() - minDate.getTime()) / 86400000) + 1 : null,
    date: p.date,
    event: p.modality,
  }));
  return { events, logical: events.length ? true : null };
}

function deriveCompleteness(s: any, catalog: any): any {
  const records: any[] = catalog.records ?? [];
  const stages = new Set<string>(records.map(r => String(r.stage_of_care ?? "").toLowerCase()).filter(Boolean));
  const types = new Set<string>(records.map(r => String(r.image_type ?? "").toLowerCase()).filter(Boolean));
  const hasStage = (needle: string) => records.length === 0 ? null : Array.from(stages).some(s => s.includes(needle));
  const hasType = (needle: string) => records.length === 0 ? null : Array.from(types).some(t => t.includes(needle));
  const c = s?.completeness ?? {};
  const merge = (modelVal: any, derived: any) => modelVal === true || modelVal === false ? modelVal : derived;
  return {
    has_pre_procedure_imaging: merge(c.has_pre_procedure_imaging, hasStage("pre")),
    has_intra_procedure_imaging: merge(c.has_intra_procedure_imaging, hasStage("intra")),
    has_post_procedure_imaging: merge(c.has_post_procedure_imaging, hasStage("post")),
    has_typed_report: merge(c.has_typed_report, hasType("typed")),
    has_handwritten_notes: merge(c.has_handwritten_notes, hasType("handwritten")),
    has_signed_stamp: merge(c.has_signed_stamp, hasType("stamp") || hasType("signature")),
    notes: c.notes ?? null,
  };
}

function computeDerived(s: any, catalog: any): any {
  return {
    ...s,
    finding_correlation: deriveFindingCorrelation(s),
    multi_image_analysis: deriveMultiImageAnalysis(catalog),
    radiology_timeline: deriveRadiologyTimeline(catalog),
    completeness: deriveCompleteness(s, catalog),
  };
}

// ── Compaction for prompt from ps3 ───────────────────────────────────────
function dedupCap(arr: any, max = 8): string[] {
  if (!Array.isArray(arr)) return [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const v of arr) {
    const s = String(v ?? "").trim();
    if (!s || seen.has(s)) continue;
    seen.add(s);
    out.push(s);
    if (out.length >= max) break;
  }
  return out;
}

function trimText(s: any, max = 1200): string {
  if (typeof s !== "string") return "";
  const collapsed = s.replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
  return collapsed.length > max ? collapsed.slice(0, max) + "…[truncated]" : collapsed;
}

function compactCatalog(catalog: any): any {
  const records = (catalog.records ?? []).map((r: any) => {
    if (r.error) return { image_index: r.image_index, source: r.source, page: r.page, error: r.error };
    return {
      image_index: r.image_index,
      source: r.source,
      page: r.page,
      pages_total: r.pages_total,
      image_type: r.image_type,
      body_region: r.body_region,
      modality_view: r.modality_view,
      stage_of_care: r.stage_of_care,
      summary: trimText(r.summary, 600),
      key_observations: dedupCap(r.key_observations),
      dates_found: dedupCap(r.dates_found),
      ocr: {
        language: r.ocr?.language,
        verbatim: trimText(r.ocr?.verbatim),
        english: trimText(r.ocr?.english),
      },
      image_quality: r.image_quality,
    };
  });
  return { claim_id: catalog.claim_id, n_images: records.length, records };
}

// ── Main API Entry Point ─────────────────────────────────────────────────

export async function processClaim(claimId: string, claimDir: string, skipSummary = false): Promise<any> {
  console.log(`[Processor] processClaim - claimId: ${claimId}, claimDir: ${claimDir}, skipSummary: ${skipSummary}`);
  const files = readdirSync(claimDir).filter(f => {
    const ext = extname(f).toLowerCase();
    return statSync(join(claimDir, f)).isFile() && (ext === ".pdf" || SUPPORTED_IMG.has(ext));
  }).sort();

  console.log(`[Processor] Found ${files.length} supported files`);

  const records: any[] = [];
  for (const f of files) {
    console.log(`[Processor] Processing file: ${f}`);
    const full = join(claimDir, f);
    const ext = extname(f).toLowerCase();
    const buffers = ext === ".pdf" ? await pdfToBuffers(full) : [readFileSync(full)];
    console.log(`[Processor]   - Total pages/images: ${buffers.length}`);

    for (let i = 0; i < buffers.length; i++) {
      const label = ext === ".pdf" ? `${f} page ${i + 1}/${buffers.length}` : f;
      console.log(`[Processor]   - Optimizing ${label}...`);
      const opt = await optimize(buffers[i]);

      console.log(`[Processor]   - Calling vision model for ${label}...`);
      const t0 = Date.now();
      const resp = await callModel(SYSTEM_OCR, USER_INSTRUCTION(label), toBase64(opt));
      const dt = Date.now() - t0;

      console.log(`[Processor]   - Vision model responded in ${dt}ms`);
      const obj = parseJson(resp.content || "{}");
      records.push({
        image_index: records.length,
        source: f,
        page: ext === ".pdf" ? i + 1 : null,
        pages_total: ext === ".pdf" ? buffers.length : null,
        ...obj,
        _usage: { prompt_tokens: resp.prompt_eval_count, completion_tokens: resp.eval_count }
      });
    }
  }

  const catalog = { claim_id: claimId, model: MODEL, n_images: records.length, records };

  let summary = null;
  if (!skipSummary) {
    console.log(`[Processor] skipSummary=false - Starting whole-claim summary synthesis...`);
    const compact = compactCatalog(catalog);
    const userText = `Claim: ${claimId}\n\nPer-image catalog (JSON below):\n\n${JSON.stringify(compact, null, 2)}`;

    console.log(`[Processor] Calling summary models (Dashboard + Reference) in parallel...`);
    const t0 = Date.now();
    const [dashResp, refResp] = await Promise.all([
      callModel(SYSTEM_DASHBOARD, userText, undefined, SUMMARY_MODEL),
      callModel(SYSTEM_REFERENCE, userText, undefined, SUMMARY_MODEL)
    ]);
    const dt = Date.now() - t0;
    console.log(`[Processor] Summary models responded in ${dt}ms`);

    console.log(`[Processor] Merging and normalizing results...`);
    const merged = { ...parseJson(refResp.content || "{}"), ...parseJson(dashResp.content || "{}") };
    const normalized = normalizeSummary(merged, claimId);

    console.log(`[Processor] Computing deterministic derivations...`);
    summary = computeDerived(normalized, catalog);
    summary.model = SUMMARY_MODEL;
    summary.generated_ms = dt;
  }

  console.log(`[Processor] processClaim complete for ${claimId}`);
  return { ...catalog, summary };
}
