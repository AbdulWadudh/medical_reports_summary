# PS2 Image Catalog

Per-image OCR + summary for **NHA AAE Hackathon 2026 — Problem Statement 2** (Radiological Image-Based Condition Detection & Report Correlation).

Walks every claim folder under `Claims/`, expands each PDF page-by-page, runs each image (or page) through a vision-language model, and emits:

- `image_type` — X-ray / IVP / KUB / USG / CT / MRI / Coronary Angiogram / Typed report / Stamp / …
- `body_region` — Abdomen / Chest / Coronary tree / etc.
- `modality_view` — PA / Lateral / RAO-Caudal / Transverse / N/A
- `stage_of_care` — pre / intra / post / uncertain
- `ocr.{language, verbatim, english}` — multi-script OCR with English translation (Indic languages)
- `summary`, `key_observations`, `dates_found`, `image_quality`

Output is **strictly assistive** — never diagnoses, approves, or rejects claims.

## Folder layout (this share)

```
image_catalog/
├── ps2_image_catalog.ts             # Bun version (CLI)
├── ps2_image_catalog.ipynb          # Jupyter version (interactive)
├── Claims/                          # input data
│   ├── SU007A/<CLAIM_ID>/...        # PCNL claims
│   ├── MC011A/<CLAIM_ID>/...        # PTCA claims
│   ├── MG029A/<CLAIM_ID>/...        # Acute COPD claims
│   └── SG039/<CLAIM_ID>/...         # Lap-chole claims (alias for SG039C)
├── catalogs/                        # generated outputs (created on first run)
│   └── <PACKAGE>/<CLAIM>/image_catalog.{json,md}
├── package.json                     # Bun deps
├── pyproject.toml                   # Python deps
├── .env.example                     # template — copy to .env and add your key
└── tsconfig.json
```

## One-time setup

1. **Copy `.env.example` → `.env`**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and put your Fireworks key in `FIREWORKS_API_KEY=fw_…`. Get a key from https://fireworks.ai/account/api-keys.

2. **Pick the path that matches your tooling:**

   **Bun (CLI):**
   ```bash
   bun install
   ```
   **Python (notebook):**
   ```bash
   uv sync                                    # if you use uv
   # or
   pip install pymupdf pillow requests json-repair python-dotenv jupyter ipykernel
   ```

## Running

All commands assume you're inside the `image_catalog/` folder.

### Bun (CLI)

```bash
bun ps2_image_catalog.ts                              # list packages + show config
bun ps2_image_catalog.ts SU007A                       # all 10 SU007A claims
bun ps2_image_catalog.ts SU007A <CLAIM_ID>            # one claim
bun ps2_image_catalog.ts SU007A <CLAIM_ID> --limit 3  # cap to first 3 pages (debug)
```

### Jupyter notebook

Open `ps2_image_catalog.ipynb` in VS Code → pick the project's `.venv` kernel → **Run All**. Edit `PACKAGE` / `CLAIM` in the run cell to switch claims.

Or headlessly from the shell:
```bash
uv run jupyter nbconvert --to notebook --execute ps2_image_catalog.ipynb \
    --output ps2_image_catalog.ipynb --ExecutePreprocessor.timeout=600
```

### Output

Both versions write the same two sibling files for each claim:
```
catalogs/<PACKAGE>/<CLAIM>/image_catalog.json    # structured records
catalogs/<PACKAGE>/<CLAIM>/image_catalog.md      # overview table + per-image deep section
```

## Provider switch

| Provider | When to use | Default model | Override |
|---|---|---|---|
| **Fireworks** (default) | Production / best quality | Your Fireworks `qwen3-vl-8b-instruct` deployment | `MODEL=accounts/<acct>/deployments/<id>` |
| **Ollama** (local) | Offline dev / no internet | `qwen3-vl:8b-instruct` (Q4) | `PROVIDER=ollama MODEL=qwen3-vl:8b-instruct` |

To switch providers from the CLI:
```bash
PROVIDER=ollama bun ps2_image_catalog.ts SU007A <CLAIM>
```

In the notebook, set the env vars in a cell before running `Config`:
```python
os.environ['PROVIDER'] = 'ollama'
os.environ.pop('MODEL', None)
```

## Output schema (per image)

```json
{
  "image_index": 3,
  "source": "claim_files.pdf",
  "page": 2, "pages_total": 3,
  "image_type": "Typed report",
  "body_region": "Abdomen — pelvicalyceal system",
  "modality_view": "N/A",
  "stage_of_care": "pre-procedure",
  "stage_evidence": "IVP report dated 24/03/2026 with no devices in field",
  "ocr": {
    "language": "English",
    "verbatim": "...",
    "english": "..."
  },
  "summary": "1-3 sentence summary of what this image shows.",
  "key_observations": ["...", "..."],
  "dates_found": ["24/03/2026"],
  "image_quality": { "rating": "good", "limitations": [] }
}
```
