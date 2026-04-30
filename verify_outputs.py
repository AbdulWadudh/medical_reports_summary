"""Verify the pipeline outputs against the PS2 reference schema.

Usage:
    uv run python verify_outputs.py
    uv run python verify_outputs.py --package MC011A
    uv run python verify_outputs.py --strict      # fail on any mismatch

Reports:
  - schema match: do all expected sections / fields appear in each case?
  - coverage:     how many fields are populated vs left as the default placeholder?
  - extras:       any unexpected sections / fields the model emitted

Reference shape lives in:  PS2_Output_Guidelines/Output Format/<pkg>.json
Actual outputs in:         outputs/package_json_outputs/<pkg>_outputs.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent
REFERENCE_DIR = Path("c:/Users/KSeventy9/Downloads/PS2_Output_Guidelines/Output Format")
OUTPUTS_DIR = REPO_ROOT / "outputs" / "package_json_outputs"

# The reference files use slightly different naming for each package
REFERENCE_FILES = {
    "MC011A": "MC011A_Json.Json",
    "MG029A": "MG029A_Json.Json",
    "SG039C": "SG039C_json.json",
    "SU007A": "SU007A_json.json",
}

# Values we consider "the model didn't fill this in"
DEFAULT_VALUES = {
    "",
    "Not assessed / Not seen",
    "Absent",
    "Good/Poor/Very Good",  # appears verbatim in the reference templates
    None,
}


@dataclass
class CaseReport:
    case_no: str
    package: str
    populated: int = 0
    total: int = 0
    missing_sections: list[str] = field(default_factory=list)
    missing_fields:   list[str] = field(default_factory=list)
    empty_fields:     list[str] = field(default_factory=list)
    extra_sections:   list[str] = field(default_factory=list)
    extra_fields:     list[str] = field(default_factory=list)

    @property
    def coverage_pct(self) -> float:
        return (100.0 * self.populated / self.total) if self.total else 0.0


def expected_shape(reference: list[dict]) -> dict[str, list[str]]:
    """Take the first case in a reference file and return {section: [field,...]}."""
    if not reference:
        return {}
    sample = reference[0]
    shape: dict[str, list[str]] = {}
    for section, body in sample.items():
        if isinstance(body, dict):
            shape[section] = list(body.keys())
        else:
            # Trailing fields like "Summary PDF" are top-level scalars
            shape[section] = []
    return shape


def is_populated(value: Any) -> bool:
    if value in DEFAULT_VALUES:
        return False
    if isinstance(value, str) and value.strip() in DEFAULT_VALUES:
        return False
    return True


def verify_case(actual: dict, shape: dict[str, list[str]], pkg: str) -> CaseReport:
    case_no = ""
    # Both reference templates use a top-level "claims details" / "Claim details" section
    for section, body in actual.items():
        if section.lower().startswith("claim") and isinstance(body, dict):
            case_no = body.get("Claim No.", "") or body.get("Claim No. ", "")
            break

    rep = CaseReport(case_no=case_no, package=pkg)

    for section, fields in shape.items():
        if section not in actual:
            rep.missing_sections.append(section)
            rep.total += max(1, len(fields))
            continue
        body = actual[section]

        if not fields:
            rep.total += 1
            if is_populated(body):
                rep.populated += 1
            else:
                rep.empty_fields.append(section)
            continue

        if not isinstance(body, dict):
            rep.missing_sections.append(f"{section} (expected dict, got {type(body).__name__})")
            rep.total += len(fields)
            continue

        for fname in fields:
            rep.total += 1
            if fname not in body:
                rep.missing_fields.append(f"{section} -> {fname}")
                continue
            v = body[fname]
            if is_populated(v):
                rep.populated += 1
            else:
                rep.empty_fields.append(f"{section} -> {fname}")

    # Extras
    for section, body in actual.items():
        if section not in shape:
            rep.extra_sections.append(section)
            continue
        if not isinstance(body, dict):
            continue
        expected_field_set = set(shape[section])
        for fname in body.keys():
            if fname not in expected_field_set:
                rep.extra_fields.append(f"{section} -> {fname}")

    return rep


def verify_package(pkg: str, ref_path: Path, actual_path: Path) -> tuple[list[CaseReport], int]:
    if not ref_path.exists():
        print(f"  [skip] reference missing: {ref_path}")
        return [], 0
    if not actual_path.exists():
        print(f"  [skip] actual output missing: {actual_path}")
        return [], 0

    reference = json.loads(ref_path.read_text(encoding="utf-8"))
    actual    = json.loads(actual_path.read_text(encoding="utf-8"))
    shape = expected_shape(reference)

    reports = [verify_case(case, shape, pkg) for case in actual]

    print(f"  reference cases: {len(reference)}  ·  actual cases: {len(actual)}")
    print(f"  expected sections: {len(shape)} ({', '.join(shape.keys())})")
    print()

    severe = 0
    for r in reports:
        bar = "#" * int(r.coverage_pct / 5) + "." * (20 - int(r.coverage_pct / 5))
        print(f"  case {r.case_no or '?':<10} [{bar}] {r.coverage_pct:5.1f}%  "
              f"({r.populated}/{r.total} fields populated)")
        if r.missing_sections:
            print(f"      missing sections : {r.missing_sections}")
            severe += len(r.missing_sections)
        if r.missing_fields:
            print(f"      missing fields   : {r.missing_fields[:3]}{'...' if len(r.missing_fields) > 3 else ''}")
            severe += len(r.missing_fields)
        if r.extra_sections:
            print(f"      extra sections   : {r.extra_sections}")
        if r.extra_fields and len(r.extra_fields) <= 3:
            print(f"      extra fields     : {r.extra_fields}")
        if r.empty_fields and len(r.empty_fields) <= 5:
            print(f"      empty fields     : {r.empty_fields[:5]}")
        elif r.empty_fields:
            print(f"      {len(r.empty_fields)} empty fields (model didn't populate)")

    return reports, severe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", choices=list(REFERENCE_FILES.keys()),
                    help="check only one package (default: all)")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any case has missing sections/fields")
    args = ap.parse_args()

    pkgs = [args.package] if args.package else list(REFERENCE_FILES.keys())

    grand_severe = 0
    grand_reports: list[CaseReport] = []
    for pkg in pkgs:
        ref_path    = REFERENCE_DIR / REFERENCE_FILES[pkg]
        actual_path = OUTPUTS_DIR / f"{pkg}_outputs.json"
        print(f"== {pkg} ==")
        print(f"  ref:    {ref_path}")
        print(f"  actual: {actual_path}")
        reports, severe = verify_package(pkg, ref_path, actual_path)
        grand_reports.extend(reports)
        grand_severe += severe
        print()

    print("=" * 60)
    if grand_reports:
        avg_cov = sum(r.coverage_pct for r in grand_reports) / len(grand_reports)
        print(f"summary: {len(grand_reports)} cases, avg coverage = {avg_cov:.1f}%, "
              f"schema problems = {grand_severe}")
    else:
        print("summary: no cases verified (run the pipeline first)")

    if args.strict and grand_severe:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
