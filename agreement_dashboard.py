#!/usr/bin/env python3
"""
agreement_dashboard.py

Fills agreement_dashboard_template.html's BUNDLE placeholder with the current
agreement_bundle.json (from agreement_analysis.py) and writes the
finished, self-contained agreement_dashboard.html, which GitHub Pages serves.

Why this is a separate tiny script (not hand-edited HTML)
-----------------------------------------------------------
The template is a normal, readable HTML/CSS/JS file with one placeholder,
`const BUNDLE = __BUNDLE_JSON__;`. Keeping the data out of the template
means re-running the whole pipeline on new/updated reports never requires
touching the dashboard's markup or chart code by hand — only the manifest
and this one substitution change.

Usage
-----
    python3 agreement_dashboard.py [--bundle out/agreement/agreement_bundle.json] \\
        [--template agreement_dashboard_template.html] [--out agreement_dashboard.html]

Relative paths are resolved from the repository root, where this script lives.
"""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bundle', default='out/agreement/agreement_bundle.json')
    ap.add_argument('--template', default='agreement_dashboard_template.html')
    ap.add_argument('--out', default='agreement_dashboard.html')
    args = ap.parse_args()

    with open(REPO_ROOT / args.bundle) as f:
        bundle = json.load(f)
    bundle_json = json.dumps(bundle, separators=(',', ':'), ensure_ascii=False)

    template = (REPO_ROOT / args.template).read_text(encoding='utf-8')
    if '__BUNDLE_JSON__' not in template:
        raise SystemExit(f"{args.template} has no __BUNDLE_JSON__ placeholder — wrong template file?")
    html = template.replace('__BUNDLE_JSON__', bundle_json)

    outpath = REPO_ROOT / args.out
    outpath.write_text(html, encoding='utf-8')
    print(f"Wrote {outpath} ({outpath.stat().st_size} bytes, bundle {len(bundle_json)} bytes)")


if __name__ == '__main__':
    main()
