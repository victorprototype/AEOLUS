"""Build a single self-contained ``index.html`` from the dataset ``.npz``.

The page inlines the CSS, the JavaScript and the sampled positions, so it works from any static host
(GitHub Pages from a branch, a plain file server, or double-clicked from disk). Plotly and the web fonts
still load from their CDNs.

Usage:
    python -m aeolus.build_site                       # uses data/hydrogen_v0.npz
    python -m aeolus.build_site --url https://github.com/<user>/<repo>/releases/download/v0.1.0/hydrogen_v0.npz
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.request

import numpy as np

WEB = os.path.join(os.path.dirname(__file__), "..", "..", "web")


def positions_json(npz_path: str, decimals: int = 3) -> str:
    z = np.load(npz_path)
    pos, step = z["walker_positions"], z["walker_checkpoint_step"]
    steps = [int(s) for s in np.unique(step)]
    snaps = [np.round(pos[step == s], decimals).tolist() for s in steps]
    return json.dumps({"steps": steps, "snaps": snaps}, separators=(",", ":"))


def build(npz_path: str, out_path: str, web_dir: str = WEB) -> int:
    read = lambda *p: open(os.path.join(web_dir, *p), encoding="utf-8").read()
    html = read("index.html")
    css, js = read("css", "style.css"), read("js", "app.js")

    link = '<link rel="stylesheet" href="css/style.css">'
    script = '<script src="js/app.js"></script>'
    if link not in html or script not in html:
        raise RuntimeError("web/index.html no longer has the expected stylesheet/script tags")

    html = html.replace(link, f"<style>\n{css}\n</style>")
    # "</" inside inline script text could close the tag early, so escape it in the data.
    data = positions_json(npz_path).replace("</", "<\\/")
    html = html.replace(script, f"<script>window.AEOLUS_DATA = {data};</script>\n<script>\n{js}\n</script>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return len(html)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/hydrogen_v0.npz", help="local dataset file")
    ap.add_argument("--url", default=None, help="download the dataset from this URL instead (e.g. a release asset)")
    ap.add_argument("--out", default="index.html")
    a = ap.parse_args()

    npz = a.npz
    if a.url:
        npz = os.path.join(tempfile.mkdtemp(), "dataset.npz")
        urllib.request.urlretrieve(a.url, npz)
        print(f"downloaded {a.url}")
    size = build(npz, a.out)
    print(f"wrote {a.out} ({size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
