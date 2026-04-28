#!/usr/bin/env python3
"""
Parallel POST load test for Klein gateway (POST /api).

Uses 64 diverse T2I prompts by default and meaningful ``output_path`` filenames
derived from each prompt (under ``KLEIN_TEST_OUTPUT_DIR`` or a repo-local default).

Usage:
  python scripts/test_klein_parallel_requests.py 10.0.0.5
  python scripts/test_klein_parallel_requests.py 127.0.0.1 --port 8080 -n 64 -j 16

Env:
  KLEIN_TEST_OUTPUT_DIR   Base directory on the **inference server** (NFS); default:
                          <flow-factory>/scripts/test_klein_server_outputs/<run_timestamp>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# 64 diverse T2I prompts (subjects, styles, lighting, composition vary)
# ---------------------------------------------------------------------------
DIVERSE_PROMPTS_64: list[str] = [
    "A red fox in fresh snow at golden hour, soft bokeh, wildlife photography",
    "Neon-lit rainy alley in Tokyo at night, reflections on wet asphalt, cinematic",
    "Watercolor still life: blue bowl, two lemons, linen cloth, soft window light",
    "Brutalist concrete stairwell, harsh side light, strong geometric shadows",
    "Bioluminescent jellyfish in deep blue ocean, ethereal glow, macro detail",
    "Sunrise over terraced rice paddies in Southeast Asia, mist, wide angle",
    "Vintage brass telescope on mahogany desk, dust motes in a sunbeam, macro",
    "Cyberpunk street food stall, holographic signage, crowded, saturated colors",
    "Minimalist white gallery wall with a single large abstract black ink stroke",
    "Ancient oak tree in foggy English moor, moody, painterly clouds",
    "Art deco ballroom chandelier, gold and glass, low angle, dramatic",
    "Hot air balloons over Cappadocia at dawn, soft pastel sky, aerial view",
    "Steaming bowl of ramen, chopsticks, nori and soft-boiled egg, overhead shot",
    "Abandoned factory hall with broken skylights, shafts of light, dust, wide",
    "Majestic peacock with tail fanned, iridescent feathers, shallow depth of field",
    "Scandinavian living room, pale wood, large window to pine forest, hygge",
    "Martian rover tracks in red sand, rocky horizon, NASA reference style",
    "Baroque violin resting on velvet, candlelight, rich chiaroscuro",
    "Kite surfers on turquoise sea, white spray, bright midday sun",
    "Stack of old hardcover books with reading glasses, cozy desk lamp",
    "Northern lights over frozen lake, stars, silhouetted pine trees",
    "Busy Moroccan spice market, woven baskets, warm saturated tones",
    "Glass skyscraper facade reflecting sunset clouds, abstract geometry",
    "Sleeping tabby cat on a sunny windowsill, potted herbs, domestic calm",
    "Steampunk brass gears and pressure gauges, intricate macro, warm metal",
    "Cherry blossoms framing a wooden bridge over a stream, spring, Japan",
    "Electric guitar close-up, worn fretboard, stage smoke, concert lighting",
    "Desert sand dunes at sunset, ripples, long shadows, empty horizon",
    "Underwater coral reef with clownfish and sea anemone, vivid colors",
    "Foggy San Francisco bay with Golden Gate Bridge partial silhouette",
    "Rustic sourdough bread on cutting board, flour dust, artisan bakery",
    "Snow leopard on rocky ledge, cold blue atmosphere, National Geographic style",
    "Parisian cafe terrace at blue hour, wicker chairs, warm interior glow",
    "Origami cranes scattered on tatami mat, soft natural side light",
    "Thunderstorm over prairie grassland, lightning bolt, dramatic sky",
    "Velvet moth with intricate wing patterns, macro, dark moody background",
    "Floating lanterns on river at night festival, bokeh, long exposure feel",
    "Crystal geode split open, purple amethyst interior, studio lighting",
    "Retro 1950s American diner interior, chrome stools, checkerboard floor",
    "Bamboo forest path, dappled green light, zen atmosphere",
    "Vintage typewriter with blank paper, desk plant, morning coffee cup",
    "Arctic iceberg arch in turquoise water, clear sky, expedition yacht small",
    "Flamenco dancer mid-twirl, red dress motion blur, dramatic spotlight",
    "Japanese koi pond from above, orange and white fish, lily pads",
    "Gothic cathedral nave, vaulted ceiling, stained glass colored light",
    "Fresh strawberries in a colander, water droplets, kitchen counter",
    "Motorcycle on coastal cliff road at sunset, ocean panorama",
    "Whimsical treehouse among autumn maple leaves, rope bridge, storybook",
    "Blacksmith forge, glowing orange metal, sparks, dark workshop",
    "Tropical waterfall in jungle, mossy rocks, long exposure silky water",
    "Woman in yellow raincoat with umbrella, puddle reflections, urban",
    "Antique world map spread on table, magnifying glass, brass compass",
    "Ski slope after fresh powder, pine trees, bright alpine sun",
    "Greenhouse interior, monstera leaves, humid light, botanical",
    "Lighthouse on stormy coast, crashing waves, dramatic clouds",
    "Colorful macarons on marble slab, pastry shop window light",
    "Silhouette of deer herd at meadow edge during lavender dusk",
    "Industrial robot arm in clean factory, blue LED accents, precision",
    "Pumpkin patch at sunset, hay bales, warm harvest palette",
    "Venice canal at night, gondola, warm lamplight on rippling water",
    "Macro of dew on spider web in grass, sunrise sparkle",
    "Tibetan prayer flags in mountain wind, snow peaks background",
    "Vintage vinyl records and turntable, warm wood shelf, cozy evening",
    "Sahara camel caravan silhouette on dune ridge, huge setting sun",
    "Modern kitchen island with fresh citrus bowl, morning sun, clean lines",
    "Fireflies in summer meadow at twilight, soft magical atmosphere",
    "Rusted shipwreck on remote beach, dramatic clouds, wide landscape",
    "Elegant swan on misty lake at dawn, mirror reflection, pastel tones",
    "Colorful hot sauce bottles on rustic wood, chili peppers, food styling",
    "Moon base habitat dome on gray regolith, starfield, hard science fiction",
    "Basketball court at night, single floodlight, wet surface reflections",
    "Stacked river stones balanced in zen garden, raked gravel patterns",
    "Hummingbird feeding at red tubular flower, frozen wing motion, macro",
    "Cozy reading nook with floor lamp, stacked books, knitted throw blanket",
]


def _slug_for_path(prompt: str, max_len: int = 44) -> str:
    """ASCII-ish slug safe for POSIX paths (no slashes)."""
    t = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.strip()).strip("_").lower()
    if not t:
        t = "prompt"
    return t[:max_len].rstrip("_") or "prompt"


def _default_output_base() -> Path:
    repo = Path(__file__).resolve().parents[1]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return repo / "scripts" / "test_klein_server_outputs" / f"run_{stamp}"


def _parse_sse_content(text: str) -> dict[str, Any] | None:
    for line in text.splitlines():
        if line.startswith("data:"):
            raw = line[len("data:") :].strip()
            if raw:
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return None
    return None


def one_request(
    base: str,
    idx: int,
    prompt: str,
    output_path: str | None,
    timeout: int,
) -> dict[str, Any]:
    """
    ``output_path`` when set is sent in JSON (Klein extension); filenames here are
    meaningful: ``t2i_{idx:02d}_{slug}.png`` under the chosen output base directory.
    """
    url = base.rstrip("/") + "/api"
    body: dict[str, Any] = {
        "context_request_id": str(uuid.uuid4()),
        "request_id": str(uuid.uuid4()),
        "prompts": prompt,
        "size": "1024*1024",
        "seed": (idx * 7919 + 42) % 1_000_000,
        "adapter_id": "flux2-klein-loadtest",
    }
    if output_path:
        body["output_path"] = output_path

    row: dict[str, Any] = {
        "idx": idx,
        "prompt": prompt[:200] + ("…" if len(prompt) > 200 else ""),
        "ok": False,
        "post_url": url,
        "request_output_path": output_path or "",
        "result_ref": "",
        "artifact_bytes": 0,
        "error": "",
    }
    try:
        r = requests.post(url, json=body, timeout=timeout)
        if r.status_code != 200:
            row["error"] = f"http {r.status_code} {r.text[:300]}"
            return row
        data = _parse_sse_content(r.text)
        if not data:
            row["error"] = "no data: line in body"
            return row
        if data.get("code") != 0:
            row["error"] = f"code={data.get('code')} msg={data.get('message')}"
            return row
        content = (data.get("data") or {}).get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            row["error"] = "empty content"
            return row
        row["result_ref"] = str(content).strip()
        if row["result_ref"].startswith("http"):
            g = requests.get(row["result_ref"], timeout=timeout)
            row["artifact_bytes"] = len(g.content or b"")
            if g.status_code != 200 or row["artifact_bytes"] < 100:
                row["error"] = f"artifact GET {g.status_code} len={row['artifact_bytes']}"
                return row
        row["ok"] = True
        return row
    except Exception as e:
        row["error"] = str(e)
        return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ip", help="Server IP (gateway external IP)")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    p.add_argument(
        "-n",
        "--requests",
        type=int,
        default=64,
        dest="n",
        help="Total requests (default 64; capped by prompt list length)",
    )
    p.add_argument("-j", "--jobs", type=int, default=16, help="Max concurrent workers")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument(
        "--no-output-path",
        action="store_true",
        help="Omit output_path from JSON (artifact URL only in response).",
    )
    args = p.parse_args()

    n = min(args.n, len(DIVERSE_PROMPTS_64))
    prompts = DIVERSE_PROMPTS_64[:n]

    out_base: Path | None = None
    output_paths: list[str | None]
    if args.no_output_path:
        output_paths = [None] * n
    else:
        raw = (os.environ.get("KLEIN_TEST_OUTPUT_DIR") or "").strip()
        out_base = Path(raw).expanduser() if raw else _default_output_base()
        out_base.mkdir(parents=True, exist_ok=True)
        output_paths = []
        for i, pr in enumerate(prompts):
            slug = _slug_for_path(pr)
            outp = str(out_base / f"t2i_{i:02d}_{slug}.png")
            output_paths.append(outp)

    base = f"http://{args.ip}:{args.port}"
    post_url = base.rstrip("/") + "/api"
    print(f"POST {post_url}  (n={n}, concurrency={args.jobs})")
    if not args.no_output_path and out_base is not None:
        print(f"  output_path base: {out_base}  (meaningful t2i_XX_<slug>.png per request)")
    if args.no_output_path:
        print("  --no-output-path: JSON without output_path")
    print()
    print(
        "Per request: result_ref = API message.content (ToolGen image_url). "
        "request_output_path = JSON output_path when enabled."
    )
    print("-" * 120)

    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {
            ex.submit(one_request, base, i, prompts[i], output_paths[i], args.timeout): i
            for i in range(n)
        }
        for fut in as_completed(futs):
            rows.append(fut.result())

    rows.sort(key=lambda r: r["idx"])
    ok = sum(1 for r in rows if r["ok"])
    fail = len(rows) - ok

    for r in rows:
        tag = "OK " if r["ok"] else "FAIL"
        outp = r.get("request_output_path") or "(none)"
        ref = r.get("result_ref") or "(empty)"
        extra = f"  artifact_bytes={r['artifact_bytes']}" if r.get("artifact_bytes") else ""
        print(f"[{tag}] req={r['idx']:04d}")
        print(f"       prompt: {r.get('prompt', '')}")
        print(f"       request_output_path: {outp}")
        print(f"       result_ref: {ref}{extra}")
        if not r["ok"] and r.get("error"):
            print(f"       error: {r['error'][:500]}")
        print()

    dt = time.perf_counter() - t0
    print("-" * 120)
    print(f"done in {dt:.1f}s  ok={ok}  fail={fail}")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
