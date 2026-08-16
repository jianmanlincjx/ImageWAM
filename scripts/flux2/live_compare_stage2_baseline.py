#!/usr/bin/env python3
"""Live Stage2 vs baseline action losses.

Train metric is flow-matching `train/loss_action`.
Infer metric is `eval/action_l2` (denoise a chunk, L2 vs GT).
Late gaps are shown with a tight y-limit plus Stage2-minus-baseline deltas.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import time
from pathlib import Path

import plotext as plt

DEFAULT_BASE = (
    "/data2/JM/Code/ImageWAM/runs/libero_flux2_klein_4b_base_imagewam/"
    "2026-08-13_00-24-52/wandb/offline-run-20260813_002545-enes553m/run-enes553m.wandb"
)
DEFAULT_STAGE2 = (
    "/data2/JM/Code/ImageWAM/runs/libero_flux2_klein_4b_goal_prior_stage2/"
    "2026-08-15_09-58-20/wandb/offline-run-20260815_095917-la38t2j5/run-la38t2j5.wandb"
)

STEP_RE = re.compile(r"_step[\x00-\x1f\x80-\xff]{0,8}(\d{1,5})")


def _metric_re(key: str) -> re.Pattern[str]:
    return re.compile(re.escape(key) + r"[\x00-\x1f\x80-\xff]{0,8}([0-9]+\.[0-9]+)")


def parse_metric(path: Path, key: str, step_mod: int, step_max: int) -> dict[int, float]:
    text = path.read_bytes().decode("latin1", errors="ignore")
    hits = [(m.start(), float(m.group(1))) for m in _metric_re(key).finditer(text)]
    steps = [(m.start(), int(m.group(1))) for m in STEP_RE.finditer(text)]
    by: dict[int, float] = {}
    j = 0
    for pos, step in steps:
        while j + 1 < len(hits) and hits[j + 1][0] < pos:
            j += 1
        if not (step_mod <= step <= step_max and step % step_mod == 0):
            continue
        if hits and hits[j][0] < pos:
            by[step] = hits[j][1]
    return by


def aligned(a: dict[int, float], b: dict[int, float]) -> tuple[list[int], list[float], list[float]]:
    steps = sorted(set(a) & set(b))
    return steps, [a[s] for s in steps], [b[s] for s in steps]


def late_window(steps: list[int], warmup_end: int = 2000, min_span: int = 2500) -> int:
    if not steps:
        return warmup_end
    last = steps[-1]
    start = max(warmup_end, last - max(min_span, last // 2))
    if last - start < 800:
        start = max(warmup_end, last - 800)
    return start


def tight_ylim(values: list[float], pad: float = 0.18) -> tuple[float, float] | None:
    xs = [v for v in values if v is not None and math.isfinite(v)]
    if len(xs) < 4:
        return None
    xs.sort()
    lo = xs[max(0, int(0.05 * (len(xs) - 1)))]
    hi = xs[min(len(xs) - 1, int(0.95 * (len(xs) - 1)))]
    if hi <= lo:
        hi = lo + max(abs(lo) * 0.02, 1e-4)
    span = hi - lo
    return lo - pad * span, hi + pad * span


def downsample(steps: list[int], *series: list[float], max_points: int = 80):
    if len(steps) <= max_points:
        return (steps,) + series
    stride = math.ceil(len(steps) / max_points)
    idx = list(range(0, len(steps), stride))
    if idx[-1] != len(steps) - 1:
        idx.append(len(steps) - 1)
    return tuple([seq[i] for i in idx] for seq in (steps,) + series)


def lr_at(step: int, warmup: int, total: int = 34720, peak: float = 1e-4, eta_min: float = 1e-6) -> float:
    if step < warmup:
        return peak * max(step, 1) / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return eta_min + (peak - eta_min) * (1 + math.cos(math.pi * min(t, 1))) / 2


def _panel(title: str, xs, ys_list, labels, colors, *, log=False, ylim=None, xlabel="step"):
    plt.title(title)
    plt.xlabel(xlabel)
    for y, label, color in zip(ys_list, labels, colors):
        plt.plot(xs, y, label=label, color=color)
    if log:
        plt.yscale("log")
    if ylim is not None:
        plt.ylim(*ylim)


def render(base_path: Path, stage2_path: Path) -> None:
    cols, rows = shutil.get_terminal_size((160, 48))
    plt.clear_terminal()
    plt.clear_figure()
    plt.theme("clear")
    plt.plotsize(max(cols - 1, 80), max(rows - 1, 32))
    plt.clf()

    # Train: flow-matching action loss. Infer: denoise a chunk and L2 vs GT action.
    base_tr = parse_metric(base_path, "train/loss_action", 10, 34720)
    s2_tr = parse_metric(stage2_path, "train/loss_action", 10, 34720)
    base_ev = parse_metric(base_path, "eval/action_l2", 100, 34720)
    s2_ev = parse_metric(stage2_path, "eval/action_l2", 100, 34720)

    tr_steps, tr_base, tr_s2 = aligned(base_tr, s2_tr)
    ev_steps, ev_base, ev_s2 = aligned(base_ev, s2_ev)
    last = tr_steps[-1] if tr_steps else (ev_steps[-1] if ev_steps else 0)
    zoom_from = late_window(tr_steps or ev_steps)

    def slice_from(steps, *series, start):
        keep = [i for i, s in enumerate(steps) if s >= start]
        return tuple([seq[i] for i in keep] for seq in (steps,) + series)

    plt.subplots(2, 2)

    plt.subplot(1, 1)
    if tr_steps:
        xs, yb, ys = downsample(tr_steps, tr_base, tr_s2, max_points=90)
        _panel(
            "train/loss_action  (FM)  log-y  full",
            xs,
            [yb, ys],
            ["baseline", "stage2"],
            ["cyan", "orange"],
            log=True,
        )

    plt.subplot(1, 2)
    if tr_steps:
        xs, yb, ys = slice_from(tr_steps, tr_base, tr_s2, start=zoom_from)
        xs, yb, ys = downsample(xs, yb, ys, max_points=90)
        ylim = tight_ylim(yb + ys, pad=0.12)
        _panel(
            f"train/loss_action  zoom ≥{zoom_from}  tight y",
            xs,
            [yb, ys],
            ["baseline", "stage2"],
            ["cyan", "orange"],
            ylim=ylim,
        )

    plt.subplot(2, 1)
    if ev_steps:
        xs, yb, ys = slice_from(ev_steps, ev_base, ev_s2, start=zoom_from)
        ylim = tight_ylim(yb + ys, pad=0.12)
        _panel(
            f"eval/action_l2  (infer)  zoom ≥{zoom_from}  tight y",
            xs,
            [yb, ys],
            ["baseline", "stage2"],
            ["cyan", "orange"],
            ylim=ylim,
        )

    plt.subplot(2, 2)
    delta_tr: list[float] = []
    delta_ev: list[float] = []
    xs_d: list[int] = []
    if tr_steps:
        xs_t, yb_t, ys_t = slice_from(tr_steps, tr_base, tr_s2, start=zoom_from)
        xs_t, yb_t, ys_t = downsample(xs_t, yb_t, ys_t, max_points=80)
        delta_tr = [a - b for a, b in zip(ys_t, yb_t)]
        xs_d = xs_t
        plt.plot(xs_t, delta_tr, label="Δ train FM", color="orange")
    if ev_steps:
        xs_e, yb_e, ys_e = slice_from(ev_steps, ev_base, ev_s2, start=zoom_from)
        delta_ev = [a - b for a, b in zip(ys_e, yb_e)]
        plt.plot(xs_e, delta_ev, label="Δ infer L2", color="red")
        if not xs_d:
            xs_d = xs_e
    if xs_d:
        plt.plot(xs_d, [0.0] * len(xs_d), label="zero", color="gray")
        ylim = tight_ylim(delta_tr + delta_ev + [0.0], pad=0.2)
        plt.title("Δ = stage2 − baseline   (>0 Stage2 worse)")
        plt.xlabel("step")
        if ylim is not None:
            plt.ylim(*ylim)

    gap_tr = (tr_s2[-1] - tr_base[-1]) if tr_steps else float("nan")
    gap_ev = (ev_s2[-1] - ev_base[-1]) if ev_steps else float("nan")
    last_tr = tr_s2[-1] if tr_steps else float("nan")
    last_ev = ev_s2[-1] if ev_steps else float("nan")
    def _side(delta: float) -> str:
        if delta > 1e-5:
            return "s2 worse"
        if delta < -1e-5:
            return "s2 better"
        return "tie"

    header = (
        f"Δ=s2-base  >0 WORSE (higher), not better   "
        f"step {last}/34720  "
        f"train FM Δ={gap_tr:+.4f} ({_side(gap_tr)})  "
        f"infer L2 Δ={gap_ev:+.4f} ({_side(gap_ev)})  "
        f"zoom≥{zoom_from}  Ctrl-C quit"
    )
    plt.show()
    print(header)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--stage2", default=DEFAULT_STAGE2)
    parser.add_argument("--refresh", type=float, default=20.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    base_path = Path(args.base)
    stage2_path = Path(args.stage2)
    while True:
        try:
            if not stage2_path.exists() or not base_path.exists():
                print("waiting for wandb files...")
            else:
                render(base_path, stage2_path)
        except Exception as exc:  # noqa: BLE001 — keep the live view up
            plt.clear_terminal()
            print(f"parse/render error: {exc!r}")
        if args.once:
            break
        time.sleep(args.refresh)


if __name__ == "__main__":
    os.environ.setdefault("TERM", "xterm-256color")
    main()
