#!/usr/bin/env python3
"""Write Stage2 vs baseline action-loss PNG with explicit Δ meaning."""
from __future__ import annotations

import argparse
import math
import os
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DEFAULT_BASE = (
    "/data2/JM/Code/ImageWAM/runs/libero_flux2_klein_4b_base_imagewam/"
    "2026-08-13_00-24-52/wandb/offline-run-20260813_002545-enes553m/run-enes553m.wandb"
)
DEFAULT_STAGE2 = (
    "/data2/JM/Code/ImageWAM/runs/libero_flux2_klein_4b_goal_prior_stage2/"
    "2026-08-15_09-58-20/wandb/offline-run-20260815_095917-la38t2j5/run-la38t2j5.wandb"
)
DEFAULT_OUT = (
    "/data2/JM/Code/ImageWAM/runs/libero_flux2_klein_4b_goal_prior_stage2/"
    "2026-08-15_09-58-20/compare_action_loss.png"
)
FONT_CANDIDATES = [
    Path("/tmp/wqymicrohei.ttf"),
    Path(
        "/data1/yejianheng/backup/中大本科/活动与组织/电赛/嘉立创杯/"
        "简易电路特性测试仪/circuit_measure/TFT_project/Project/output/truefont/wqymicrohei.ttf"
    ),
]

STEP_RE = re.compile(r"_step[\x00-\x1f\x80-\xff]{0,8}(\d{1,5})")
C_BASE = "#2563eb"
C_S2 = "#ea580c"
C_ZERO = "#9ca3af"
C_WORSE = "#b91c1c"
C_BETTER = "#15803d"


def setup_font() -> str:
    for src in FONT_CANDIDATES:
        if not src.exists():
            continue
        dst = Path("/tmp/wqymicrohei.ttf")
        if src != dst:
            dst.write_bytes(src.read_bytes())
        fm.fontManager.addfont(str(dst))
        name = fm.FontProperties(fname=str(dst)).get_name()
        plt.rcParams["font.family"] = name
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["axes.formatter.use_mathtext"] = False
        return name
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


def parse_metric(path: Path, key: str, step_mod: int, step_max: int = 34720) -> dict[int, float]:
    text = path.read_bytes().decode("latin1", errors="ignore")
    pat = re.compile(re.escape(key) + r"[\x00-\x1f\x80-\xff]{0,8}([0-9]+\.[0-9]+)")
    hits = [(m.start(), float(m.group(1))) for m in pat.finditer(text)]
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


def aligned(a: dict[int, float], b: dict[int, float]):
    ss = sorted(set(a) & set(b))
    return ss, [a[s] for s in ss], [b[s] for s in ss]


def downsample(xs, *ys, n=200):
    if len(xs) <= n:
        return (xs,) + ys
    stride = max(1, math.ceil(len(xs) / n))
    idx = list(range(0, len(xs), stride))
    if idx[-1] != len(xs) - 1:
        idx.append(len(xs) - 1)
    return tuple([seq[i] for i in idx] for seq in (xs,) + ys)


def slice_from(steps, *series, start):
    keep = [i for i, s in enumerate(steps) if s >= start]
    return tuple([seq[i] for i in keep] for seq in (steps,) + series)


def window_mean_delta(steps, y_s2, y_base, last_n_steps: int = 1000) -> float | None:
    if not steps:
        return None
    lo = steps[-1] - last_n_steps
    ds = [a - b for s, a, b in zip(steps, y_s2, y_base) if s >= lo]
    return sum(ds) / len(ds) if ds else None


def verdict(delta: float) -> tuple[str, str]:
    if delta > 1e-5:
        return "Stage2 更高 = 更差（不是优于）", C_WORSE
    if delta < -1e-5:
        return "Stage2 更低 = 更好", C_BETTER
    return "基本打平", "#334155"


def draw_callout(fig, x0, y0, w, h, title, s2, base, delta, mean_delta, face):
    ax = fig.add_axes([x0, y0, w, h])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.04),
            0.98,
            0.92,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.6,
            edgecolor=face,
            facecolor="#fff7ed" if face == C_WORSE else "#f0fdf4",
        )
    )
    ax.text(0.05, 0.78, title, fontsize=11, fontweight="semibold", color="#0f172a", va="center")
    ax.text(
        0.05,
        0.48,
        f"Stage2  {s2:.4f}     baseline  {base:.4f}",
        fontsize=12,
        color="#1e293b",
        va="center",
        family="DejaVu Sans",
    )
    mean_txt = f"    近1000 step 均值 Δ={mean_delta:+.4f}" if mean_delta is not None else ""
    ax.text(
        0.05,
        0.22,
        f"Δ = Stage2 - baseline = {delta:+.4f}{mean_txt}",
        fontsize=12,
        fontweight="bold",
        color=face,
        va="center",
    )
    ax.text(0.72, 0.78, verdict(delta)[0], fontsize=11, fontweight="bold", color=face, va="center", ha="center")


def render(base_path: Path, stage2_path: Path, out_path: Path) -> dict:
    setup_font()
    tr_s, tr_b, tr_2 = aligned(
        parse_metric(base_path, "train/loss_action", 10),
        parse_metric(stage2_path, "train/loss_action", 10),
    )
    ev_s, ev_b, ev_2 = aligned(
        parse_metric(base_path, "eval/action_l2", 100),
        parse_metric(stage2_path, "eval/action_l2", 100),
    )
    last = tr_s[-1]
    zoom = max(2000, last - max(2500, last // 2))
    gap_tr = tr_2[-1] - tr_b[-1]
    gap_ev = ev_2[-1] - ev_b[-1]
    mean_tr = window_mean_delta(tr_s, tr_2, tr_b)
    mean_ev = window_mean_delta(ev_s, ev_2, ev_b)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(13.2, 9.2), dpi=140)
    fig.suptitle(
        f"Stage2 vs baseline · 差值定义：Δ = Stage2 - baseline    "
        f"Δ>0 更差（数值更高）    Δ<0 更好    · 当前 step {last}/34720",
        fontsize=13,
        fontweight="semibold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        "这不是「正数=我们更好」。loss / L2 都是越低越好，所以 +Δ 表示 Stage2 落后。",
        ha="center",
        fontsize=10.5,
        color=C_WORSE,
        fontweight="semibold",
    )

    draw_callout(
        fig,
        0.04,
        0.84,
        0.45,
        0.09,
        "训练 loss    train/loss_action（flow matching）",
        tr_2[-1],
        tr_b[-1],
        gap_tr,
        mean_tr,
        verdict(gap_tr)[1],
    )
    draw_callout(
        fig,
        0.51,
        0.84,
        0.45,
        0.09,
        "推理指标    eval/action_l2（去噪动作 vs GT）",
        ev_2[-1],
        ev_b[-1],
        gap_ev,
        mean_ev,
        verdict(gap_ev)[1],
    )

    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.07, hspace=0.32, wspace=0.22)

    ax = axes[0, 0]
    xs, yb, y2 = downsample(tr_s, tr_b, tr_2, n=180)
    ax.plot(xs, yb, color=C_BASE, lw=1.4, label="baseline")
    ax.plot(xs, y2, color=C_S2, lw=1.4, label="stage2")
    ax.set_yscale("log")
    ax.set_title("训练 loss 全程（log y）")
    ax.set_xlabel("step")
    ax.set_ylabel("train/loss_action")
    ax.legend(frameon=False, loc="upper right")
    ax.axvline(5000, color="#94a3b8", ls="--", lw=0.8, alpha=0.8)
    ax.text(5000, ax.get_ylim()[1], "  Stage2 warmup 结束", va="top", fontsize=8, color="#64748b")
    ax.text(
        0.98,
        0.08,
        f"最新 Δ={gap_tr:+.4f}",
        transform=ax.transAxes,
        ha="right",
        color=verdict(gap_tr)[1],
        fontsize=10,
        fontweight="bold",
    )

    ax = axes[0, 1]
    xs, yb, y2 = slice_from(tr_s, tr_b, tr_2, start=zoom)
    xs, yb, y2 = downsample(xs, yb, y2, n=180)
    ax.plot(xs, yb, color=C_BASE, lw=1.4, label="baseline")
    ax.plot(xs, y2, color=C_S2, lw=1.4, label="stage2")
    lo, hi = min(yb + y2), max(yb + y2)
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(f"训练 loss 后期放大  ≥{zoom}")
    ax.set_xlabel("step")
    ax.set_ylabel("train/loss_action")
    ax.legend(frameon=False)
    ax.text(
        0.98,
        0.92,
        f"最新 Δ={gap_tr:+.4f}  近1000步均值 Δ={mean_tr:+.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=verdict(gap_tr)[1],
        fontsize=9.5,
        fontweight="bold",
    )

    ax = axes[1, 0]
    xs, yb, y2 = slice_from(ev_s, ev_b, ev_2, start=zoom)
    ax.plot(xs, yb, color=C_BASE, lw=1.6, marker="o", ms=3, label="baseline")
    ax.plot(xs, y2, color=C_S2, lw=1.6, marker="o", ms=3, label="stage2")
    lo, hi = min(yb + y2), max(yb + y2)
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(f"推理指标  eval/action_l2  后期放大  ≥{zoom}")
    ax.set_xlabel("step")
    ax.set_ylabel("action L2")
    ax.legend(frameon=False)
    ax.text(
        0.98,
        0.92,
        f"最新 Δ={gap_ev:+.4f}  近1000步均值 Δ={mean_ev:+.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=verdict(gap_ev)[1],
        fontsize=9.5,
        fontweight="bold",
    )

    ax = axes[1, 1]
    xs_t, yb_t, y2_t = slice_from(tr_s, tr_b, tr_2, start=zoom)
    xs_t, yb_t, y2_t = downsample(xs_t, yb_t, y2_t, n=160)
    dtr = [a - b for a, b in zip(y2_t, yb_t)]
    xs_e, yb_e, y2_e = slice_from(ev_s, ev_b, ev_2, start=zoom)
    dev = [a - b for a, b in zip(y2_e, yb_e)]
    ax.axhline(0, color=C_ZERO, lw=1.2)
    ax.plot(xs_t, dtr, color=C_S2, lw=1.3, label="Δ 训练 loss")
    ax.plot(xs_e, dev, color=C_WORSE, lw=1.6, marker="o", ms=3, label="Δ 推理 L2")
    vals = dtr + dev + [0.0]
    span = max(vals) - min(vals)
    ax.set_ylim(min(vals) - 0.15 * span, max(vals) + 0.15 * span)
    ax.set_title("差值图    零线以上 = Stage2 更差")
    ax.set_xlabel("step")
    ax.set_ylabel("Δ = Stage2 - baseline")
    ax.legend(frameon=False)
    ax.text(
        0.02,
        0.96,
        "零线以上：Stage2 落后\n零线以下：Stage2 领先",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="#334155",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.png")
    fig.savefig(tmp, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(out_path)
    return {
        "step": last,
        "train_s2": tr_2[-1],
        "train_base": tr_b[-1],
        "train_delta": gap_tr,
        "train_mean_delta": mean_tr,
        "eval_s2": ev_2[-1],
        "eval_base": ev_b[-1],
        "eval_delta": gap_ev,
        "eval_mean_delta": mean_ev,
        "out": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--stage2", default=DEFAULT_STAGE2)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--refresh", type=float, default=0.0)
    args = parser.parse_args()
    while True:
        info = render(Path(args.base), Path(args.stage2), Path(args.out))
        print(
            f"step {info['step']}  "
            f"train Δ={info['train_delta']:+.4f} (mean {info['train_mean_delta']:+.4f})  "
            f"infer Δ={info['eval_delta']:+.4f} (mean {info['eval_mean_delta']:+.4f})  "
            f"-> {info['out']}",
            flush=True,
        )
        if args.refresh <= 0:
            break
        time.sleep(args.refresh)


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    main()
