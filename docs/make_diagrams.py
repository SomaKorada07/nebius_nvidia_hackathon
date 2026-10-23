#!/usr/bin/env python3
"""Draw the architecture diagrams used in reflex_model.md.

Usage:
    python docs/make_diagrams.py

Writes docs/figures/arch_v1.png, arch_v2.png, arch_v3.png and training_v3.png. Every diagram uses
the same layout, and the part that is new in that version is drawn in its accent colour.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
SURFACE, INK, INK2, EDGE, FILL, FROZEN = "#fcfcfb", "#0b0b0b", "#52514e", "#c3c2b7", "#f3f2ee", "#e6e5e0"
ACCENT = {"v1": ("#6b4fd8", "#e7e2fa"), "v2": ("#2a78d6", "#dbe8f8"), "v3": ("#1baf7a", "#d5f1e6")}


def canvas(width=13.0, height=5.4, xmax=124, ymax=52):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, new=None, frozen=False, size=9, weight="normal", dashed=False, align="center"):
    edge, fill = ACCENT[new] if new else (EDGE, FROZEN if frozen else FILL)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", linewidth=2 if new else 1.2,
                                edgecolor=edge, facecolor=fill, linestyle="--" if dashed else "-", zorder=2))
    tx = x + w / 2 if align == "center" else x + 1.2
    ax.text(tx, y + h / 2, text, ha=align, va="center", fontsize=size, color=INK, fontweight=weight, zorder=3,
            linespacing=1.35)


def stack(ax, x, y, w, h, text, new=None, size=9):
    """Three offset cards: one copy of the same network per option."""
    for k in (2, 1):
        box(ax, x + 1.2 * k, y + 1.2 * k, w, h, "", new=new)
    box(ax, x, y, w, h, text, new=new, size=size)


def arrow(ax, start, end, color=INK2, dashed=False, label=None, label_offset=(0, 1.2), rad=0.0):
    ax.annotate("", xy=end, xytext=start, zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, linestyle="--" if dashed else "-",
                                shrinkA=2, shrinkB=2, connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (start[0] + end[0]) / 2 + label_offset[0], (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=7.5, color=color, style="italic", zorder=3)


def title(ax, version, heading, subtitle):
    ax.text(0.5, 50.5, f"Reflex {version}", fontsize=14, fontweight="bold", color=ACCENT[version][0], va="top")
    ax.text(16, 50.5, heading, fontsize=12, fontweight="bold", color=INK, va="top")
    ax.text(0.5, 46.3, subtitle, fontsize=9, color=INK2, va="top")


def legend(ax, version, text, y=0.6):
    """Key along the bottom edge, clear of the title and the diagram."""
    edge, fill = ACCENT[version]
    ax.add_patch(FancyBboxPatch((1, y), 3, 2, boxstyle="round,pad=0.2", facecolor=fill, edgecolor=edge, lw=2))
    ax.text(5.5, y + 1, text, fontsize=8.5, color=INK2, va="center")
    ax.add_patch(FancyBboxPatch((32, y), 3, 2, boxstyle="round,pad=0.2", facecolor=FROZEN, edgecolor=EDGE, lw=1.2))
    ax.text(36.5, y + 1, "frozen (never trained)", fontsize=8.5, color=INK2, va="center")


def architecture(version: str) -> None:
    fig, ax = canvas(13.2, 5.6, 128, 52)
    v1, v3 = version == "v1", version == "v3"
    headings = {
        "v1": ("the order-free set head", "Every option is scored by the same network and sees the list through an average, "
                                          "so shuffling the options cannot change the answer."),
        "v2": ("options get a description", "Same model as v1. Only the option text changes: "
                                            "\"name: description\" instead of the bare label name."),
        "v3": ("NONE decided from the option scores", "Same option scorer as v2. NONE_OF_THESE is now decided by a tiny head "
                                                      "that only looks at how well the options scored."),
    }
    title(ax, version, *headings[version])
    legend(ax, version, {"v1": "what makes it order-free", "v2": "new in v2", "v3": "new in v3"}[version])

    # inputs
    box(ax, 1, 31, 21, 8, "Question\n\"my card hasn't arrived\"", size=9)
    options = ("card_arrival\nlost_or_stolen_card\ncard_linking" if v1 else
               "card_arrival: Ask when a card\nthat was ordered will arrive.\n\nlost_or_stolen_card: Report\na card as lost or stolen.\n…")
    box(ax, 1, 6, 21, 20, "Options (any order)\n\n" + options, new="v2" if version == "v2" else None,
        size=8.5 if v1 else 7.9)

    # frozen encoder
    box(ax, 27, 6, 11, 33, "Nemotron\nembedding\nmodel\n\n2048-d\nvectors", frozen=True, size=9)
    arrow(ax, (22.4, 35), (26.6, 35))
    arrow(ax, (22.4, 16), (26.6, 16))

    # per-option matcher and list summary
    stack(ax, 44, 11, 18, 9, "Match each option\nφ(question, optionᵢ)", new="v1" if v1 else None)
    ax.text(53, 8.4, "same network for every option", ha="center", fontsize=7.5, color=INK2, style="italic")
    arrow(ax, (38.4, 15.5), (43.6, 15.5), label="cᵢ", label_offset=(0, 0.4))
    arrow(ax, (38.4, 33), (52, 23.4), label="s", label_offset=(1.5, 0.6))
    box(ax, 70, 22, 18, 7, "List summary\naverage of all options", new="v1" if v1 else None, size=8.3)
    arrow(ax, (63.8, 18.5), (69.6, 24))
    box(ax, 70, 7, 18, 9, "Option scores\nzᵢ = ρ(matchᵢ, summary)", size=8)
    arrow(ax, (63.8, 13.5), (69.6, 11.5))
    arrow(ax, (79, 21.6), (79, 16.4))

    # NONE_OF_THESE score
    if not v3:
        box(ax, 70, 34, 18, 7, "NONE score\nψ(question, summary)", size=8.3)
        arrow(ax, (79, 29.4), (79, 33.6))
        arrow(ax, (38.4, 37.5), (69.6, 37.5), label="question vector s", label_offset=(0, 0.5))
        none_out, none_rad = (88.4, 37.5), -0.25
    else:
        box(ax, 92, 21, 16, 11, "6 numbers about\nthe scores\n\nbest · gap to 2nd\nspread · evidence\nbest cosine · K",
            new="v3", size=7.6)
        arrow(ax, (88.4, 12), (94, 20.6), color=ACCENT["v3"][0], dashed=True, label="no gradient\nback",
              label_offset=(-5.2, 0.7))
        box(ax, 92, 35, 16, 7, "Tiny NONE head\n6 → 16 → 1", new="v3", size=8.5)
        arrow(ax, (100, 32.4), (100, 34.6), color=ACCENT["v3"][0])
        ax.text(62, 40.5, "the NONE decision no longer reads the question vector itself", fontsize=7.5,
                color=ACCENT["v3"][0], style="italic", ha="center")
        none_out, none_rad = (108.4, 38.5), -0.3

    # output
    box(ax, 112, 14, 15, 14, "softmax over\nz₁ … z_K, z_NONE\n\n→ one option or\nNONE_OF_THESE", size=8.3,
        weight="bold")
    arrow(ax, (88.4, 11.5), (111.6, 17))
    arrow(ax, none_out, (119.5, 28.4), rad=none_rad)
    fig.savefig(OUT / f"arch_{version}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def training_v3() -> None:
    fig, ax = canvas(13.0, 4.6, 124, 44)
    ax.text(0.5, 43, "Reflex v3", fontsize=14, fontweight="bold", color=ACCENT["v3"][0], va="top")
    ax.text(16, 43, "training in three steps", fontsize=12, fontweight="bold", color=INK, va="top")
    ax.text(0.5, 38.8, "36 training labels are hidden from the scorer and play the role of new labels, "
                       "so the NONE head learns what the scores look like when the right option is unfamiliar.",
            fontsize=9, color=INK2, va="top")

    box(ax, 1, 14, 18, 14, "182 training\nlabels", size=10, weight="bold")
    box(ax, 26, 24, 20, 9, "146 base labels", size=9)
    box(ax, 26, 9, 20, 9, "36 pseudo-unseen\nlabels", new="v3", size=9)
    arrow(ax, (19.4, 24), (25.6, 28))
    arrow(ax, (19.4, 18), (25.6, 13.5))

    box(ax, 54, 24, 23, 9, "Step 1  train the scorer\non base labels only", size=8.5)
    arrow(ax, (46.4, 28.5), (53.6, 28.5))
    box(ax, 54, 9, 23, 9, "Step 2  freeze the scorer,\ntrain the NONE head on lists\nmixing base + pseudo-unseen",
        new="v3", size=8.2)
    arrow(ax, (46.4, 13.5), (53.6, 13.5))
    arrow(ax, (65.5, 23.6), (65.5, 18.4), label="frozen scorer", label_offset=(5.5, -1.0))

    box(ax, 86, 16, 21, 10, "Step 3  train a fresh scorer\non all 182 labels, plug in\nthe NONE head from step 2",
        size=8.2)
    arrow(ax, (77.4, 13.5), (85.6, 18.5), color=ACCENT["v3"][0], label="NONE head", label_offset=(4.5, -2.5))
    box(ax, 111, 16, 12, 10, "final v3\nmodel", size=9.5, weight="bold")
    arrow(ax, (107.4, 21), (110.6, 21))

    box(ax, 54, 0.5, 69, 5, "45 real unseen labels: never used for training or model choice; evaluation only",
        frozen=True, size=8.2)
    fig.savefig(OUT / "training_v3.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    for version in ("v1", "v2", "v3"):
        architecture(version)
    training_v3()
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
