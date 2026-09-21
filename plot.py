#!/usr/bin/env python3
"""XKCD-style bar chart of achieved memory bandwidth per kernel.

Usage:
  python3 plot_bw.py "naive (Left)=112.3" "less_redundant (Left)=240.1" "shared (Left)=300"
  python3 plot_bw.py --peak 960 --out bw.png "naive=112.3" "shared=300"

Each positional argument is LABEL=VALUE (value in GB/s). The split is done on the
LAST '=', so labels may contain '='.
"""
import argparse
import sys

import matplotlib.pyplot as plt


def parse_pair(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected LABEL=VALUE, got '{s}'")
    label, value = s.rsplit("=", 1)
    try:
        return label.strip(), float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"value is not a number in '{s}'")


def wrap_label(label):
    # Horizontal tick labels: put each space-separated word on its own line
    # ("less_redundant (Left)" -> "less_redundant\n(Left)") so neighbours don't collide.
    return "\n".join(label.split())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pairs", nargs="+", type=parse_pair, help="LABEL=VALUE pairs (GB/s)")
    p.add_argument("--peak", type=float, default=None,
                   help="GPU peak bandwidth in GB/s: draws a reference line and %% labels")
    p.add_argument("--title", default="Mat-vec product: achieved bandwidth")
    p.add_argument("--ylabel", default="bandwidth [GB/s]")
    p.add_argument("--colors", default="#000000",
                   help="comma-separated bar colors, cycled (default black)")
    p.add_argument("--peak-color", default="#E3001B", help="peak line color (default red #E3001B)")
    p.add_argument("--out", default="bandwidth.png", help="output image (default bandwidth.png)")
    p.add_argument("--no-show", action="store_true", help="only save, do not open a window")
    args = p.parse_args()

    labels = [l for l, _ in args.pairs]
    values = [v for _, v in args.pairs]

    n = len(values)
    palette = [c.strip() for c in args.colors.split(",") if c.strip()]
    colors = [palette[k % len(palette)] for k in range(n)]

    # Horizontal labels need room: size each bar slot from the longest label line.
    longest = max(len(line) for l in labels for line in wrap_label(l).split("\n"))
    slot_in = max(0.9, 0.058 * longest + 0.05)

    with plt.xkcd():
        plt.rcParams.update({"font.size": 9})
        fig, ax = plt.subplots(figsize=(max(5.5, min(1.5 + slot_in * n, 24.0)), 4.2))
        x = list(range(n))
        bars = ax.bar(x, values, width=0.4, color=colors, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([wrap_label(l) for l in labels], rotation=0, ha="center")
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylabel(args.ylabel)

        top = max(values)
        if args.peak is not None:
            top = max(top, args.peak)
            ax.axhline(args.peak, linestyle="-", linewidth=2.5, color=args.peak_color, zorder=1,
                       label=f"GPU peak {args.peak:g} GB/s")
            leg = ax.legend(loc="upper right", frameon=False)
            for t in leg.get_texts():
                t.set_color(args.peak_color)
        ax.set_ylim(0, 1.33 * top)

        # Labels go INSIDE the bars (white, no xkcd stroke), so nothing drawn above
        # a bar can ever collide with them. Bars too short to hold the text get it above.
        for b, v in zip(bars, values):
            pct = f"\n{100.0 * v / args.peak:.1f}%" if args.peak else ""
            txt = f"{v:.1f}{pct}"
            cx = b.get_x() + b.get_width() / 2
            if v >= 0.2 * top:
                ax.text(cx, v - 0.03 * top, txt, ha="center", va="top", color="white",
                        fontweight="bold", zorder=3, path_effects=[])
            else:
                ax.text(cx, v + 0.03 * top, txt, ha="center", va="bottom", color="black", zorder=3)

        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f"saved {args.out}")
        if not args.no_show:
            plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())