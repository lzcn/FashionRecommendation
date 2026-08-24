#!/usr/bin/env python3
"""Render the ML-S BasicRecall three-panel figure using only the stdlib."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


COLORS = {1: "#5b8def", 10: "#222222", 100: "#2eb872", 1000: "#d62728"}
DASHES = {1: "", 10: "7 5 1.5 5", 100: "8 6", 1000: "2 5"}


def marker(x, y, top):
    color = COLORS[top]
    if top == 1:
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}"/>'
    if top == 10:
        return (f'<path d="M{x-4:.2f},{y:.2f}H{x+4:.2f} M{x:.2f},{y-4:.2f}V{y+4:.2f}" '
                f'stroke="{color}" stroke-width="1.4"/>')
    if top == 100:
        return (f'<path d="M{x-4:.2f},{y-4:.2f}L{x+4:.2f},{y+4:.2f} '
                f'M{x+4:.2f},{y-4:.2f}L{x-4:.2f},{y+4:.2f}" stroke="{color}" stroke-width="1.4"/>')
    return (f'<path d="M{x:.2f},{y-5:.2f}V{y+5:.2f} M{x-2.5:.2f},{y-5:.2f}H{x+2.5:.2f}" '
            f'stroke="{color}" stroke-width="1.3"/>')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    values = defaultdict(list)
    with args.input.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["power"]), int(row["requested_samples"]), int(row["top"]))
            values[key].append(float(row["recall"]))
    means = {key: sum(xs) / len(xs) for key, xs in values.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output.with_name("basic_recall_summary.csv")
    with summary_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["power", "samples", "top", "mean_recall", "repeats"])
        for key in sorted(values):
            writer.writerow([*key, f"{means[key]:.10g}", len(values[key])])

    width, height = 1120, 380
    left, top_margin, panel_w, panel_h, gap = 68, 46, 310, 265, 48
    samples = sorted({key[1] for key in means})
    tops = [1, 10, 100, 1000]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:"Times New Roman",Times,serif;fill:#111}.tick{font-size:16px}.title{font-size:24px}.label{font-size:20px}</style>',
        '<title>ML-S recall versus number of samples for Core powers one, two, and three</title>',
        '<desc>Three panels show mean recall over ten runs at top 1, 10, 100, and 1000.</desc>',
    ]
    for power in (1, 2, 3):
        x0 = left + (power - 1) * (panel_w + gap)
        y0 = top_margin
        svg.append(f'<text class="title" x="{x0 + panel_w/2}" y="30" text-anchor="middle">ML-S (Core<tspan baseline-shift="super" font-size="15">{power}</tspan>)</text>')
        for yt in range(0, 6):
            value = yt / 5
            y = y0 + panel_h * (1 - value)
            svg.append(f'<path d="M{x0},{y:.2f}H{x0+panel_w}" stroke="#d9d9d9" stroke-width="0.7"/>')
            svg.append(f'<text class="tick" x="{x0-9}" y="{y+5:.2f}" text-anchor="end">{value:.1f}</text>')
        for exponent in range(4, 8):
            x = x0 + panel_w * (exponent - 4) / 3
            svg.append(f'<path d="M{x:.2f},{y0}V{y0+panel_h}" stroke="#ededed" stroke-width="0.7"/>')
            svg.append(f'<text class="tick" x="{x:.2f}" y="{y0+panel_h+25}" text-anchor="middle">10<tspan baseline-shift="super" font-size="11">{exponent}</tspan></text>')
        svg.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#111" stroke-width="1.5"/>')
        for series_top in tops:
            points = []
            for sample in samples:
                value = means[(power, sample, series_top)]
                x = x0 + panel_w * (math.log10(sample) - 4) / 3
                y = y0 + panel_h * (1 - value)
                points.append((x, y))
            dash = f' stroke-dasharray="{DASHES[series_top]}"' if DASHES[series_top] else ""
            path = " ".join(("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}" for i, (x, y) in enumerate(points))
            svg.append(f'<path d="{path}" fill="none" stroke="{COLORS[series_top]}" stroke-width="2.1"{dash}/>')
            svg.extend(marker(x, y, series_top) for x, y in points)
        if power == 1:
            lx, ly = x0 + 12, y0 + 15
            svg.append(f'<rect x="{lx-6}" y="{ly-10}" width="122" height="105" fill="white" stroke="#111"/>')
            for idx, series_top in enumerate(tops):
                yy = ly + idx * 25
                dash = f' stroke-dasharray="{DASHES[series_top]}"' if DASHES[series_top] else ""
                svg.append(f'<path d="M{lx},{yy}H{lx+30}" stroke="{COLORS[series_top]}" stroke-width="2"{dash}/>')
                svg.append(marker(lx + 15, yy, series_top))
                svg.append(f'<text class="tick" x="{lx+38}" y="{yy+5}">top-{series_top}</text>')
    svg.append(f'<text class="label" x="18" y="{top_margin+panel_h/2}" text-anchor="middle" transform="rotate(-90 18 {top_margin+panel_h/2})">recall</text>')
    svg.append(f'<text class="label" x="{width/2}" y="370" text-anchor="middle">number of samples</text>')
    svg.append('</svg>')
    args.output.write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    main()
