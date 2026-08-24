#!/usr/bin/env python3
"""Compare Core runtimes from two star_sampling benchmark CSV files."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def load(path):
    unique = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["power"]), int(row["requested_samples"]), int(row["repeat"]))
            unique[key] = float(row["total_s"])
    grouped = defaultdict(list)
    for (power, samples, _), value in unique.items():
        grouped[(power, samples)].append(value)
    return {key: sum(xs) / len(xs) for key, xs in grouped.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--old-label", default="head ratio 1.34")
    parser.add_argument("--new-label", default="head ratio 3.26")
    args = parser.parse_args()
    old, new = load(args.old), load(args.new)
    keys = sorted(set(old) & set(new))
    samples = sorted({sample for _, sample in keys})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = args.output.with_name("runtime_comparison_summary.csv")
    with summary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["power", "samples", "old_mean_s", "new_mean_s",
                         "new_over_old", "old_over_new"])
        for power, sample in keys:
            writer.writerow([power, sample, f"{old[(power,sample)]:.10g}",
                             f"{new[(power,sample)]:.10g}",
                             f"{new[(power,sample)]/old[(power,sample)]:.10g}",
                             f"{old[(power,sample)]/new[(power,sample)]:.10g}"])

    width, height = 1120, 380
    left, top, panel_w, panel_h, gap = 68, 46, 310, 265, 48
    all_values = list(old.values()) + list(new.values())
    ymin = 10 ** math.floor(math.log10(min(all_values)))
    ymax = 10 ** math.ceil(math.log10(max(all_values)))
    lymin, lymax = math.log10(ymin), math.log10(ymax)
    colors = ["#5b8def", "#2eb872"]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:"Times New Roman",Times,serif;fill:#111}.tick{font-size:16px}.title{font-size:24px}.label{font-size:20px}</style>',
        '<title>Runtime comparison for ML-S Core sampling</title>',
        '<desc>Three panels compare mean total runtime over ten runs for factor tensors with head ratios 1.34 and 3.26.</desc>',
    ]
    for power in (1, 2, 3):
        x0 = left + (power - 1) * (panel_w + gap)
        svg.append(f'<text class="title" x="{x0+panel_w/2}" y="30" text-anchor="middle">ML-S (Core<tspan baseline-shift="super" font-size="15">{power}</tspan>)</text>')
        for exponent in range(int(lymin), int(lymax) + 1):
            y = top + panel_h * (lymax - exponent) / (lymax - lymin)
            svg.append(f'<path d="M{x0},{y:.2f}H{x0+panel_w}" stroke="#dddddd" stroke-width="0.7"/>')
            svg.append(f'<text class="tick" x="{x0-9}" y="{y+5:.2f}" text-anchor="end">10<tspan baseline-shift="super" font-size="11">{exponent}</tspan></text>')
        for exponent in range(4, 8):
            x = x0 + panel_w * (exponent - 4) / 3
            svg.append(f'<path d="M{x:.2f},{top}V{top+panel_h}" stroke="#eeeeee" stroke-width="0.7"/>')
            svg.append(f'<text class="tick" x="{x:.2f}" y="{top+panel_h+25}" text-anchor="middle">10<tspan baseline-shift="super" font-size="11">{exponent}</tspan></text>')
        svg.append(f'<rect x="{x0}" y="{top}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#111" stroke-width="1.5"/>')
        for series_index, dataset in enumerate((old, new)):
            points = []
            for sample in samples:
                value = dataset[(power, sample)]
                x = x0 + panel_w * (math.log10(sample) - 4) / 3
                y = top + panel_h * (lymax - math.log10(value)) / (lymax - lymin)
                points.append((x, y))
            path = " ".join(("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}" for i, (x, y) in enumerate(points))
            dash = ' stroke-dasharray="8 5"' if series_index else ""
            svg.append(f'<path d="{path}" fill="none" stroke="{colors[series_index]}" stroke-width="2.2"{dash}/>')
            for x, y in points:
                if series_index == 0:
                    svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{colors[series_index]}"/>')
                else:
                    svg.append(f'<rect x="{x-3.5:.2f}" y="{y-3.5:.2f}" width="7" height="7" fill="{colors[series_index]}"/>')
        if power == 1:
            lx, ly = x0 + 12, top + 16
            svg.append(f'<rect x="{lx-6}" y="{ly-11}" width="164" height="57" fill="white" stroke="#111"/>')
            for idx, (label, color) in enumerate(((args.old_label, colors[0]), (args.new_label, colors[1]))):
                yy = ly + idx * 26
                dash = ' stroke-dasharray="8 5"' if idx else ""
                svg.append(f'<path d="M{lx},{yy}H{lx+30}" stroke="{color}" stroke-width="2.2"{dash}/>')
                svg.append(f'<text class="tick" x="{lx+38}" y="{yy+5}">{label}</text>')
    svg.append(f'<text class="label" x="18" y="{top+panel_h/2}" text-anchor="middle" transform="rotate(-90 18 {top+panel_h/2})">total time (seconds)</text>')
    svg.append(f'<text class="label" x="{width/2}" y="370" text-anchor="middle">number of samples</text>')
    svg.append('</svg>')
    args.output.write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    main()
