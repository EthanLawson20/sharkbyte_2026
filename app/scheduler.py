"""reads the schedule csv, groups by pill so each bottle only gets picked up once,
then walks the arm through it."""

import argparse
import csv
from collections import defaultdict

from arm_stub import StubArm
import verify

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    by_pill = defaultdict(list)
    for r in rows:
        r["count"] = int(r["count"])
        by_pill[r["pill"]].append(r)
    # monday to sunday so the arm sweeps left to right
    for pill in by_pill:
        by_pill[pill].sort(key=lambda r: (DAYS.index(r["day"]), r["slot"]))
    return by_pill


def run(by_pill, arm, check=True):
    results = []
    for pill, cells in by_pill.items():
        arm.pick_bottle(pill)
        for cell in cells:
            arm.dispense_to_tray(cell["count"])
            ok = True
            if check:
                ok = verify.tray_has(pill, cell["count"])
            if not ok:
                print(f"!! verify failed for {pill} {cell['day']}/{cell['slot']}, stopping")
                results.append((cell, "FAIL"))
                arm.home()
                return results
            for _ in range(cell["count"]):
                arm.place_in_cell(cell["day"], cell["slot"])
            results.append((cell, "OK"))
    arm.home()
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    if args.stub:
        arm = StubArm()
    else:
        from real_arm import RealArm
        arm = RealArm()
    try:
        out = run(load(args.csv), arm, check=not args.no_verify)
    finally:
        if hasattr(arm, "close"):
            arm.close()
    for cell, status in out:
        print(f"{status:4} {cell['pill']} -> {cell['day']}/{cell['slot']} x{cell['count']}")