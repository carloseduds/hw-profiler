"""Command-line interface for hw_profiler."""

from __future__ import annotations

import argparse
import json

from .core import build_profile, print_human_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hw-profiler")
    parser.add_argument("--json", action="store_true", help="Print full profile as JSON.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument("--out", type=str, default=None, help="Write JSON to a file path.")
    parser.add_argument("--no-topology", action="store_true", help="Skip topology collection.")
    args = parser.parse_args(argv)

    profile = build_profile(include_topology=not args.no_topology)

    if args.json or args.out:
        payload = json.dumps(profile, indent=2 if args.pretty else None, ensure_ascii=False)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(payload)
        if args.json:
            print(payload)
    else:
        print_human_summary(profile)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
