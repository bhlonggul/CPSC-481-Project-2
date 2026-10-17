"""Check Diagnostics against the answer key in expected_answers.csv. Makes live Gemini calls.

Usage:
  python eval_against_key.py                        # the 11-row sample, spec case first
  python eval_against_key.py --all                  # all 81 rows
  python eval_against_key.py --all --offset 20 --limit 10
  python eval_against_key.py --limit 1 --verbose    # just the spec case, with every print

Each row is one diagnose() call (up to 3 requests if a reply needs a retry). To stay inside
free-tier rate limits, --delay waits between calls, and a 429 stops the run and prints the
--offset to resume from. Needs GEMINI_API_KEY, like diagnostics.py.
"""
import argparse
import csv
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

from diagnostics import Diagnostics

KEY_FILE = Path(__file__).with_name("expected_answers.csv")
INPUT_COLUMNS = ("visit_to_asia", "smoking", "xray", "dyspnea")
TOLERANCE = 0.01  # target: every probability within 0.01 of the key

# Spec case first, then rows covering all three diseases, including the three with
# the smallest gap between the top two posteriors.
SAMPLE = [
    ("No", "Yes", "Abnormal", "Absent"),   # Cancer 0.4367 (the spec case)
    ("Yes", "NA", "Abnormal", "Absent"),   # TB 0.2237, Cancer 0.2173 close behind
    ("Yes", "No", "Abnormal", "NA"),       # TB 0.4673
    ("Yes", "No", "Abnormal", "Present"),  # TB 0.6339
    ("NA", "Yes", "Abnormal", "Present"),  # Cancer 0.7252, Bronchitis 0.7133 close behind
    ("NA", "NA", "Abnormal", "Absent"),    # Cancer 0.2541
    ("Yes", "Yes", "Abnormal", "Absent"),  # Cancer 0.3844
    ("NA", "NA", "NA", "NA"),              # Bronchitis 0.45 (no evidence)
    ("No", "NA", "Abnormal", "NA"),        # Bronchitis 0.5068, Cancer 0.4924 close behind
    ("No", "No", "Abnormal", "Absent"),    # Bronchitis 0.0916
    ("NA", "Yes", "Normal", "Present"),    # Bronchitis 0.9225
]


def load_key(path=KEY_FILE):
    """Read the answer key: one dict per row, with the probability as a float."""
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["probability"] = float(row["probability"])
    return rows


def inputs_of(row):
    return tuple(row[column] for column in INPUT_COLUMNS)


def select_rows(rows, use_all=False, offset=0, limit=None):
    """The SAMPLE rows (or every row with use_all), then the slice [offset, offset + limit)."""
    if not use_all:
        by_inputs = {inputs_of(row): row for row in rows}
        rows = [by_inputs[inputs] for inputs in SAMPLE]
    end = None if limit is None else offset + limit
    return rows[offset:end]


def evaluate(rows, diagnose, delay=10.0, sleep=time.sleep, verbose=False):
    """Run diagnose on each row, print a line per row, and return a summary dict.

    diagnose's own prints are hidden unless verbose, but shown for any row that misses.
    """
    summary = {"tested": 0, "correct": 0, "within_tolerance": 0, "errors": 0,
               "mean_error": None, "max_error": None, "misses": [], "stopped_at": None}
    gaps = []  # |probability - key| for the rows that got the disease right
    for index, row in enumerate(rows):
        if index:
            sleep(delay)
        inputs = inputs_of(row)
        label = f"[{index + 1}/{len(rows)}] {','.join(inputs)}  key: {row['disease']} {row['probability']:.4f}"
        captured = io.StringIO()
        passed = False
        try:
            with redirect_stdout(sys.stdout if verbose else captured):
                disease, probability = diagnose(*inputs)
        except Exception as error:
            if getattr(error, "code", None) == 429:  # out of quota: every later row would fail too
                print(f"{label}  rate limited, stopping: {error}")
                summary["stopped_at"] = index
                break
            summary["errors"] += 1
            outcome = f"ERROR: {error}"
        else:
            if disease == row["disease"]:
                gap = abs(probability - row["probability"])
                gaps.append(gap)
                summary["correct"] += 1
                passed = gap <= TOLERANCE + 1e-9  # slack so a gap of exactly 0.01 still counts
                outcome = f"got: {disease} {probability:.4f}  |error| {gap:.4f}  {'OK' if passed else 'TOO FAR'}"
            else:
                outcome = f"got: {disease} {probability:.4f}  WRONG DISEASE"
        summary["tested"] += 1
        print(f"{label}  {outcome}")
        if passed:
            summary["within_tolerance"] += 1
        else:
            summary["misses"].append(row)
            print(captured.getvalue(), end="")  # what diagnose printed, to see what went wrong
    if gaps:
        summary["mean_error"] = sum(gaps) / len(gaps)
        summary["max_error"] = max(gaps)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check Diagnostics against expected_answers.csv (makes live Gemini calls).")
    parser.add_argument("--all", action="store_true", help="use all 81 rows instead of the 11-row sample")
    parser.add_argument("--offset", type=int, default=0, help="skip this many of the selected rows")
    parser.add_argument("--limit", type=int, help="run at most this many rows")
    parser.add_argument("--delay", type=float, default=10.0, help="seconds to wait between calls (default: 10)")
    parser.add_argument("--verbose", action="store_true", help="show everything diagnose() prints")
    args = parser.parse_args(argv)

    rows = select_rows(load_key(), args.all, args.offset, args.limit)
    summary = evaluate(rows, Diagnostics().diagnose, args.delay, verbose=args.verbose)

    tested = summary["tested"]
    print()
    print(f"Rows tested: {tested}")
    print(f"Disease accuracy: {summary['correct']}/{tested}")
    if summary["mean_error"] is not None:
        print(f"Probability error on rows with the right disease: mean {summary['mean_error']:.4f}, "
              f"max {summary['max_error']:.4f}")
    print(f"Within {TOLERANCE} of the key: {summary['within_tolerance']}/{tested}")
    print(f"Errors: {summary['errors']}")
    if summary["stopped_at"] is not None:
        print(f"Stopped by the rate limit. Resume with the same options plus "
              f"--offset {args.offset + summary['stopped_at']}")
    return 1 if summary["misses"] or summary["stopped_at"] is not None else 0


if __name__ == "__main__":
    sys.exit(main())
