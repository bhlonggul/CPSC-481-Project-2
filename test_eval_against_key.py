"""Offline tests for eval_against_key.py. diagnose() is faked, so nothing calls Gemini.

Run with: python -m unittest -v test_eval_against_key
"""
import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from google.genai import errors

import eval_against_key
from eval_against_key import evaluate, load_key, select_rows

SPEC_CASE = ("No", "Yes", "Abnormal", "Absent")
NO_EVIDENCE = ("NA", "NA", "NA", "NA")


def key_row(inputs, disease, probability):
    asia, smoking, xray, dyspnea = inputs
    return {"visit_to_asia": asia, "smoking": smoking, "xray": xray, "dyspnea": dyspnea,
            "disease": disease, "probability": probability}


def rate_limit_error():
    return errors.ClientError(429, {"error": {
        "code": 429, "message": "Resource has been exhausted", "status": "RESOURCE_EXHAUSTED"}})


class FakeDiagnose:
    """diagnose() stand-in: prints a marker line, then answers from a table and records the call.

    An answer is [disease, probability], or an exception to raise.
    """

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def __call__(self, *inputs):
        self.calls.append(inputs)
        print(f"diagnose output for {','.join(inputs)}")
        answer = self.answers[inputs]
        if isinstance(answer, Exception):
            raise answer
        return answer


def quiet(function, *args, **kwargs):
    """Call function, capturing what it prints; return (result, printed text)."""
    out = io.StringIO()
    with redirect_stdout(out):
        result = function(*args, **kwargs)
    return result, out.getvalue()


class SelectRowsTests(unittest.TestCase):
    def setUp(self):
        self.key = load_key()

    def test_sample_starts_with_the_spec_case_and_covers_all_three_diseases(self):
        sample = select_rows(self.key)
        self.assertEqual(sample[0], key_row(SPEC_CASE, "Cancer", 0.4367))
        self.assertEqual({row["disease"] for row in sample}, {"TB", "Cancer", "Bronchitis"})

    def test_all_uses_every_row_of_the_key(self):
        self.assertEqual(len(select_rows(self.key, use_all=True)), 81)

    def test_offset_and_limit_take_a_slice(self):
        rows = select_rows(self.key, use_all=True, offset=10, limit=5)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0], key_row(("NA", "Yes", "NA", "Present"), "Bronchitis", 0.8802))
        self.assertEqual(rows[-1], key_row(("NA", "Yes", "Abnormal", "Absent"), "Cancer", 0.4361))


class EvaluateTests(unittest.TestCase):
    def test_scores_disease_accuracy_and_probability_error(self):
        rows = [
            key_row(SPEC_CASE, "Cancer", 0.4367),
            key_row(NO_EVIDENCE, "Bronchitis", 0.45),
            key_row(("Yes", "No", "Abnormal", "NA"), "TB", 0.4673),
            key_row(("NA", "NA", "Abnormal", "Absent"), "Cancer", 0.2541),
        ]
        diagnose = FakeDiagnose({
            SPEC_CASE: ["Cancer", 0.4367],                           # exact
            NO_EVIDENCE: ["Bronchitis", 0.455],                      # off by 0.005: within 0.01
            ("Yes", "No", "Abnormal", "NA"): ["TB", 0.4873],         # off by 0.02: outside
            ("NA", "NA", "Abnormal", "Absent"): ["Bronchitis", 0.1935],  # wrong disease
        })
        summary, _ = quiet(evaluate, rows, diagnose, delay=0)
        self.assertEqual(summary["tested"], 4)
        self.assertEqual(summary["correct"], 3)
        self.assertEqual(summary["within_tolerance"], 2)
        self.assertAlmostEqual(summary["mean_error"], (0.0 + 0.005 + 0.02) / 3)
        self.assertAlmostEqual(summary["max_error"], 0.02)
        self.assertEqual(summary["misses"], rows[2:])

    def test_an_error_of_exactly_the_tolerance_still_counts_as_within(self):
        rows = [key_row(SPEC_CASE, "Cancer", 0.4367)]
        summary, _ = quiet(evaluate, rows, FakeDiagnose({SPEC_CASE: ["Cancer", 0.4467]}), delay=0)
        self.assertEqual(summary["within_tolerance"], 1)
        self.assertEqual(summary["misses"], [])

    def test_a_row_that_raises_counts_as_an_error_and_the_run_continues(self):
        rows = [key_row(SPEC_CASE, "Cancer", 0.4367), key_row(NO_EVIDENCE, "Bronchitis", 0.45)]
        diagnose = FakeDiagnose({
            SPEC_CASE: RuntimeError("Gemini gave no usable diagnosis"),
            NO_EVIDENCE: ["Bronchitis", 0.45],
        })
        summary, printed = quiet(evaluate, rows, diagnose, delay=0)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(len(diagnose.calls), 2)
        self.assertEqual(summary["misses"], [rows[0]])
        self.assertIn("Gemini gave no usable diagnosis", printed)

    def test_rate_limit_stops_the_run(self):
        third = ("Yes", "No", "Abnormal", "NA")
        rows = [key_row(SPEC_CASE, "Cancer", 0.4367), key_row(NO_EVIDENCE, "Bronchitis", 0.45),
                key_row(third, "TB", 0.4673)]
        diagnose = FakeDiagnose({SPEC_CASE: ["Cancer", 0.4367], NO_EVIDENCE: rate_limit_error(),
                                 third: ["TB", 0.4673]})
        summary, _ = quiet(evaluate, rows, diagnose, delay=0)
        self.assertEqual(summary["stopped_at"], 1)
        self.assertEqual(summary["tested"], 1)
        self.assertEqual(len(diagnose.calls), 2)  # the third row is never sent

    def test_waits_between_calls_but_not_before_the_first(self):
        rows = [key_row(SPEC_CASE, "Cancer", 0.4367)] * 3
        sleeps = []
        quiet(evaluate, rows, FakeDiagnose({SPEC_CASE: ["Cancer", 0.4367]}), delay=7.5, sleep=sleeps.append)
        self.assertEqual(sleeps, [7.5, 7.5])

    def test_shows_diagnose_output_only_for_rows_that_miss(self):
        rows = [key_row(SPEC_CASE, "Cancer", 0.4367), key_row(NO_EVIDENCE, "Bronchitis", 0.45)]
        diagnose = FakeDiagnose({SPEC_CASE: ["Cancer", 0.4367], NO_EVIDENCE: ["TB", 0.3]})
        _, printed = quiet(evaluate, rows, diagnose, delay=0)
        self.assertNotIn("diagnose output for No,Yes,Abnormal,Absent", printed)
        self.assertIn("diagnose output for NA,NA,NA,NA", printed)

    def test_verbose_shows_diagnose_output_for_every_row(self):
        rows = [key_row(SPEC_CASE, "Cancer", 0.4367)]
        _, printed = quiet(evaluate, rows, FakeDiagnose({SPEC_CASE: ["Cancer", 0.4367]}), delay=0, verbose=True)
        self.assertIn("diagnose output for No,Yes,Abnormal,Absent", printed)


class MainTests(unittest.TestCase):
    def run_main(self, argv, answers):
        """Run main(argv) with Diagnostics replaced by a fake; return (exit status, printed, fake diagnose)."""
        fake_diagnose = FakeDiagnose(answers)

        class FakeDiagnostics:
            def __init__(self):
                self.diagnose = fake_diagnose

        with mock.patch.object(eval_against_key, "Diagnostics", FakeDiagnostics):
            status, printed = quiet(eval_against_key.main, argv)
        return status, printed, fake_diagnose

    def test_exit_status_is_zero_only_when_every_row_passes(self):
        status, _, _ = self.run_main(["--limit", "1", "--delay", "0"], {SPEC_CASE: ["Cancer", 0.4367]})
        self.assertEqual(status, 0)
        status, _, _ = self.run_main(["--limit", "1", "--delay", "0"], {SPEC_CASE: ["Bronchitis", 0.2897]})
        self.assertEqual(status, 1)

    def test_rate_limit_message_gives_the_offset_to_resume_from(self):
        answers = {
            ("NA", "NA", "Abnormal", "NA"): ["Bronchitis", 0.5066],   # key row 3
            ("NA", "NA", "Abnormal", "Present"): rate_limit_error(),  # key row 4
            ("NA", "NA", "Abnormal", "Absent"): ["Cancer", 0.2541],   # key row 5
        }
        status, printed, fake = self.run_main(["--all", "--offset", "3", "--limit", "3", "--delay", "0"], answers)
        self.assertEqual(status, 1)
        self.assertEqual(len(fake.calls), 2)
        self.assertIn("--offset 4", printed)


if __name__ == "__main__":
    unittest.main()
