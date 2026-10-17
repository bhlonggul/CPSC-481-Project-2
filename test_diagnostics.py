"""Offline tests for diagnostics.Diagnostics. No network or real API key needed.

The real google-genai client is used, but its HTTP transport is swapped for
FakeGemini, so each test sees the exact request Diagnostics sends and the SDK
parses a reply shaped like a real generateContent response.

Run with: python -m unittest -v test_diagnostics
"""
import io
import json
import os
import re
import unittest
from contextlib import redirect_stdout
from unittest import mock

import httpx
from google import genai
from google.genai import errors, types

import diagnostics
from diagnostics import Diagnostics

SPEC_CASE = ("No", "Yes", "Abnormal", "Absent")
GOOD_REPLY = '{"disease": "Cancer", "probability": 0.4367}'

# Every CPT row, P(variable=True | parents), copied by hand from PROJECT2_PLAN.md.
CPT_LINES = [
    "P(Asia=True) = 0.01",
    "P(Smoking=True) = 0.5",
    "P(TB=True | Asia=True) = 0.05",
    "P(TB=True | Asia=False) = 0.01",
    "P(Cancer=True | Smoking=True) = 0.1",
    "P(Cancer=True | Smoking=False) = 0.01",
    "P(Bronchitis=True | Smoking=True) = 0.6",
    "P(Bronchitis=True | Smoking=False) = 0.3",
    "P(TBorC=True | TB=True, Cancer=True) = 1.0",
    "P(TBorC=True | TB=True, Cancer=False) = 1.0",
    "P(TBorC=True | TB=False, Cancer=True) = 1.0",
    "P(TBorC=True | TB=False, Cancer=False) = 0.0",
    "P(Xray=True | TBorC=True) = 0.99",
    "P(Xray=True | TBorC=False) = 0.05",
    "P(Dyspnea=True | TBorC=True, Bronchitis=True) = 0.9",
    "P(Dyspnea=True | TBorC=True, Bronchitis=False) = 0.7",
    "P(Dyspnea=True | TBorC=False, Bronchitis=True) = 0.8",
    "P(Dyspnea=True | TBorC=False, Bronchitis=False) = 0.1",
]


class FakeGemini:
    """Stands in for the Gemini API: records each request and answers with the next canned reply.

    A reply is the model's text, None for a candidate with no text (the model ran out
    of tokens while thinking), or an httpx.Response to send back as-is.
    """

    def __init__(self):
        self.replies = []
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, httpx.Response):
            return reply
        if reply is None:
            candidate = {"finishReason": "MAX_TOKENS", "index": 0}
        else:
            candidate = {
                "content": {"role": "model", "parts": [{"text": reply}]},
                "finishReason": "STOP",
                "index": 0,
            }
        return httpx.Response(200, json={
            "candidates": [candidate],
            "usageMetadata": {
                "promptTokenCount": 900,
                "candidatesTokenCount": 15,
                "thoughtsTokenCount": 1800,
                "totalTokenCount": 2715,
            },
            "modelVersion": "gemini-3.8-flash",
            "responseId": "fake-response-id",
        })

    def payload(self):
        return json.loads(self.requests[-1].content)

    def prompt(self):
        return self.payload()["contents"][0]["parts"][0]["text"]


def evidence_lines(prompt):
    """The prompt's '- <Variable> = <value>' evidence lines, as {variable: value}."""
    found = {}
    for line in prompt.splitlines():
        match = re.match(r"- (Asia|Smoking|Xray|Dyspnea) = (True|False)\b", line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def line_starting(text, prefix):
    return next(line for line in text.splitlines() if line.startswith(prefix))


class DiagnosticsTestCase(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("GEMINI_MODEL", None)
        os.environ.pop("GOOGLE_API_KEY", None)

        # Diagnostics builds a real SDK client; only the network underneath is fake.
        self.gemini = FakeGemini()
        real_client = genai.Client

        def offline_client(**kwargs):
            http = httpx.Client(transport=httpx.MockTransport(self.gemini))
            return real_client(**kwargs, http_options=types.HttpOptions(httpx_client=http))

        client = mock.patch.object(diagnostics.genai, "Client", side_effect=offline_client)
        client.start()
        self.addCleanup(client.stop)

    def new_diagnostics(self):
        with redirect_stdout(io.StringIO()):
            return Diagnostics()

    def run_case(self, *replies, args=SPEC_CASE):
        """Run diagnose(*args) against the canned replies; return (result, printed output)."""
        self.gemini.replies = list(replies)
        self.gemini.requests = []
        dx = self.new_diagnostics()
        out = io.StringIO()
        with redirect_stdout(out):
            result = dx.diagnose(*args)
        return result, out.getvalue()


class ConstructorTests(DiagnosticsTestCase):
    def test_missing_api_key_raises_error_naming_the_variable(self):
        del os.environ["GEMINI_API_KEY"]
        with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
            self.new_diagnostics()

    def test_api_key_comes_from_the_environment(self):
        self.run_case(GOOD_REPLY)
        self.assertEqual(self.gemini.requests[-1].headers["x-goog-api-key"], "test-key")

    def test_default_model_is_gemini_3_8_flash(self):
        self.run_case(GOOD_REPLY)
        self.assertTrue(self.gemini.requests[-1].url.path.endswith("/models/gemini-3.8-flash:generateContent"))

    def test_gemini_model_env_var_overrides_the_default(self):
        os.environ["GEMINI_MODEL"] = "gemini-3.6-flash"
        self.run_case(GOOD_REPLY)
        self.assertTrue(self.gemini.requests[-1].url.path.endswith("/models/gemini-3.6-flash:generateContent"))


class InputTests(DiagnosticsTestCase):
    def test_accepts_the_spec_parameter_names_as_keywords(self):
        self.gemini.replies = [GOOD_REPLY]
        dx = self.new_diagnostics()
        with redirect_stdout(io.StringIO()):
            result = dx.diagnose(visit_to_asia="No", smoking="Yes", xray_result="Abnormal", dyspnea="Absent")
        self.assertEqual(result, ["Cancer", 0.4367])

    def test_rejects_values_outside_each_inputs_allowed_set(self):
        cases = [
            (("Sometimes", "Yes", "Abnormal", "Absent"), "visit_to_asia"),
            (("No", "yes", "Abnormal", "Absent"), "smoking"),  # case-sensitive, like Project 1
            (("No", "Yes", "Blurry", "Absent"), "xray_result"),
            (("No", "Yes", "Yes", "Absent"), "xray_result"),  # a value from another input's set
            (("No", "Yes", "Abnormal", "Maybe"), "dyspnea"),
            (("No", "Yes", "Abnormal", " Absent"), "dyspnea"),  # no whitespace stripping
        ]
        dx = self.new_diagnostics()
        for args, bad_argument in cases:
            with self.subTest(args=args):
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, bad_argument):
                    dx.diagnose(*args)
        self.assertEqual(self.gemini.requests, [])  # nothing invalid reaches the model


class PromptTests(DiagnosticsTestCase):
    def test_states_every_observed_input_as_evidence(self):
        self.run_case(GOOD_REPLY)
        self.assertEqual(
            evidence_lines(self.gemini.prompt()),
            {"Asia": "False", "Smoking": "True", "Xray": "True", "Dyspnea": "False"},
        )

    def test_leaves_na_inputs_out_of_the_evidence(self):
        self.run_case(GOOD_REPLY, args=("NA", "Yes", "Abnormal", "NA"))
        prompt = self.gemini.prompt()
        self.assertEqual(evidence_lines(prompt), {"Smoking": "True", "Xray": "True"})
        unobserved = line_starting(prompt, "Unobserved")
        for variable in ("Asia", "Dyspnea"):
            self.assertIn(variable, unobserved)
        for variable in ("Smoking", "Xray"):
            self.assertNotIn(variable, unobserved)

    def test_all_na_inputs_give_no_evidence(self):
        self.run_case(GOOD_REPLY, args=("NA", "NA", "NA", "NA"))
        prompt = self.gemini.prompt()
        self.assertEqual(evidence_lines(prompt), {})
        self.assertIn("Observed evidence: none", prompt)

    def test_includes_every_cpt_row(self):
        self.run_case(GOOD_REPLY)
        prompt = self.gemini.prompt()
        for line in CPT_LINES:
            with self.subTest(line=line):
                self.assertIn(line, prompt)

    def test_asks_for_all_three_posteriors(self):
        self.run_case(GOOD_REPLY)
        prompt = self.gemini.prompt()
        for query in ("P(TB=True | evidence)", "P(Cancer=True | evidence)", "P(Bronchitis=True | evidence)"):
            with self.subTest(query=query):
                self.assertIn(query, prompt)


class RequestTests(DiagnosticsTestCase):
    def test_requests_json_matching_the_diagnosis_schema(self):
        self.run_case(GOOD_REPLY)
        config = self.gemini.payload()["generationConfig"]
        self.assertEqual(config["responseMimeType"], "application/json")
        schema = config.get("responseSchema") or config.get("responseJsonSchema")
        self.assertEqual(schema["properties"]["disease"]["enum"], ["TB", "Cancer", "Bronchitis"])
        self.assertEqual(sorted(schema["required"]), ["disease", "probability"])

    def test_keeps_thinking_high_and_leaves_temperature_unset(self):
        self.run_case(GOOD_REPLY)
        config = self.gemini.payload()["generationConfig"]
        thinking = config["thinkingConfig"]
        # The SDK copies ThinkingConfig verbatim, so the key is snake_case on the wire.
        self.assertEqual(thinking.get("thinkingLevel") or thinking.get("thinking_level"), "HIGH")
        self.assertNotIn("temperature", config)


class ResultTests(DiagnosticsTestCase):
    def test_returns_disease_and_probability_as_a_list(self):
        result, _ = self.run_case(GOOD_REPLY)
        self.assertEqual(result, ["Cancer", 0.4367])
        self.assertIs(type(result), list)
        self.assertIs(type(result[1]), float)

    def test_retries_once_after_an_unusable_reply(self):
        unusable = {
            "not JSON": "Cancer, about 44%",
            "unknown disease": '{"disease": "Flu", "probability": 0.4}',
            "percent instead of probability": '{"disease": "Cancer", "probability": 43.67}',
            "negative probability": '{"disease": "Cancer", "probability": -0.1}',
            "missing field": '{"disease": "Cancer"}',
            "no text": None,
        }
        for label, bad_reply in unusable.items():
            with self.subTest(label):
                result, _ = self.run_case(bad_reply, GOOD_REPLY)
                self.assertEqual(result, ["Cancer", 0.4367])
                self.assertEqual(len(self.gemini.requests), 2)

    def test_gives_up_after_three_unusable_replies(self):
        bad_reply = '{"disease": "Flu", "probability": 0.4}'
        with self.assertRaises(RuntimeError):
            self.run_case(bad_reply, bad_reply, bad_reply, GOOD_REPLY)
        self.assertEqual(len(self.gemini.requests), 3)

    def test_api_errors_are_raised_without_retrying(self):
        quota = httpx.Response(429, json={"error": {
            "code": 429, "message": "Resource has been exhausted", "status": "RESOURCE_EXHAUSTED"}})
        with self.assertRaises(errors.ClientError) as caught:
            self.run_case(quota, GOOD_REPLY)
        self.assertEqual(caught.exception.code, 429)
        self.assertEqual(len(self.gemini.requests), 1)


class OutputTests(DiagnosticsTestCase):
    def test_prints_each_intermediate_step(self):
        _, printed = self.run_case(GOOD_REPLY)
        inputs = line_starting(printed, "Inputs")
        for value in SPEC_CASE:
            self.assertIn(value, inputs)
        self.assertIn("'Smoking': True", line_starting(printed, "Evidence"))
        self.assertIn("gemini-3.8-flash", line_starting(printed, "Prompt"))
        self.assertIn("P(Xray=True | TBorC=True) = 0.99", printed)  # the prompt itself
        self.assertIn(GOOD_REPLY, line_starting(printed, "Raw response"))
        parsed = line_starting(printed, "Parsed")
        self.assertIn("Cancer", parsed)
        self.assertIn("0.4367", parsed)

    def test_printed_output_is_plain_ascii(self):
        # Windows pipes use a legacy code page; one non-ASCII character in a print would crash diagnose.
        self.gemini.replies = [GOOD_REPLY]
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="ascii", errors="strict")
        with redirect_stdout(stream):
            Diagnostics().diagnose(*SPEC_CASE)
        stream.flush()
        self.assertIn(b"Raw response", raw.getvalue())


if __name__ == "__main__":
    unittest.main()
