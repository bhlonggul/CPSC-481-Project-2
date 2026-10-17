"""CPSC 481 Project 2: diagnose lung diseases by prompting Gemini.

Same interface as Project 1's Diagnostics class, but instead of running AIMA's
enumeration_ask, diagnose() describes the Asia Bayesian network and the evidence
to Gemini, asks it to do the exact inference, and reads back a structured answer.

The API key comes from the GEMINI_API_KEY environment variable and is never
stored in this file.
"""
import os
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel

# Structured output, checked against google-genai 2.31.0: generate_content with
# response_mime_type="application/json" and response_schema=<Pydantic model> makes
# Gemini reply with JSON that matches Diagnosis.

DEFAULT_MODEL = "gemini-3.8-flash"  # override with the GEMINI_MODEL environment variable
MAX_ATTEMPTS = 3  # the first try plus up to 2 retries when a reply is unusable


class Diagnosis(BaseModel):
    """The JSON shape Gemini must reply with."""
    disease: Literal["TB", "Cancer", "Bronchitis"]
    probability: float


# Every variable in the network, used to list the unobserved ones in the prompt
VARIABLES =("Asia", "Smoking", "TB", "Cancer", "Bronchitis", "TBorC", "Xray", "Dyspnea")

# Static part of the prompt: the network structure and CPTs, copied from Project 1.
# Plain ASCII only, so printing it never fails on a Windows console.
NETWORK_DESCRIPTION = """\
You are given a Bayesian network. Every variable is Boolean (True or False).

Variables and their parents:
- Asia: the patient visited Asia. No parents.
- Smoking: the patient smokes. No parents.
- TB: tuberculosis. Parent: Asia.
- Cancer: lung cancer. Parent: Smoking.
- Bronchitis: bronchitis. Parent: Smoking.
- TBorC: TB or Cancer (a deterministic OR). Parents: TB, Cancer.
- Xray: X-ray result, True = abnormal, False = normal. Parent: TBorC.
- Dyspnea: shortness of breath, True = present, False = absent. Parents: TBorC, Bronchitis.

Conditional probability tables. For every row, P(variable=False | parents) = 1 - P(variable=True | parents).
P(Asia=True) = 0.01
P(Smoking=True) = 0.5
P(TB=True | Asia=True) = 0.05
P(TB=True | Asia=False) = 0.01
P(Cancer=True | Smoking=True) = 0.1
P(Cancer=True | Smoking=False) = 0.01
P(Bronchitis=True | Smoking=True) = 0.6
P(Bronchitis=True | Smoking=False) = 0.3
P(TBorC=True | TB=True, Cancer=True) = 1.0
P(TBorC=True | TB=True, Cancer=False) = 1.0
P(TBorC=True | TB=False, Cancer=True) = 1.0
P(TBorC=True | TB=False, Cancer=False) = 0.0
P(Xray=True | TBorC=True) = 0.99
P(Xray=True | TBorC=False) = 0.05
P(Dyspnea=True | TBorC=True, Bronchitis=True) = 0.9
P(Dyspnea=True | TBorC=True, Bronchitis=False) = 0.7
P(Dyspnea=True | TBorC=False, Bronchitis=True) = 0.8
P(Dyspnea=True | TBorC=False, Bronchitis=False) = 0.1

The joint distribution is the product of these tables:
P(Asia, Smoking, TB, Cancer, Bronchitis, TBorC, Xray, Dyspnea)
  = P(Asia) * P(Smoking) * P(TB | Asia) * P(Cancer | Smoking) * P(Bronchitis | Smoking)
    * P(TBorC | TB, Cancer) * P(Xray | TBorC) * P(Dyspnea | TBorC, Bronchitis)
"""

TASK = """\
Task:
1. Use exact inference by enumeration. For each query, sum the joint distribution over all
   unobserved variables with the evidence fixed, then divide by P(evidence), which is the
   same sum without fixing the query variable.
2. Compute P(TB=True | evidence), P(Cancer=True | evidence) and P(Bronchitis=True | evidence).
   Compute each one separately: a patient can have more than one disease, so the three
   numbers do not have to add up to 1.
3. The diagnosis is the disease with the largest of the three posteriors. Report that disease
   and its posterior probability as a number between 0 and 1, rounded to 4 decimal places.
Double-check the arithmetic before answering.
"""


class Diagnostics:
    """ Use Gemini and a Bayesian network to diagnose between three lung diseases """

    def __init__(self):
        # The API key comes from the environment, never from this file. Without a key the
        # object can still be created; diagnose() then explains how to set one.
        api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key) if api_key else None
        self.model = os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        # Ask for JSON matching Diagnosis. Thinking stays on at "high", Google's setting for
        # math; temperature is left unset because Google deprecates it for Gemini 3.8 Flash
        # and warns that low values hurt reasoning.
        self.config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Diagnosis,
            thinking_config=types.ThinkingConfig(thinking_level="high"),
        )
        print(f"Diagnostics ready, using Gemini model {self.model}")
        if not api_key:
            print("Warning: GEMINI_API_KEY is not set, so diagnose() can't call Gemini yet")

    def diagnose(self, visit_to_asia, smoking, xray_result, dyspnea):
        print(f"Inputs: visit_to_asia={visit_to_asia}, smoking={smoking}, "
              f"xray_result={xray_result}, dyspnea={dyspnea}")

        # 1. Turn the inputs into evidence. "NA" means not observed, so it is left out.
        values = {
            "Asia": self._convert("visit_to_asia", visit_to_asia, "Yes", "No"),
            "Smoking": self._convert("smoking", smoking, "Yes", "No"),
            "Xray": self._convert("xray_result", xray_result, "Abnormal", "Normal"),
            "Dyspnea": self._convert("dyspnea", dyspnea, "Present", "Absent"),
        }
        evidence = {variable: value for variable, value in values.items() if value is not None}
        print(f"Evidence: {evidence}")

        # 2. Build the prompt: the network, the evidence, and what to compute
        prompt = self._prompt(evidence)
        print(f"Prompt sent to {self.model}:\n{prompt}")

        # 3. Ask Gemini, retrying when the reply can't be parsed or validated
        if self.client is None:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Create a key at https://aistudio.google.com/apikey "
                "and set it before running, for example\n"
                '  PowerShell: $env:GEMINI_API_KEY = "your-key"\n'
                '  bash:       export GEMINI_API_KEY="your-key"')
        for attempt in range(1, MAX_ATTEMPTS + 1):
            response = self.client.models.generate_content(
                model=self.model, contents=prompt, config=self.config)
            print(f"Raw response (attempt {attempt}): {response.text}")
            try:
                disease, probability = self._parse(response.text)
            except ValueError as error:  # includes pydantic's ValidationError
                print(f"Unusable response: {error}")
                continue
            print(f"Parsed result: disease={disease}, probability={probability}")
            return [disease, probability]

        raise RuntimeError(
            f"Gemini gave no usable diagnosis in {MAX_ATTEMPTS} attempts (see the responses above)")

    @staticmethod
    def _convert(name, value, true_value, false_value):
        """Map one input string to True/False, or None for "NA" (not observed)."""
        if value == "NA":
            return None
        if value == true_value:
            return True
        if value == false_value:
            return False
        raise ValueError(f"{name} must be '{true_value}', '{false_value}', or 'NA'")

    @staticmethod
    def _prompt(evidence):
        if evidence:
            observed = "Observed evidence:\n" + "".join(
                f"- {variable} = {value}\n" for variable, value in evidence.items())
        else:
            observed = "Observed evidence: none\n"
        unobserved = ", ".join(variable for variable in VARIABLES if variable not in evidence)
        return (f"{NETWORK_DESCRIPTION}\n{observed}"
                f"Unobserved variables (not evidence, sum them out): {unobserved}\n\n{TASK}")

    @staticmethod
    def _parse(text):
        """Validate Gemini's JSON reply; return (disease, probability) or raise ValueError."""
        diagnosis = Diagnosis.model_validate_json(text)  # also rejects a reply with no text (None)
        if not 0.0 <= diagnosis.probability <= 1.0:
            raise ValueError(f"probability {diagnosis.probability} is outside [0, 1]")
        return diagnosis.disease, float(diagnosis.probability)
