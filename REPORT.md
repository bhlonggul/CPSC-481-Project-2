# CPSC 481 Project 2: Bayesian Network Diagnosis with Gemini

## (a) Why this LLM

We used Gemini 3.8 Flash through Google's `google-genai` SDK. It is Google's current stable Flash model and supports both structured JSON output and an adjustable "thinking" level for multi-step arithmetic. The spec's suggested Gemini 2.5 Flash is now limited to accounts that have used it before.

## (b) Major challenge

The main challenge was making the model's answer both correct and machine-readable. Exact inference means summing over every combination of the unobserved variables, and a free-text reply would be hard to parse. We gave the model the full network, the observed evidence and the enumeration method, set its thinking level to "high", and required a JSON reply matching a schema in which the disease must be TB, Cancer or Bronchitis; we validate it and retry up to twice. API churn added to this: Google's docs show two different structured-output APIs, so we confirmed the call shape against the installed SDK before writing any code.

## (c) Something surprising

With thinking on, Gemini matched exact enumeration to four decimal places on every case we tested (the spec case and an 11-row sample), even when the top two diseases were only 0.0064 apart. However, each answer took about 40-55 seconds, where `enumeration_ask` returns almost instantly. Google also advises against lowering the temperature for Gemini 3 models, which is the opposite of the usual "use a low temperature for math" advice.
