# CPSC 481 Project 2: Bayesian Network Diagnosis with Gemini

Handoff plan for Claude Code. Read this whole file before writing code.

## Goal

Rewrite the `Diagnostics` class from Project 1 so it gets its answer by prompting Gemini instead of running AIMA's `enumeration_ask`. Same network, same inputs, same outputs, same GUI.

Project 1 (reference and source of truth): https://github.com/yyuqi1010/CPSC481-project-1

## Files to start with

| File | Source | Notes |
|------|--------|-------|
| `diagnostics_gui.py` | Project 1 repo / Canvas | Do not modify |
| `diagnostics.py` | Canvas skeleton | Replace the body; remove `from aima.probability import *` (AIMA is not used here) |
| `expected_answers.csv` | Generated from Project 1 | 81 rows: every input combo with the correct disease and probability |
| Project 1 `diagnostics.py` | Project 1 repo | Copy the CPT numbers from here into the prompt |

## Hard requirements (from the spec)

- File `diagnostics.py`, class `Diagnostics`, both case-sensitive.
- Constructor takes no parameters.
- One public method: `diagnose(self, visit_to_asia, smoking, xray_result, dyspnea)`.
- Inputs are strings:
  - `visit_to_asia`: `Yes`, `No`, `NA`
  - `smoking`: `Yes`, `No`, `NA`
  - `xray_result`: `Abnormal`, `Normal`, `NA`
  - `dyspnea`: `Present`, `Absent`, `NA`
  - `NA` means unobserved (not evidence).
- Returns a **list** of two items: `[disease, probability]`.
  - `disease` is exactly `"TB"`, `"Cancer"`, or `"Bronchitis"`.
  - `probability` is a float in [0, 1].
- Include comments and `print` statements showing intermediate steps (prompt sent, raw response, parsed result).
- **No API key in the code.** Read it from the `GEMINI_API_KEY` environment variable.

## The network (CPTs, P(node = True | parents))

```
Asia:        P = 0.01
Smoking:     P = 0.5
TB | Asia:            T: 0.05   F: 0.01
Cancer | Smoking:     T: 0.10   F: 0.01
Bronchitis | Smoking: T: 0.60   F: 0.30
TBorC | TB, Cancer:   (T,T): 1.0  (T,F): 1.0  (F,T): 1.0  (F,F): 0.0
Xray | TBorC:         T: 0.99   F: 0.05      (Xray True = Abnormal)
Dyspnea | TBorC, Bronchitis: (T,T): 0.9  (T,F): 0.7  (F,T): 0.8  (F,F): 0.1
```

Known check: `diagnose("No", "Yes", "Abnormal", "Absent")` returns `["Cancer", ~0.4367]`.

## Tech choices

- SDK: `google-genai` (`pip install google-genai pydantic`), imported as `from google import genai`.
- Model: default `gemini-3.8-flash`, overridable with a `GEMINI_MODEL` env var. The spec suggests Gemini 2.5 Flash, but Google now limits 2.5 to accounts that used it before, so a new key may be refused.
- **Before coding, confirm the current SDK call shape** (Context7 or ai.google.dev). Recent docs show structured output via `client.interactions.create(..., response_format={...})`; older code uses `client.models.generate_content(..., config={"response_mime_type": ..., "response_schema": ...})`. Use whichever the installed version supports and note it in a comment.

## Design

1. **Constructor**
   - Create the Gemini client. If `GEMINI_API_KEY` is missing, raise a clear error explaining how to set it.
   - Build the static part of the prompt (network description) once and store it.

2. **Input validation**
   - Map each argument to an allowed value set; raise `ValueError` on anything else (same behavior as Project 1).

3. **Prompt**
   - Describe the network as structure plus CPTs (the block above, in plain text).
   - State the evidence, listing only observed variables, and say unobserved ones must be summed out.
   - Ask for the posterior P(TB=True | e), P(Cancer=True | e), P(Bronchitis=True | e) by exact inference (enumeration over hidden variables), then the most likely disease and its probability.
   - Keep the model's reasoning on (thinking) so the arithmetic is reliable; keep temperature low.

4. **Structured output**
   - Pydantic schema:
     ```python
     class Diagnosis(BaseModel):
         disease: Literal["TB", "Cancer", "Bronchitis"]
         probability: float
     ```
   - If accuracy is poor, try adding the three posteriors as fields (`p_tb`, `p_cancer`, `p_bronchitis`) and picking the max in Python. Compare both against the answer key and keep the better one.

5. **Post-processing**
   - Validate the parsed result: disease in the allowed set, probability in [0, 1].
   - Retry up to 2 times on a parse or validation failure, then raise a clear error.
   - Return `[disease, float(probability)]`.

6. **Print statements**: inputs, evidence dict, model name, raw response text, parsed result.

## Testing

1. **Offline unit tests** (`test_diagnostics.py`, no network): mock the client and check input validation, prompt contents (evidence included, NA excluded), parsing, retry behavior, and return type.
2. **Live eval script** (`eval_against_key.py`): run `diagnose` on rows from `expected_answers.csv`, report disease accuracy and mean / max absolute probability error. Add a `--limit N` flag and a delay between calls to stay inside free-tier rate limits. Start with the spec case plus ~10 rows covering all three diseases.
3. **Target**: correct disease on every tested row and probability within 0.01 of the key. Iterate on the prompt until this holds.

## Deliverables (Gradescope, all three at once, include all group members)

1. `diagnostics.py` with no API key in it.
2. Screenshot (PNG/JPG) of the GUI or terminal output while running.
3. Short PDF report:
   - (a) Why this LLM (1-2 sentences)
   - (b) Major challenge (1-4 sentences)
   - (c) Something surprising about LLMs (1-3 sentences)
   Draft these from what actually happens during testing (e.g., model access issues, arithmetic errors, structured output).

## Milestones

- [x] Set up project folder, copy GUI, skeleton, and `expected_answers.csv`; add `.gitignore` (include `.env`)
- [x] Confirm SDK call shape for structured output with the installed `google-genai` version
- [x] Implement `Diagnostics` with validation, prompt, structured output, retries, prints
- [x] Offline unit tests pass
- [ ] Live spec case returns Cancer ~0.4367
- [ ] Eval script passes on the sample, then on as many rows as rate limits allow
- [ ] Run the GUI and take the screenshot
- [ ] Draft the report answers
- [ ] Final check: no key in any file, names match the spec exactly

## Open questions for the user

- New repo, or a `project2/` folder in the Project 1 repo?
- Has the API key been created, and does `gemini-2.5-flash` work on it?
- Same group as Project 1?
