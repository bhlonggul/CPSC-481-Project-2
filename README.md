# CPSC-481-Project-2

Bayesian network diagnosis with Gemini. `diagnostics.py` gives Gemini the Asia network and the evidence, has it do the exact inference that AIMA's `enumeration_ask` did in Project 1, and returns `[disease, probability]`. `diagnostics_gui.py` is the unchanged Project 1 GUI.

## Setup

```
pip install -r requirements.txt
```

Get a key at https://aistudio.google.com/apikey. Set it in the terminal you run from; it never goes in the code:

```
# PowerShell
$env:GEMINI_API_KEY = "your-key"
# bash
export GEMINI_API_KEY="your-key"
```

You can also keep `GEMINI_API_KEY=your-key` in a `.env` file, which git ignores. Load it into a PowerShell session with:

```
Get-Content .env | ForEach-Object { if ($_ -match '^\s*(GEMINI_[A-Z_]+)\s*=\s*(.*?)\s*$') { Set-Item "env:$($Matches[1])" $Matches[2].Trim('"', "'") } }
```

`GEMINI_MODEL` selects a different model; the default is `gemini-3.8-flash`.

## Run

- GUI: `python diagnostics_gui.py`. After each Diagnose click, expect a 40-55 s wait while Gemini thinks.
- Offline tests (no key or network needed): `python -m unittest -v`
- Live check against `expected_answers.csv`: `python eval_against_key.py` runs the 11-row sample. Add `--all` for all 81 rows, or see `--help`.
