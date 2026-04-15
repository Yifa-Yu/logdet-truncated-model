# Log-det truncated model experiments

This repo contains code for the truncated log-det model in the fixed-`P` setting.

What the script does:

1. **Validate the four-case closed form** for the truncated linearized log-det update against a CVXPY solve.
2. **Compare three scalar-scale update rules on a quadratic**:
   - `basic-logdet` (plain fixed-η log-det, no truncation),
   - `truncated-linearized-logdet` (linearized `h` + exact interval lower-level truncation),
   - `exact-quadratic-logdet` (non-linearized exact quadratic subproblem).

## Install

```bash
pip install -r requirements.txt
```

## Run

The script **shows plots directly** and **does not write summary.txt**.

```bash
python code/trunc_logdet_quadratic_compare.py
```

Optional arguments:

```bash
python code/trunc_logdet_quadratic_compare.py \
  --eta 1.0 \
  --alpha_hi 1e2 \
  --K 60 \
  --n_tests 250
```

Skip the validation panel if `cvxpy` is unavailable:

```bash
python code/trunc_logdet_quadratic_compare.py --skip_validation
```
