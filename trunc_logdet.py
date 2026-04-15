import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    import cvxpy as cp
except Exception:
    cp = None

Array = np.ndarray


def project_interval(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


# ---------------------------
# Part 1: validate the four-case explicit formula against CVXPY
# ---------------------------

def trunc_objective(alpha: float, alpha_k: float, h_k: float, d_k: float, hbar_k: float, eta_k: float) -> float:
    t = alpha / alpha_k
    return max(h_k + d_k * (alpha - alpha_k), hbar_k) + (1.0 / eta_k) * (t - math.log(t) - 1.0)


def trunc_logdet_explicit(alpha_k: float, h_k: float, d_k: float, hbar_k: float, eta_k: float,
                          alpha_lo: float, alpha_hi: float) -> float:
    if abs(d_k) <= 1e-14:
        return project_interval(alpha_k, alpha_lo, alpha_hi)

    b_k = alpha_k + (hbar_k - h_k) / d_k

    if d_k > 0.0:
        cand = alpha_k / (1.0 + eta_k * alpha_k * d_k)
        lo = max(alpha_lo, b_k)
        hi = alpha_hi
        if lo > hi:
            return project_interval(lo, alpha_lo, alpha_hi)
        return project_interval(cand, lo, hi)

    denom = 1.0 + eta_k * alpha_k * d_k
    if denom > 0.0:
        cand = alpha_k / denom
        lo = alpha_lo
        hi = min(alpha_hi, b_k)
        if lo > hi:
            return project_interval(hi, alpha_lo, alpha_hi)
        return project_interval(cand, lo, hi)

    return project_interval(min(alpha_hi, b_k), alpha_lo, alpha_hi)


def trunc_logdet_cvxpy(alpha_k: float, h_k: float, d_k: float, hbar_k: float, eta_k: float,
                       alpha_lo: float, alpha_hi: float) -> float:
    if cp is None:
        raise RuntimeError("cvxpy is not installed")

    # Solve in t = alpha / alpha_k coordinates for better numerical scaling.
    t = cp.Variable(pos=True)
    t_lo = alpha_lo / alpha_k
    t_hi = alpha_hi / alpha_k
    obj = cp.maximum(h_k + alpha_k * d_k * (t - 1.0), hbar_k) + (1.0 / eta_k) * (t - cp.log(t) - 1.0)
    prob = cp.Problem(cp.Minimize(obj), [t >= t_lo, t <= t_hi])
    last_err = None
    for solver in [cp.CLARABEL, cp.SCS]:
        try:
            if solver == cp.SCS:
                prob.solve(solver=solver, eps=1e-10, max_iters=50000, verbose=False)
            else:
                prob.solve(solver=solver, verbose=False)
            if t.value is not None:
                return float(alpha_k * t.value)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"cvxpy solve failed: {last_err}")


CASE_INFO = {
    1: ("case 1: d=0", "tab:blue"),
    2: ("case 2: d>0", "tab:orange"),
    3: ("case 3: d<0, denom>0", "tab:green"),
    4: ("case 4: d<0, denom<=0", "tab:red"),
}


def classify_case(alpha_k: float, d_k: float, eta_k: float, tol: float = 1e-14) -> int:
    if abs(d_k) <= tol:
        return 1
    if d_k > 0.0:
        return 2
    if 1.0 + eta_k * alpha_k * d_k > 0.0:
        return 3
    return 4


def sample_test_instance(rng: np.random.Generator, target_case: int, alpha_hi: float = 1e2):
    alpha_lo = rng.uniform(1e-3, 2e-1)
    alpha_k = 10 ** rng.uniform(math.log10(alpha_lo) + 0.05, math.log10(alpha_hi) - 0.05)
    eta_k = 10 ** rng.uniform(-1.5, 1.0)
    h_k = rng.uniform(-3.0, 1.0)
    hbar_k = h_k - rng.uniform(0.0, 3.0)

    if target_case == 1:
        d_k = 0.0
    elif target_case == 2:
        d_k = rng.uniform(1e-6, 10.0)
    elif target_case == 3:
        upper = 0.98 / max(alpha_k * eta_k, 1e-12)
        upper = max(upper, 2e-6)
        d_k = -rng.uniform(1e-6, upper)
    elif target_case == 4:
        lower = 1.02 / max(alpha_k * eta_k, 1e-12)
        d_k = -rng.uniform(lower, lower + 10.0)
    else:
        raise ValueError(target_case)
    return alpha_k, h_k, d_k, hbar_k, eta_k, alpha_lo, alpha_hi


def validate_explicit_formula(n_tests: int = 250, seed: int = 3, alpha_hi: float = 1e2):
    rng = np.random.default_rng(seed)
    rows = []
    case_order = [1, 2, 3, 4]
    for j in range(n_tests):
        target_case = case_order[j % 4]
        alpha_k, h_k, d_k, hbar_k, eta_k, alpha_lo, alpha_hi = sample_test_instance(rng, target_case, alpha_hi)
        case_id = classify_case(alpha_k, d_k, eta_k)
        a_exp = trunc_logdet_explicit(alpha_k, h_k, d_k, hbar_k, eta_k, alpha_lo, alpha_hi)
        a_cvx = trunc_logdet_cvxpy(alpha_k, h_k, d_k, hbar_k, eta_k, alpha_lo, alpha_hi)
        abs_err = abs(a_exp - a_cvx)
        rel_err = abs_err / max(1.0, abs(a_cvx))
        obj_exp = trunc_objective(a_exp, alpha_k, h_k, d_k, hbar_k, eta_k)
        obj_cvx = trunc_objective(a_cvx, alpha_k, h_k, d_k, hbar_k, eta_k)
        obj_gap = abs(obj_exp - obj_cvx)
        rows.append((a_cvx, a_exp, abs_err, rel_err, obj_gap, alpha_k, h_k, hbar_k, d_k, eta_k, alpha_lo, alpha_hi, case_id))
    return np.array(rows, dtype=float)


# ---------------------------
# Part 2: quadratic comparison
# ---------------------------

def make_rotated_quadratic(seed: int = 4):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    H = U @ np.diag([1.0, 80.0]) @ U.T
    b = np.array([0.5, -0.25])
    x_star = -np.linalg.solve(H, b)

    def f(x: Array) -> float:
        return float(0.5 * x @ H @ x + b @ x)

    def g(x: Array) -> Array:
        return H @ x + b

    fstar = f(x_star)
    return f, g, H, b, x_star, fstar


@dataclass
class QuadHistory:
    name: str
    gaps: Array
    alphas: Array
    accepts: Array


def normalized_h_quadratic(H: Array, x: Array, b: Array, P: Array, alpha: float) -> float:
    g = H @ x + b
    Pg = P @ g
    denom = float(g @ Pg)
    if denom <= 1e-20:
        return 0.0
    return (-alpha * denom + 0.5 * alpha * alpha * float(Pg @ (H @ Pg))) / denom


def normalized_hprime_quadratic(H: Array, x: Array, b: Array, P: Array, alpha: float) -> float:
    g = H @ x + b
    denom = float(g @ (P @ g))
    if denom <= 1e-20:
        return -1.0
    x_probe = x - alpha * (P @ g)
    g_probe = H @ x_probe + b
    return -float(g_probe @ (P @ g)) / denom


def effective_curvature(H: Array, x: Array, b: Array, P: Array) -> float:
    g = H @ x + b
    Pg = P @ g
    denom = float(g @ Pg)
    if denom <= 1e-20:
        return 0.0
    return float(Pg @ (H @ Pg)) / denom


def exact_hbar_quadratic(H: Array, x: Array, b: Array, P: Array, alpha_lo: float, alpha_hi: float) -> float:
    lam_eff = effective_curvature(H, x, b, P)
    if lam_eff <= 1e-18:
        alpha_star = alpha_hi
    else:
        alpha_star = project_interval(1.0 / lam_eff, alpha_lo, alpha_hi)
    return normalized_h_quadratic(H, x, b, P, alpha_star)


def exact_quadratic_root_unprojected(alpha_k: float, lam_eff: float, eta_k: float) -> float:
    if lam_eff <= 1e-18:
        denom = 1.0 / alpha_k - eta_k
        if denom <= 1e-18:
            return math.inf
        return 1.0 / denom
    bcoef = 1.0 / (eta_k * alpha_k) - 1.0
    disc = bcoef * bcoef + 4.0 * lam_eff / eta_k
    return (-bcoef + math.sqrt(disc)) / (2.0 * lam_eff)


def exact_quadratic_logdet_update(alpha_k: float, lam_eff: float, eta_k: float, alpha_lo: float, alpha_hi: float) -> float:
    root = exact_quadratic_root_unprojected(alpha_k, lam_eff, eta_k)
    if not math.isfinite(root):
        return alpha_hi
    return project_interval(root, alpha_lo, alpha_hi)


def basic_logdet_fixed_eta_update(alpha_k: float, d: float, eta_k: float, alpha_lo: float, alpha_hi: float) -> float:
    denom = 1.0 + eta_k * alpha_k * d
    if denom <= 1e-18:
        return alpha_hi
    return project_interval(alpha_k / denom, alpha_lo, alpha_hi)


def linearized_truncated_exactlb_update(alpha_k: float, h_k: float, d: float, hbar_exact: float,
                                        eta_k: float, alpha_lo: float, alpha_hi: float) -> float:
    return trunc_logdet_explicit(alpha_k, h_k, d, hbar_exact, eta_k, alpha_lo, alpha_hi)


def run_quadratic_compare(P: Array, x0: Array, K: int = 60, eta_const: float = 1.0, alpha_hi_override: float = 1e2):
    f, gfun, H, b, x_star, fstar = make_rotated_quadratic(seed=4)
    evals, evecs = np.linalg.eigh(P)
    Psqrt = evecs @ np.diag(np.sqrt(np.maximum(evals, 1e-15))) @ evecs.T
    Lp = float(np.max(np.linalg.eigvalsh(Psqrt @ H @ Psqrt)))
    alpha_lo = 1.0 / Lp
    alpha_hi = float(alpha_hi_override)
    alpha0 = project_interval(1.25 / Lp, alpha_lo, alpha_hi)

    methods = ["basic-logdet", "truncated-linearized-logdet", "exact-quadratic-logdet"]
    states = {m: x0.astype(float).copy() for m in methods}
    alphas_cur = {m: alpha0 for m in methods}
    histories = {m: dict(gaps=[max(f(states[m]) - fstar, 1e-16)], alphas=[alpha0], accepts=[1]) for m in methods}
    alpha_gap_lin_exact = [0.0]

    for _ in range(K):
        for method in methods:
            x = states[method]
            alpha = alphas_cur[method]
            g = gfun(x)
            if np.linalg.norm(g) <= 1e-14:
                histories[method]["gaps"].append(max(f(x) - fstar, 1e-16))
                histories[method]["alphas"].append(alpha)
                histories[method]["accepts"].append(0)
                continue
            d = normalized_hprime_quadratic(H, x, b, P, alpha)
            if method == "basic-logdet":
                alpha_new = basic_logdet_fixed_eta_update(alpha, d, eta_const, alpha_lo, alpha_hi)
            elif method == "truncated-linearized-logdet":
                h_k = normalized_h_quadratic(H, x, b, P, alpha)
                hbar_exact = exact_hbar_quadratic(H, x, b, P, alpha_lo, alpha_hi)
                alpha_new = linearized_truncated_exactlb_update(alpha, h_k, d, hbar_exact, eta_const, alpha_lo, alpha_hi)
            else:
                lam_eff = effective_curvature(H, x, b, P)
                alpha_new = exact_quadratic_logdet_update(alpha, lam_eff, eta_const, alpha_lo, alpha_hi)
            x_trial = x - alpha * (P @ g)
            if f(x_trial) <= f(x):
                states[method] = x_trial
                acc = 1
            else:
                acc = 0
            alphas_cur[method] = float(alpha_new)
            histories[method]["gaps"].append(max(f(states[method]) - fstar, 1e-16))
            histories[method]["alphas"].append(alphas_cur[method])
            histories[method]["accepts"].append(acc)
        alpha_gap_lin_exact.append(abs(alphas_cur["truncated-linearized-logdet"] - alphas_cur["exact-quadratic-logdet"]))

    hist = {m: QuadHistory(m, np.array(histories[m]["gaps"]), np.array(histories[m]["alphas"]), np.array(histories[m]["accepts"])) for m in methods}
    return hist, np.array(alpha_gap_lin_exact), alpha_lo, alpha_hi


# ---------------------------
# Plotting / outputs (show directly; do not write files or summaries)
# ---------------------------

def make_validation_figure(rows: np.ndarray):
    a_cvx = rows[:, 0]
    a_exp = rows[:, 1]
    abs_err = rows[:, 2]
    obj_gap = rows[:, 4]
    case_ids = rows[:, 12].astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    mn = min(a_cvx.min(), a_exp.min())
    mx = max(a_cvx.max(), a_exp.max())
    for cid in [1, 2, 3, 4]:
        mask = case_ids == cid
        label, color = CASE_INFO[cid]
        axes[0].scatter(a_cvx[mask], a_exp[mask], s=18, alpha=0.8, color=color, label=f"{label} (n={int(mask.sum())})")
    axes[0].plot([mn, mx], [mn, mx], 'k--', linewidth=1)
    axes[0].set_xlabel('CVXPY solution')
    axes[0].set_ylabel('explicit formula')
    axes[0].set_title('Explicit formula vs CVXPY, colored by case')
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    order = np.argsort(abs_err)
    ordered_err = abs_err[order]
    ordered_case = case_ids[order]
    for cid in [1, 2, 3, 4]:
        mask = ordered_case == cid
        label, color = CASE_INFO[cid]
        xs = np.arange(len(order))[mask]
        ys = ordered_err[mask]
        axes[1].scatter(xs, ys, s=18, alpha=0.8, color=color, label=label)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('sorted test index')
    axes[1].set_ylabel('absolute $\\alpha$-error')
    axes[1].set_title('Absolute $\\alpha$-error across tests')
    axes[1].grid(True, alpha=0.25)

    order_obj = np.argsort(obj_gap)
    ordered_obj = obj_gap[order_obj]
    ordered_case_obj = case_ids[order_obj]
    for cid in [1, 2, 3, 4]:
        mask = ordered_case_obj == cid
        label, color = CASE_INFO[cid]
        xs = np.arange(len(order_obj))[mask]
        ys = ordered_obj[mask]
        axes[2].scatter(xs, ys, s=18, alpha=0.8, color=color, label=label)
    axes[2].set_yscale('log')
    axes[2].set_xlabel('sorted test index')
    axes[2].set_ylabel('objective gap')
    axes[2].set_title('Objective-value discrepancy across tests')
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    plt.show()


def make_quadratic_figure(hist_I, hist_diag, alpha_gap_I, alpha_gap_diag):
    K = len(next(iter(hist_I.values())).gaps) - 1
    ks = np.arange(K + 1)
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 9.0))

    style_map = {
        "basic-logdet": dict(color="tab:green", linestyle=(0, (3, 1, 1, 1)), linewidth=2.0),
        "truncated-linearized-logdet": dict(color="tab:blue", linestyle="--", linewidth=2.0),
        "exact-quadratic-logdet": dict(color="tab:orange", linestyle="-", linewidth=2.0),
    }

    for name, h in hist_I.items():
        style = style_map.get(name, {})
        axes[0, 0].semilogy(ks, h.gaps, label=name, **style)
        axes[1, 0].plot(ks, h.alphas, label=name, **style)
    for name, h in hist_diag.items():
        style = style_map.get(name, {})
        axes[0, 1].semilogy(ks, h.gaps, label=name, **style)
        axes[1, 1].plot(ks, h.alphas, label=name, **style)

    axes[2, 0].semilogy(ks, np.maximum(alpha_gap_I, 1e-18), color="black", linewidth=2.0,
                        label=r"$|\alpha_k^{\rm lin}-\alpha_k^{\rm ex}|$")
    axes[2, 1].semilogy(ks, np.maximum(alpha_gap_diag, 1e-18), color="black", linewidth=2.0,
                        label=r"$|\alpha_k^{\rm lin}-\alpha_k^{\rm ex}|$")

    axes[0, 0].set_title(r'Quadratic, fixed $P=I$')
    axes[0, 1].set_title(r'Quadratic, fixed $P=\mathrm{Diag}(H)^{-1}$')

    for ax in axes[0]:
        ax.set_xlabel('iteration $k$')
        ax.set_ylabel('objective gap')
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel('iteration $k$')
        ax.set_ylabel(r'$\alpha_k$')
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes[2]:
        ax.set_xlabel('iteration $k$')
        ax.set_ylabel(r'$|\alpha_k^{\rm lin}-\alpha_k^{\rm ex}|$')
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle('Quadratic experiment: basic log-det vs truncated linearized vs exact', y=0.995)
    fig.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Validation and quadratic comparison for log-det truncated models.')
    parser.add_argument('--n_tests', type=int, default=250)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--K', type=int, default=60)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--alpha_hi', type=float, default=1e2)
    parser.add_argument('--skip_validation', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if (not args.skip_validation) and (cp is not None):
        rows = validate_explicit_formula(n_tests=args.n_tests, seed=args.seed, alpha_hi=args.alpha_hi)
        make_validation_figure(rows)

    _, _, H, _, _, _ = make_rotated_quadratic(seed=4)
    x0 = np.array([2.0, -1.5])
    P_I = np.eye(2)
    P_diag = np.diag(1.0 / np.diag(H))
    hist_I, alpha_gap_I, _, _ = run_quadratic_compare(P_I, x0, K=args.K, eta_const=args.eta, alpha_hi_override=args.alpha_hi)
    hist_diag, alpha_gap_diag, _, _ = run_quadratic_compare(P_diag, x0, K=args.K, eta_const=args.eta, alpha_hi_override=args.alpha_hi)
    make_quadratic_figure(hist_I, hist_diag, alpha_gap_I, alpha_gap_diag)


if __name__ == '__main__':
    main()
