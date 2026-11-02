"""Tuned Fourier-invariant Hermite-Gaussian test function.

Maximizes the minimum of g(lambda) over lambda>=0 (equivalently minimizes the
upper bound C4 = max (1-g)/g = 1/g_min - 1) using an epigraph formulation
with smooth constraints, then polishes with direct minimization.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_DEADLINE = 210.0
_MARGIN = 20.0
_START = None


def _hermite_values(x):
    """Vectorised physicists' Hermite polynomials H_0,H_4,H_8,H_12 at points x."""
    x = np.asarray(x, dtype=float)
    h0 = np.ones_like(x)
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    x12 = x6 * x6
    h4 = 16.0 * x4 - 48.0 * x2 + 12.0
    h8 = 256.0 * x8 - 3584.0 * x6 + 13440.0 * x4 - 13440.0 * x2 + 1680.0
    h12 = (4096.0 * x12 - 212992.0 * x10 + 3317760.0 * x8
           - 25333440.0 * x6 + 100810080.0 * x4
           - 201620160.0 * x2 + 798336.0)
    return h0, h4, h8, h12


def _g_of_lambda(a, lam):
    """g(lambda) = (a0 H0 + a4 H4 + a8 H8 + a12 H12)(lambda) * exp(-lambda^2/2),
    with a12 chosen so g(0)=0."""
    a0, a4, a8 = a
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 798336.0
    h0, h4, h8, h12 = _hermite_values(lam)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-0.5 * lam * lam)


def _bound_exact(a, grid=None):
    """Exact max of (1-g)/g over lambda>=0 with adaptive refinement."""
    if grid is None:
        grid = np.concatenate([
            np.linspace(0.0, 5.0, 50001),
            np.linspace(5.0, 15.0, 30000),
            np.linspace(15.0, 40.0, 10000),
            np.geomspace(1e-8, 0.01, 1000),
        ])
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-14, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    idx = int(np.argmax(ratio))
    val = float(ratio[idx])
    x0 = float(grid[idx])
    for level in range(3):
        half = 0.02 / (level + 1)
        lo = max(0.0, x0 - half)
        hi = x0 + half
        local = np.linspace(lo, hi, 2001)
        gl = _g_of_lambda(a, local)
        with np.errstate(divide="ignore", invalid="ignore"):
            rl = (1.0 - gl) / gl
        rl = np.where(gl > 1e-14, rl, 1e18)
        rl = np.where(np.isfinite(rl), rl, 1e18)
        idx2 = int(np.argmax(rl))
        val = max(val, float(rl[idx2]))
        x0 = float(local[idx2])
    return val


def _min_g(a, grid=None):
    """Minimum of g(lambda) over lambda>=0."""
    if grid is None:
        grid = np.concatenate([
            np.linspace(0.0, 5.0, 50001),
            np.linspace(5.0, 15.0, 30000),
            np.linspace(15.0, 40.0, 10000),
        ])
    g = _g_of_lambda(a, grid)
    return float(np.min(g))


def _objective_epigraph(a, grid, beta=50.0):
    """Epigraph objective: -t where t is soft-min of g values.
    Uses log-sum-exp in the negative domain: soft_min(g) = -soft_max(-g)
    = -(-beta * log(mean(exp(-g/beta)))) = beta * log(mean(exp(-g/beta)))
    Actually: soft_min(g_i) ≈ -log-sum-exp(-g_i/beta) * beta
    Let's use: soft_min = -beta * log(mean(exp(-g/beta)))
    """
    g = _g_of_lambda(a, grid)
    # soft minimum: -beta * log(mean(exp(-g/beta)))
    neg_g_over_beta = -g / beta
    m = np.max(neg_g_over_beta)
    s = np.exp(neg_g_over_beta - m)
    soft_min_val = -beta * (m + math.log(float(np.mean(s))))
    # We want to maximize soft_min, so minimize -soft_min
    return -soft_min_val


def _objective_direct(a, grid, beta=30.0):
    """Direct objective: max (1-g)/g approximated via soft-max.
    soft_max(ratio_i) = beta * log(mean(exp(ratio_i/beta)))
    """
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-12, ratio, 1e15)
    ratio = np.where(np.isfinite(ratio), ratio, 1e15)
    ratio = np.clip(ratio, -10.0, 1e12)
    # soft max
    r_over_beta = ratio / beta
    m = np.max(r_over_beta)
    s = np.exp(r_over_beta - m)
    soft_max_val = beta * (m + math.log(float(np.mean(s))))
    return soft_max_val


def _objective_grad(a, grid, beta=40.0):
    """Objective with analytical gradient for the epigraph formulation.
    soft_min(g) = -beta * log(mean(exp(-g/beta)))
    d(soft_min)/da_j = sum_i (g_i/beta * exp(-g_i/beta)) / mean(exp(-g_i/beta)) * (dg_i/da_j)
    Actually: soft_min = -beta * log(mean(exp(-g/beta)))
    d(soft_min)/da_j = -beta * (1/mean) * sum(exp(-g_i/beta) * (-1/beta) * dg_i/da_j)
                      = sum(exp(-g_i/beta) * dg_i/da_j) / mean(exp(-g_i/beta))
    
    We minimize -soft_min, so gradient of objective = -gradient of soft_min
    """
    g = _g_of_lambda(a, grid)
    neg_g_over_beta = -g / beta
    m = np.max(neg_g_over_beta)
    exp_vals = np.exp(neg_g_over_beta - m)
    mean_exp = float(np.mean(exp_vals))
    weights = exp_vals / mean_exp  # softmax weights
    
    # dg/da for each coefficient
    # g = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-lam^2/2)
    # a12 = -(a0 + 12*a4 + 1680*a8)/798336
    # da12/da0 = -1/798336, da12/da4 = -12/798336, da12/da8 = -1680/798336
    
    a0, a4, a8 = a
    h0, h4, h8, h12 = _hermite_values(grid)
    exp_factor = np.exp(-0.5 * grid * grid)
    
    # dg/da0 = (H0 + da12/da0 * H12) * exp_factor = (H0 - H12/798336) * exp_factor
    # dg/da4 = (H4 + da12/da4 * H12) * exp_factor = (H4 - 12*H12/798336) * exp_factor
    # dg/da8 = (H8 + da12/da8 * H12) * exp_factor = (H8 - 1680*H12/798336) * exp_factor
    
    dg_da0 = (h0 - h12 / 798336.0) * exp_factor
    dg_da4 = (h4 - 12.0 * h12 / 798336.0) * exp_factor
    dg_da8 = (h8 - 1680.0 * h12 / 798336.0) * exp_factor
    
    # gradient of soft_min w.r.t. a_j = sum_i weights_i * dg_i/da_j
    grad_soft_min = np.array([
        float(np.dot(weights, dg_da0)),
        float(np.dot(weights, dg_da4)),
        float(np.dot(weights, dg_da8)),
    ])
    
    # We minimize -soft_min, so objective gradient = -grad_soft_min
    soft_min_val = -beta * (m + math.log(mean_exp))
    return -soft_min_val, -grad_soft_min


def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    best_a = _SEED.copy()
    best_val = _bound_exact(_SEED)
    
    def remaining():
        return _DEADLINE - _MARGIN - (time.time() - _START)
    
    from scipy.optimize import minimize, differential_evolution
    
    # Grid for optimization (dense enough for smooth surrogate)
    grid_opt = np.concatenate([
        np.linspace(0.001, 5.0, 20000),
        np.linspace(5.0, 12.0, 10000),
    ])
    
    # --- Phase 1: Global search with differential evolution on smooth surrogate ---
    if remaining() > 50.0:
        try:
            bounds = [(-3.0, 0.5), (-0.5, 0.5), (-0.01, 0.01)]
            result_de = differential_evolution(
                lambda x: _objective_epigraph(x, grid_opt, beta=50.0),
                bounds=bounds,
                seed=42,
                maxiter=200,
                tol=1e-8,
                popsize=25,
                polish=True,
                init="sobol",
            )
            cand = np.asarray(result_de.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    # Second DE with different parameters
    if remaining() > 40.0:
        try:
            bounds = [(-3.0, 0.5), (-0.5, 0.5), (-0.01, 0.01)]
            result_de2 = differential_evolution(
                lambda x: _objective_direct(x, grid_opt, beta=30.0),
                bounds=bounds,
                seed=123,
                maxiter=150,
                tol=1e-8,
                popsize=20,
                polish=True,
            )
            cand = np.asarray(result_de2.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    # Third DE with tighter bounds around seed
    if remaining() > 35.0:
        try:
            bounds = [(-1.5, -0.8), (-0.2, 0.2), (-0.005, 0.005)]
            result_de3 = differential_evolution(
                lambda x: _objective_epigraph(x, grid_opt, beta=80.0),
                bounds=bounds,
                seed=777,
                maxiter=200,
                tol=1e-9,
                popsize=30,
                polish=True,
            )
            cand = np.asarray(result_de3.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    # --- Phase 2: Multi-start local optimization with gradients ---
    rng = np.random.default_rng(0)
    starts = [best_a.copy(), _SEED.copy()]
    for scale in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        starts.append(best_a + scale * rng.standard_normal(3))
        starts.append(_SEED + scale * rng.standard_normal(3))
    for i in range(3):
        for sign in (-1, 1):
            for scale in (0.01, 0.03, 0.1, 0.3):
                p = best_a.copy()
                p[i] += sign * scale
                starts.append(p)
    
    # Hand-picked starts
    hand_picks = [
        [-1.13, 0.04, 0.0003],
        [-1.10, 0.035, 0.0004],
        [-1.15, 0.045, 0.0002],
        [-1.05, 0.03, 0.0005],
        [-1.20, 0.05, 0.0001],
        [-1.00, 0.025, 0.0006],
        [-1.08, 0.038, 0.00035],
        [-1.12, 0.042, 0.00025],
        [-1.18, 0.048, 0.00015],
        [-1.25, 0.055, 0.00005],
    ]
    for hp in hand_picks:
        starts.append(np.array(hp, dtype=float))
    
    # L-BFGS-B with gradient on epigraph objective
    for temp_beta in (80.0, 50.0, 30.0, 20.0):
        for s in starts:
            if remaining() < 5.0:
                break
            try:
                res = minimize(
                    lambda x: _objective_epigraph(x, grid_opt, beta=temp_beta),
                    s,
                    jac=lambda x: _objective_grad(x, grid_opt, beta=temp_beta)[1],
                    method="L-BFGS-B",
                    options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-12},
                )
            except Exception:
                continue
            cand = np.asarray(res.x, dtype=float)
            if not np.all(np.isfinite(cand)):
                continue
            v = _bound_exact(cand)
            if v < best_val:
                best_val = v
                best_a = cand.copy()
    
    # --- Phase 3: High-precision polish on exact objective ---
    if remaining() > 20.0:
        try:
            res = minimize(
                _bound_exact, best_a, method="L-BFGS-B",
                options={"maxiter": 1000, "ftol": 1e-16, "gtol": 1e-14},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    # Second polish from perturbed point
    if remaining() > 15.0:
        for pert_dir in (np.array([1.0, -1.0, 0.5]), np.array([-0.5, 1.0, 1.0]), np.array([0.5, 0.5, -1.0])):
            if remaining() < 5.0:
                break
            try:
                pert = best_a + 0.005 * pert_dir
                res2 = minimize(
                    _bound_exact, pert, method="L-BFGS-B",
                    options={"maxiter": 800, "ftol": 1e-16, "gtol": 1e-14},
                )
                cand = np.asarray(res2.x, dtype=float)
                if np.all(np.isfinite(cand)):
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
            except Exception:
                pass
    
    # --- Phase 4: Fine local search ---
    if remaining() > 10.0:
        for step in (0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001):
            if remaining() < 3.0:
                break
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    for k in (-1, 0, 1):
                        if remaining() < 1.0:
                            break
                        cand = best_a + step * np.array([i, j, k], dtype=float)
                        v = _bound_exact(cand)
                        if v < best_val:
                            best_val = v
                            best_a = cand.copy()
    
    # --- Phase 5: Nelder-Mead polish ---
    if remaining() > 5.0:
        try:
            res_nm = minimize(
                _bound_exact, best_a, method="Nelder-Mead",
                options={"maxiter": 1000, "xatol": 1e-12, "fatol": 1e-15, "adaptive": True},
            )
            cand = np.asarray(res_nm.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    # --- Phase 6: Final coordinate descent ---
    if remaining() > 3.0:
        for _ in range(50):
            if remaining() < 1.0:
                break
            improved = False
            for dim in range(3):
                for delta in (0.001, -0.001, 0.0005, -0.0005, 0.0001, -0.0001):
                    cand = best_a.copy()
                    cand[dim] += delta
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
                        improved = True
            if not improved:
                break
    
    return np.array(best_a, dtype=float)
# EVOLVE-BLOCK-END
