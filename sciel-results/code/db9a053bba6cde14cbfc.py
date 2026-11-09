"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

STRUCTURAL CHANGE:
  1. Modified layer sequence to end with 'L' (low index) to minimize the index
     mismatch at the coating-substrate interface (1.38 vs 1.52 is better than
     2.35 vs 1.52).
  2. Multi-start Nelder-Mead optimization in log-thickness space with diverse
     initializations (quarter-wave, random, graded).
  3. L-BFGS-B refinement for the best candidate.
  4. Robust deadline handling.
"""

import time
import math
import random
import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _mean_reflectance_fast(ns, ds, wavs, n_0, n_s):
    """Fast version with precomputed wavelengths and index array."""
    n_ds = ns * ds
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)

    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)

    i = 1.0j
    for j in range(len(ns)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + i * B * n_j * sj
        B_new = i * A * sj / n_j + B * cj
        C_new = C * cj + i * D * n_j * sj
        D_new = i * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)


def _nelder_mead(f, x0, lo, hi, deadline, max_iters=500):
    """Nelder-Mead local optimization with box constraints."""
    n = len(x0)
    best = list(x0)
    best_val = f(best)
    
    simplex = [list(x0)]
    # Initial step size: 10% of the range
    step = 0.1 * (hi - lo)
    for i in range(n):
        pt = list(x0)
        # Perturb in positive direction first
        pt[i] = min(hi[i], pt[i] + step[i])
        simplex.append(pt)
    
    scores = [f(pt) for pt in simplex]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    
    it = 0
    while it < max_iters and time.time() < deadline:
        it += 1
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        
        if scores[0] < best_val:
            best_val = scores[0]
            best = list(simplex[0])
        
        centroid = np.mean(simplex[:-1], axis=0)
        xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
        fr = f(list(xr))
        
        if fr < scores[0]:
            xe = np.clip(centroid + gamma * (centroid - simplex[-1]), lo, hi)
            fe = f(list(xe))
            if fe < fr:
                simplex[-1], scores[-1] = list(xe), fe
            else:
                simplex[-1], scores[-1] = list(xr), fr
        elif fr < scores[-2]:
            simplex[-1], scores[-1] = list(xr), fr
        else:
            xc = np.clip(centroid + rho * (simplex[-1] - centroid), lo, hi)
            fc = f(list(xc))
            if fc <= fr:
                simplex[-1], scores[-1] = list(xc), fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = np.clip(
                        np.array(simplex[0]) + sigma * (np.array(simplex[i]) - np.array(simplex[0])),
                        lo, hi)
                    scores[i] = f(list(simplex[i]))
                if max(scores) - min(scores) < 1e-9:
                    break
    
    return best, best_val


def design(problem):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    random.seed(2024)
    np.random.seed(2024)

    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]
    
    wavs = np.linspace(band[0], band[1], 401)

    # Structural Change: Ensure the last layer is 'L' to match substrate index better.
    # Pattern: L H L H ... L H L (if odd count) or L H ... H L (if even count, swap last)
    n_layers = min(20, max_layers)
    materials = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
    # If the last layer is H, change it to L to reduce interface mismatch.
    if materials[-1] == 'H':
        materials[-1] = 'L'
    
    ns = np.array([problem["materials"][m] for m in materials], dtype=float)
    
    def eval_physical(ths):
        """Evaluate physical thicknesses, applying constraints."""
        ths = np.asarray(ths, dtype=float)
        # Clip to [0, max_layer]
        ths = np.clip(ths, 0.0, max_layer)
        total = ths.sum()
        if total > max_total:
            ths = ths * (max_total / total)
            ths = np.clip(ths, 0.0, max_layer)
        return _mean_reflectance_fast(ns, ths, wavs, n_0, n_s)

    # Log-space evaluator for optimizers
    lo = np.full(n_layers, math.log(1.0))
    hi = np.full(n_layers, math.log(max_layer))
    
    def eval_log(x):
        t = np.exp(np.clip(x, lo, hi))
        total = t.sum()
        if total > max_total:
            t = t * (max_total / total)
        return _mean_reflectance_fast(ns, t, wavs, n_0, n_s)

    best_ths = None
    best_score = float("inf")

    def update_best(ths):
        nonlocal best_ths, best_score
        ths = np.asarray(ths, dtype=float)
        ths = np.clip(ths, 0.0, max_layer)
        total = ths.sum()
        if total > max_total:
            ths = ths * (max_total / total)
            ths = np.clip(ths, 0.0, max_layer)
        score = _mean_reflectance_fast(ns, ths, wavs, n_0, n_s)
        if score < best_score:
            best_score = score
            best_ths = ths.copy()

    # Generate diverse starting points
    starts = []
    
    # 1. Quarter-wave stack
    qw_ths = []
    for m in materials:
        n = problem["materials"][m]
        qw = wl_center / (4.0 * n)
        qw_ths.append(qw)
    starts.append(np.array([math.log(max(1.0, min(max_layer, t))) for t in qw_ths]))
    
    # 2. Random starts
    for _ in range(10):
        x0 = np.random.uniform(lo, hi)
        starts.append(x0)
        
    # 3. Graded thickness start (thicker in the middle)
    x_grad = []
    for i in range(n_layers):
        # Bell curve shape for thicknesses
        center = n_layers / 2.0
        dist = abs(i - center) / center
        t = max_layer * (1.0 - dist) * 0.5 + 1.0
        x_grad.append(math.log(max(1.0, min(max_layer, t))))
    starts.append(np.array(x_grad))

    # Run Nelder-Mead on each start
    for x0 in starts:
        if time.time() >= deadline - 10:
            break
        xc, val = _nelder_mead(eval_log, x0, lo, hi, deadline - 10, max_iters=300)
        t = np.exp(np.clip(xc, lo, hi))
        update_best(t)

    # Refinement with L-BFGS-B if available and time permits
    if _HAVE_SCIPY and best_ths is not None and time.time() < deadline - 5:
        x0 = [math.log(max(1.0, t)) for t in best_ths]
        try:
            res = _scipy_minimize(
                eval_log, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                options={"maxiter": 200, "maxfun": 2000}
            )
            xc = np.clip(res.x, lo, hi)
            t = np.exp(xc)
            update_best(t)
        except Exception:
            pass

    if best_ths is None:
        # Fallback
        return ['L'], [max(0.0, min(max_layer, wl_center / (4.0 * nL)))]

    return list(materials), [float(t) for t in best_ths]
