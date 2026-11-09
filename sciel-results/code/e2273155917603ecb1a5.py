"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

Strategy:
1. Fixed 20-layer alternating L/H structure.
2. Differential Evolution (DE) for global optimization of layer thicknesses.
   - DE explores the 20-dimensional space of thicknesses.
   - The fitness function handles the total thickness constraint by scaling down
     if the sum exceeds 1000 nm. This allows DE to explore configurations that
     might naturally want to be thicker but are constrained, effectively finding
     the optimal distribution of the 1000 nm budget.
   - Initialization includes sparse solutions (few non-zero layers) and dense
     quarter-wave stacks to guide the search towards known good regions.
3. L-BFGS-B local refinement from the best DE solution.
"""

import time
import math
import random
import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
    from scipy.optimize import differential_evolution as _scipy_de
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _mean_reflectance(materials, thicknesses, problem):
    """Mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = [n_mat[m] for m in materials]
    ds = np.asarray(thicknesses, dtype=float)

    # Filter out zero thickness layers to avoid numerical issues with division by zero
    # in the characteristic matrix calculation if n_j is used in denominators?
    # The formula uses 1/n_j. If d=0, delta=0, sin=0, cos=1.
    # B_new = i*A*0/n_j + B*1 = B.
    # A_new = A*1 + i*B*n_j*0 = A.
    # So zero thickness layers are identity matrices.
    # We can just keep them in the loop, it's fast enough for 20 layers.
    
    n_ds = np.array([n * d for n, d in zip(ns, ds)])
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)

    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)

    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + i * B * n_j * sj
        B_new = i * A * sj / n_j + B * cj
        C_new = C * cj + i * D * n_j * sj
        D_new = i * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)


def _make_evaluator(materials, problem, max_total, max_layer):
    """
    Returns an evaluator function for DE.
    Input: x (array of thicknesses)
    Output: Mean reflectance.
    Handles scaling if sum(x) > max_total.
    Clips x to [0, max_layer].
    """
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    n = len(materials)
    
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]

    def eval_x(x):
        # Clip to valid range
        t = np.clip(x, 0.0, max_layer)
        total = t.sum()
        if total > max_total:
            t = t * (max_total / total)
        
        # Fast reflectance calculation
        n_ds = ns * t
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

    return eval_x


def design(problem):
    start_time = time.time()
    deadline = start_time + 120  # Safety margin

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    random.seed(42)
    np.random.seed(42)

    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    qw_L = wl_center / (4.0 * nL)
    qw_H = wl_center / (4.0 * nH)

    # Fixed 20-layer alternating L/H structure
    n_layers = min(20, max_layers)
    materials = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
    
    eval_x = _make_evaluator(materials, problem, max_total, max_layer)
    
    best_ths = None
    best_score = float("inf")

    def maybe_record(ths):
        nonlocal best_ths, best_score
        # Ensure constraints are met for the record
        ths = np.clip(ths, 0.0, max_layer)
        total = ths.sum()
        if total > max_total:
            ths = ths * (max_total / total)
        
        # Check if all layers are effectively zero?
        if np.all(ths < 1e-6):
            return
            
        s = eval_x(ths)
        if s < best_score:
            best_score = s
            best_ths = list(ths)

    # 1. Generate initial diverse candidates
    candidates = []
    
    # Candidate 1: Single quarter-wave L layer (Layer 0)
    t1 = np.zeros(n_layers)
    t1[0] = qw_L
    candidates.append(t1)
    
    # Candidate 2: 3-layer quarter-wave stack (L-H-L)
    t2 = np.zeros(n_layers)
    t2[0] = qw_L
    t2[1] = qw_H
    t2[2] = qw_L
    candidates.append(t2)
    
    # Candidate 3: 5-layer quarter-wave stack (L-H-L-H-L)
    t3 = np.zeros(n_layers)
    for i in range(5):
        t3[i] = qw_L if i % 2 == 0 else qw_H
    candidates.append(t3)
    
    # Candidate 4: 7-layer quarter-wave stack
    t4 = np.zeros(n_layers)
    for i in range(7):
        t4[i] = qw_L if i % 2 == 0 else qw_H
    candidates.append(t4)
    
    # Candidate 5: Random dense stack
    t5 = np.random.uniform(qw_L*0.5, qw_H*1.5, n_layers)
    candidates.append(t5)
    
    # Candidate 6: Random sparse stack (few non-zero)
    t6 = np.zeros(n_layers)
    non_zero_count = random.randint(3, 10)
    indices = random.sample(range(n_layers), non_zero_count)
    for idx in indices:
        t6[idx] = random.uniform(qw_L*0.5, qw_H*1.5)
    candidates.append(t6)

    # Evaluate candidates
    for c in candidates:
        maybe_record(c)

    # 2. Differential Evolution
    if _HAVE_SCIPY and time.time() < deadline - 30:
        try:
            bounds = [(0.0, max_layer)] * n_layers
            # DE parameters
            # popsize=20 is standard, but we might want more for 20D
            # maxiter=100 is a good balance
            de_res = _scipy_de(
                eval_x, 
                bounds, 
                maxiter=100, 
                popsize=20, 
                seed=42, 
                tol=1e-8,
                # init='differential' is default, creates random population
                # We can try to seed it with our candidates? 
                # DE doesn't directly accept a population seed in the same way, 
                # but we can run a custom loop or just rely on DE's exploration.
                # Given the deadline, standard DE is safer.
            )
            if de_res.success or de_res.fun < best_score:
                maybe_record(de_res.x)
                # Polish the DE result
                if time.time() < deadline - 10:
                    res = _scipy_minimize(
                        eval_x, de_res.x, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 200, "maxfun": 2000, "ftol": 1e-12, "gtol": 1e-10}
                    )
                    maybe_record(res.x)
        except Exception:
            pass

    # 3. Local refinement of the best candidate found so far
    if best_ths is not None and time.time() < deadline - 5:
        x0 = np.array(best_ths)
        try:
            bounds = [(0.0, max_layer)] * n_layers
            res = _scipy_minimize(
                eval_x, x0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 500, "maxfun": 5000, "ftol": 1e-14, "gtol": 1e-12}
            )
            maybe_record(res.x)
        except Exception:
            pass

    # 4. Final fallback if nothing was recorded (should not happen)
    if best_ths is None:
        best_ths = [qw_L] + [0.0] * (n_layers - 1)
        best_mats = materials
    else:
        best_mats = materials

    return best_mats, best_ths
