"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

Strategy:
  1. Exact characteristic-matrix reflectance model (vectorized over 401 wavelengths).
  2. Global optimization (differential evolution) for thicknesses with multiple patterns.
  3. Local polish with L-BFGS-B.
  4. Explicit deadline guard.
"""

import time
import math
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize

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
    
    n = len(materials)
    if n == 0:
        return 100.0
    
    n_ds = np.array([n_i * d for n_i, d in zip(ns, ds)])
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)
    
    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)
    
    i = 1.0j
    for j in range(n):
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


def design(problem):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    
    best_score = float('inf')
    best_mats = None
    best_ths = None
    
    def update_best(mats, ths):
        nonlocal best_score, best_mats, best_ths
        if len(mats) > max_layers:
            return
        if any(t < 0 or t > max_layer for t in ths):
            return
        if sum(ths) > max_total + 1e-6:
            return
        score = _mean_reflectance(mats, ths, problem)
        if score < best_score:
            best_score = score
            best_mats = list(mats)
            best_ths = list(ths)
    
    def make_objective(mats):
        """Create objective function with penalty for total thickness."""
        def objective(x):
            x = np.clip(x, 0, max_layer)
            total = np.sum(x)
            if total > max_total:
                penalty = 1000.0 * (total - max_total) / max_total
                return _mean_reflectance(mats, list(x), problem) + penalty
            return _mean_reflectance(mats, list(x), problem)
        return objective
    
    def get_patterns(n_layers):
        """Generate material patterns for given layer count."""
        patterns = []
        # Alternating L/H
        patterns.append(['L' if i % 2 == 0 else 'H' for i in range(n_layers)])
        # Alternating H/L
        patterns.append(['H' if i % 2 == 0 else 'L' for i in range(n_layers)])
        # L-heavy on air side
        n_L = int(n_layers * 0.6)
        patterns.append(['L'] * n_L + ['H'] * (n_layers - n_L))
        # H-heavy on air side
        n_H = int(n_layers * 0.6)
        patterns.append(['H'] * n_H + ['L'] * (n_layers - n_H))
        return patterns
    
    # Try different layer counts
    layer_counts = [6, 8, 10, 12, 14, 16, 18, 20]
    layer_counts = [c for c in layer_counts if c <= max_layers]
    
    for n_layers in layer_counts:
        if time.time() > deadline - 60:
            break
        
        patterns = get_patterns(n_layers)
        bounds = [(0.0, max_layer)] * n_layers
        
        for mats in patterns:
            if time.time() > deadline - 35:
                break
            
            objective = make_objective(mats)
            
            # Differential evolution
            try:
                de_result = differential_evolution(
                    objective,
                    bounds,
                    seed=42 + n_layers,
                    maxiter=50,
                    tol=1e-4,
                    polish=False,
                    init='latinhypercube',
                    popsize=12,
                    mutation=(0.3, 0.9),
                    recombination=0.8
                )
                
                x_best = np.clip(de_result.x, 0, max_layer)
                total = np.sum(x_best)
                if total > max_total:
                    x_best = x_best * (max_total / total)
                    x_best = np.clip(x_best, 0, max_layer)
                
                ths = list(x_best)
                if all(0 <= t <= max_layer for t in ths) and sum(ths) <= max_total + 1e-6:
                    update_best(mats, ths)
                
                # Polish with L-BFGS-B
                if time.time() < deadline - 25:
                    res = minimize(
                        objective,
                        x_best,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 300, 'maxfun': 5000}
                    )
                    ths = list(res.x)
                    total = sum(ths)
                    if total > max_total:
                        ths = [t * (max_total / total) for t in ths]
                    ths = [max(0, min(max_layer, t)) for t in ths]
                    if all(0 <= t <= max_layer for t in ths) and sum(ths) <= max_total + 1e-6:
                        update_best(mats, ths)
                        
            except Exception:
                pass
    
    # Final polish on best solution
    if best_mats is not None and time.time() < deadline - 15:
        mats = best_mats
        ths = best_ths
        n = len(mats)
        bounds = [(0.0, max_layer)] * n
        objective = make_objective(mats)
        
        try:
            res = minimize(
                objective,
                np.array(ths),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 10000}
            )
            new_ths = list(res.x)
            total = sum(new_ths)
            if total > max_total:
                new_ths = [t * (max_total / total) for t in new_ths]
            new_ths = [max(0, min(max_layer, t)) for t in new_ths]
            if all(0 <= t <= max_layer for t in new_ths) and sum(new_ths) <= max_total + 1e-6:
                update_best(mats, new_ths)
        except Exception:
            pass
    
    # Fallback
    if best_mats is None:
        qw_L = wl_center / (4.0 * nL)
        best_mats = ['L']
        best_ths = [min(qw_L, max_layer)]
    
    return list(best_mats), list(best_ths)
