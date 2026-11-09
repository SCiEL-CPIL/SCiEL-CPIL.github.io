"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

STRUCTURAL CHANGE:
  1. Represent the coating as a continuous refractive index profile defined by
     20 piecewise-linear nodes (effective indices between 1.38 and 2.35).
  2. Enforce constant optical path length (OD = n_eff * d) across the entire
     stack to minimize dispersion-induced phase errors.
  3. Optimize the 20 continuous node values (effective indices) using
     multi-start L-BFGS-B in log-index space, with a final polish pass.
  4. Fallback to alternating L/H optimization if the continuous profile
     yields poor results.
  5. Explicit deadline guard with safety margin.
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


def _continuous_profile_to_layers(node_indices, problem, max_total, max_layer):
    """
    Convert a continuous piecewise-linear index profile (20 nodes) into
    20 layers with constant OD (optical path length) constraint.
    
    The profile has 20 nodes -> 19 segments. We create 20 layers where:
    - Layers 1-19 correspond to the 19 segments, with n_eff = average of adjacent nodes
    - Layer 20 is a final layer with n_eff = last node index
    - OD is constant: n_eff_j * d_j = OD for all j
    - Total thickness = sum(d_j) <= max_total
    - Each d_j <= max_layer
    """
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    n_nodes = len(node_indices)
    
    # Clip node indices to [nL, nH]
    node_indices = np.clip(node_indices, nL, nH)
    
    # Compute effective indices for each layer
    # 19 segments -> 19 layers with n_eff = (node[i] + node[i+1])/2
    # Plus 1 final layer with n_eff = node[-1]
    n_layers = n_nodes  # 20 layers
    n_eff = np.zeros(n_layers)
    
    for j in range(n_nodes - 1):
        n_eff[j] = 0.5 * (node_indices[j] + node_indices[j + 1])
    n_eff[n_nodes - 1] = node_indices[-1]
    
    # Determine OD such that total thickness = max_total
    # sum(n_eff_j * d_j) = sum(OD) = n_layers * OD = max_total
    # So OD = max_total / n_layers
    OD = max_total / n_layers
    
    # Compute thicknesses
    ds = OD / n_eff
    
    # Check max_layer constraint
    if np.any(ds > max_layer):
        # Scale OD down so that max(ds) = max_layer
        max_ds = np.max(ds)
        OD = max_layer * np.min(n_eff)
        ds = OD / n_eff
    
    # Check total thickness
    total = np.sum(ds)
    if total > max_total:
        ds = ds * (max_total / total)
    
    # Ensure non-negative and within bounds
    ds = np.clip(ds, 0.0, max_layer)
    
    # Now we need to map each layer's n_eff to either 'L' or 'H'
    # Since we only have two materials, we need to decide which material
    # to use for each layer based on n_eff
    # If n_eff < (nL + nH)/2, use 'L', else use 'H'
    mid_index = 0.5 * (nL + nH)
    materials = []
    for n in n_eff:
        if n < mid_index:
            materials.append('L')
        else:
            materials.append('H')
    
    return materials, list(ds)


def _make_evaluator_continuous(problem, max_total, max_layer):
    """
    Evaluator for continuous index profile optimization.
    x = log(node_indices), length 20
    Returns mean reflectance + penalty for constraint violations.
    """
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    lo = np.full(20, math.log(nL))
    hi = np.full(20, math.log(nH))
    
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]
    
    def eval_x(x):
        node_indices = np.exp(np.clip(x, lo, hi))
        materials, thicknesses = _continuous_profile_to_layers(node_indices, problem, max_total, max_layer)
        
        # Check constraints
        total = sum(thicknesses)
        penalty = 0.0
        if total > max_total:
            penalty = 100.0 * (total - max_total) / max_total
        for t in thicknesses:
            if t > max_layer:
                penalty += 100.0 * (t - max_layer) / max_layer
        
        # Compute reflectance
        n_mat = {k: v for k, v in problem["materials"].items()}
        ns = np.array([n_mat[m] for m in materials], dtype=float)
        ds = np.asarray(thicknesses, dtype=float)
        
        val = _mean_reflectance_fast(ns, ds, wavs, n_0, n_s)
        return val + penalty
    
    return eval_x, lo, hi


def _make_evaluator_alt(materials, problem, max_total, max_layer, penalty_weight=100.0):
    """Return (eval, lo, hi) for alternating L/H optimization."""
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    n = len(materials)
    lo = np.full(n, math.log(0.5))
    hi = np.full(n, math.log(max_layer))
    
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]

    def eval_x(x):
        t = np.exp(np.clip(x, lo, hi))
        total = t.sum()
        penalty = 0.0
        if total > max_total:
            penalty = penalty_weight * (total - max_total) / max_total
        val = _mean_reflectance_fast(ns, t, wavs, n_0, n_s)
        return val + penalty

    return eval_x, lo, hi


def design(problem):
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    random.seed(42)
    np.random.seed(42)

    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]

    best_ths = None
    best_mats = None
    best_score = float("inf")

    def maybe_record(mats, ths):
        nonlocal best_ths, best_mats, best_score
        if len(ths) != len(mats):
            return
        total = sum(ths)
        if total > max_total + 1e-6:
            ths = [t * (max_total / total) for t in ths]
        ths = [max(0.0, min(max_layer, t)) for t in ths]
        if any(t < 0 or t > max_layer for t in ths):
            return
        s = _mean_reflectance(mats, ths, problem)
        if s < best_score:
            best_score = s
            best_ths = list(ths)
            best_mats = list(mats)

    # ============================================================
    # PHASE 1: Continuous index profile optimization (new approach)
    # ============================================================
    
    # Generate diverse initial profiles for the 20-node continuous profile
    profile_inits = []
    
    # 1. Linear gradient from L to H (air-side to substrate-side)
    lin = np.linspace(nL, nH, 20)
    profile_inits.append(lin)
    
    # 2. Cosine taper
    frac = np.linspace(0, 1, 20)
    cos_prof = 0.5 * (nL + nH) + 0.5 * (nH - nL) * np.cos(np.pi * frac)
    profile_inits.append(cos_prof)
    
    # 3. Smoothstep
    s = frac * frac * (3 - 2 * frac)
    smooth_prof = nL * (1 - s) + nH * s
    profile_inits.append(smooth_prof)
    
    # 4-9. Random perturbations of the linear profile
    for _ in range(6):
        pert = lin + np.random.normal(0, 0.15, size=20)
        pert = np.clip(pert, nL, nH)
        profile_inits.append(pert)
    
    # 10-12. Geometric mean profiles
    geo_lin = np.geomspace(nL, nH, 20)
    profile_inits.append(geo_lin)
    geo_pert = geo_lin * np.exp(np.random.normal(0, 0.1, size=20))
    geo_pert = np.clip(geo_pert, nL, nH)
    profile_inits.append(geo_pert)
    geo_pert2 = geo_lin * np.exp(np.random.normal(0, 0.2, size=20))
    geo_pert2 = np.clip(geo_pert2, nL, nH)
    profile_inits.append(geo_pert2)
    
    # Optimize each profile initialization
    if _HAVE_SCIPY:
        eval_x, lo, hi = _make_evaluator_continuous(problem, max_total, max_layer)
        
        for idx, prof in enumerate(profile_inits):
            if time.time() >= deadline - 20:
                break
            
            x0 = np.log(prof)
            
            try:
                res = _scipy_minimize(
                    eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                    options={"maxiter": 200, "maxfun": 3000, "maxls": 20},
                )
                xc = np.clip(res.x, lo, hi)
                node_indices = np.exp(xc)
                materials, thicknesses = _continuous_profile_to_layers(node_indices, problem, max_total, max_layer)
                maybe_record(materials, thicknesses)
            except Exception:
                pass
    
    # ============================================================
    # PHASE 2: Alternating L/H optimization (fallback/refinement)
    # ============================================================
    
    if time.time() < deadline - 30:
        # Try multiple layer counts and initialization strategies
        layer_counts = [12, 14, 16, 18, 20]
        
        for n_layers in layer_counts:
            if time.time() >= deadline - 20:
                break
            
            materials = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
            
            # Quarter-wave initialization
            qw_L = wl_center / (4.0 * nL)
            qw_H = wl_center / (4.0 * nH)
            thicknesses = []
            for m in materials:
                qw = qw_L if m == 'L' else qw_H
                t = max(1.0, min(max_layer, qw))
                thicknesses.append(t)
            total = sum(thicknesses)
            if total > max_total:
                scale = max_total / total
                thicknesses = [t * scale for t in thicknesses]
            
            x0 = np.array([math.log(max(0.5, t)) for t in thicknesses])
            
            if _HAVE_SCIPY:
                eval_x, lo, hi = _make_evaluator_alt(materials, problem, max_total, max_layer)
                try:
                    res = _scipy_minimize(
                        eval_x, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                        options={"maxiter": 300, "maxfun": 4000, "maxls": 25},
                    )
                    xc = np.clip(res.x, lo, hi)
                    t = np.exp(xc)
                    total = t.sum()
                    if total > max_total:
                        t = t * (max_total / total)
                    maybe_record(materials, list(t))
                except Exception:
                    pass
            
            # Random perturbations
            for _ in range(3):
                if time.time() >= deadline - 15:
                    break
                x0_pert = x0 + np.random.normal(0, 0.3, size=n_layers)
                x0_pert = np.clip(x0_pert, lo, hi)
                
                if _HAVE_SCIPY:
                    eval_x, lo, hi = _make_evaluator_alt(materials, problem, max_total, max_layer)
                    try:
                        res = _scipy_minimize(
                            eval_x, x0_pert, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                            options={"maxiter": 200, "maxfun": 3000, "maxls": 20},
                        )
                        xc = np.clip(res.x, lo, hi)
                        t = np.exp(xc)
                        total = t.sum()
                        if total > max_total:
                            t = t * (max_total / total)
                        maybe_record(materials, list(t))
                    except Exception:
                        pass
    
    # ============================================================
    # PHASE 3: Final polish on the best stack found
    # ============================================================
    
    if best_ths is not None and time.time() < deadline - 10:
        if _HAVE_SCIPY:
            eval_x, lo, hi = _make_evaluator_alt(best_mats, problem, max_total, max_layer)
            x0 = np.array([math.log(max(0.5, t)) for t in best_ths])
            
            for attempt in range(3):
                if time.time() >= deadline - 5:
                    break
                if attempt == 0:
                    x_start = x0
                else:
                    x_start = x0 + np.random.normal(0, 0.1, size=len(x0))
                    x_start = np.clip(x_start, lo, hi)
                
                try:
                    res = _scipy_minimize(
                        eval_x, x_start, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                        options={"maxiter": 500, "maxfun": 8000, "maxls": 30},
                    )
                    xc = np.clip(res.x, lo, hi)
                    t = np.exp(xc)
                    total = t.sum()
                    if total > max_total:
                        t = t * (max_total / total)
                    maybe_record(best_mats, list(t))
                except Exception:
                    pass

    if best_ths is None:
        # Fallback: single quarter-wave L layer at band center
        qw_L = wl_center / (4.0 * nL)
        return ['L'], [max(0.0, min(max_layer, qw_L))]

    return list(best_mats), list(best_ths)
