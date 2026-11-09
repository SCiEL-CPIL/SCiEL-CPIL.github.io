"""
Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

Strategy:
  1. Fast exact characteristic-matrix reflectance model (vectorized over 401 wavelengths).
  2. Structural representation: The stack is constrained to a "graded" structure.
     We fix the sequence of materials to alternate L/H (or H/L) to maximize index
     contrast per layer, which is physically optimal for broadband performance.
     The number of layers is a variable parameter (3 to 20).
  3. Optimization: Differential Evolution (DE) is used to optimize the thicknesses
     of these layers. DE is robust for global optimization in noisy/complex landscapes
     and handles the box constraints naturally.
  4. We run DE for multiple different layer counts (N) and material starting indices
     (L or H), keeping the best result found within the time budget.
"""

import time
import math
import random
import numpy as np

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = [n_mat[m] for m in materials]
    ds = np.asarray(thicknesses, dtype=float)

    # Build per-layer optical path lengths
    # delta_j = 2*pi*n_j*d_j/lambda
    n_ds = np.array([n * d for n, d in zip(ns, ds)])  # (N,)
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]  # (N, K)
    c = np.cos(delta)
    s = np.sin(delta)

    # Initialize characteristic matrix product as identity for all wavelengths
    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)

    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        # Update A,B,C,D
        A_new = A * cj + 1j * B * n_j * sj
        B_new = 1j * A * sj / n_j + B * cj
        C_new = C * cj + 1j * D * n_j * sj
        D_new = 1j * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    # Reflectance: r = (n_0 * (A + B * n_s) - (C + D * n_s)) / (n_0 * (A + B * n_s) + (C + D * n_s))
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)

def _differential_evolution(objective, bounds, pop_size=20, max_iter=50, F=0.8, CR=0.9, deadline=None):
    """
    Simple Differential Evolution implementation.
    objective: function taking a list of floats, returning a float (lower is better)
    bounds: list of (min, max) tuples
    """
    n_params = len(bounds)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # Initialize population
    pop = np.random.uniform(lower, upper, size=(pop_size, n_params))
    scores = np.array([objective(ind) for ind in pop])
    
    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_individual = pop[best_idx].copy()
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        for i in range(pop_size):
            if deadline and time.time() > deadline:
                break
                
            # Select three distinct random individuals
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Mutation
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Crossover
            trial = pop[i].copy()
            j_rand = np.random.randint(0, n_params)
            for j in range(n_params):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            # Ensure bounds
            trial = np.clip(trial, lower, upper)
            
            # Evaluate
            trial_score = objective(trial)
            
            # Selection
            if trial_score <= scores[i]:
                pop[i] = trial
                scores[i] = trial_score
                if trial_score < best_score:
                    best_score = trial_score
                    best_individual = trial.copy()
                    
    return best_individual, best_score

def design(problem):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    # Precompute quarter-wave thicknesses at center wavelength for initialization
    qw = {}
    for m, n in materials_dict.items():
        qw[m] = wl_center / (4.0 * n)

    best_result = None
    best_score = float('inf')
    
    # We will test alternating sequences starting with L or H
    # Lengths: 3 to 20 (odd lengths often preferred for symmetric AR, but DE can find any)
    # We prioritize longer stacks as they offer more degrees of freedom for broadband performance.
    # However, too many layers might be hard to optimize quickly.
    # Let's try lengths: 3, 5, 7, 9, 11, 13, 15, 17, 19
    
    lengths_to_try = [19, 17, 15, 13, 11, 9, 7, 5, 3]
    
    for start_mat in ['L', 'H']:
        for N in lengths_to_try:
            if time.time() > deadline - 10:
                break
                
            # Construct the material sequence
            materials = []
            for i in range(N):
                if i % 2 == 0:
                    materials.append(start_mat)
                else:
                    materials.append('H' if start_mat == 'L' else 'L')
            
            # Bounds for DE
            # Total thickness constraint: sum(d) <= max_total
            # Individual constraint: 0 <= d <= max_layer
            # To handle sum constraint in DE, we can scale the output or use a penalty.
            # Simple approach: DE optimizes in [0, max_layer], then we check/clip sum.
            # However, clipping sum might move us out of the basin.
            # Better: Define bounds such that the average thickness is reasonable.
            # max_total / N is the average. Let's allow up to max_total/N * 1.5 per layer?
            # No, max_layer is 400. max_total is 1000.
            # If N=19, avg is ~52nm.
            # Let's just use [0, max_layer] and rely on the objective function to penalize or
            # we can normalize.
            # Actually, the problem states: "sum must not exceed max_total_nm".
            # If DE produces a sum > max_total, the result is invalid (or we must clip).
            # The evaluator checks constraints. If invalid, score 0.
            # So we must ensure validity.
            # Strategy: Run DE on [0, max_layer]. If best individual violates sum, scale it down.
            # Scaling down might not be optimal, but it's valid.
            # To improve, we can add a soft penalty in the objective function for exceeding sum.
            
            def objective(ind):
                ths = np.clip(ind, 0, max_layer)
                total = np.sum(ths)
                if total > max_total:
                    # Penalty to guide DE towards valid region
                    # Scale factor
                    scale = max_total / total
                    ths_valid = ths * scale
                    # Return a high value to discourage this, but also evaluate the valid version
                    # to keep the gradient direction useful?
                    # Or just return the score of the scaled version + penalty?
                    # Let's return the score of the scaled version. This makes the landscape
                    # continuous with the valid region.
                    score = _mean_reflectance(materials, ths_valid, problem)
                    # Add a small penalty for being out of bounds to encourage staying in
                    penalty = 0.1 * (total - max_total) / max_total
                    return score + penalty
                else:
                    return _mean_reflectance(materials, ths, problem)
            
            bounds = [(0.0, max_layer)] * N
            
            # Initialize with quarter-wave values scaled to fit
            # For alternating L/H, a good start is quarter wave at center
            init = np.array([qw[m] for m in materials])
            total_init = np.sum(init)
            if total_init > max_total:
                init = init * (max_total / total_init)
            
            # DE parameters
            # Smaller population for speed if N is large?
            pop_size = 20
            max_iter = 30
            
            try:
                best_ind, best_sc = _differential_evolution(
                    objective, 
                    bounds, 
                    pop_size=pop_size, 
                    max_iter=max_iter, 
                    deadline=deadline - 5
                )
                
                # Final validation and clipping
                best_ths = np.clip(best_ind, 0, max_layer)
                total_ths = np.sum(best_ths)
                if total_ths > max_total:
                    best_ths = best_ths * (max_total / total_ths)
                
                # Re-evaluate exact score for the valid, clipped thickness
                final_score = _mean_reflectance(materials, list(best_ths), problem)
                
                if final_score < best_score:
                    best_score = final_score
                    best_result = (materials, list(best_ths))
                    
            except Exception as e:
                pass

    # If no valid result found, return a simple 1-layer AR coating
    if best_result is None:
        # 1 layer L
        return ["L"], [qw["L"]]
        
    mats, ths = best_result
    return list(mats), list(ths)
