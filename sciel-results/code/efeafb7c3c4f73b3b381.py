"""
Broadband antireflection coating design: Differential Evolution over joint sequence and thickness space.
"""
import time
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
    
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    ds = np.asarray(thicknesses, dtype=float)

    # delta_j = 2*pi*n_j*d_j/lambda -> (N, K)
    delta = 2.0 * np.pi * (ns[:, None] * ds[:, None]) / wavs[None, :]
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
        A_new = A * cj + 1j * B * n_j * sj
        B_new = 1j * A * sj / n_j + B * cj
        C_new = C * cj + 1j * D * n_j * sj
        D_new = 1j * C * sj / n_j + D * cj
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
    deadline = start_time + 145  # Safety margin for 150s limit
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    n_L = materials_dict['L']
    n_H = materials_dict['H']
    
    # DE Parameters
    pop_size = 40
    F = 0.8  # Scale factor
    CR = 0.9  # Crossover rate
    
    # Encoding:
    # The chromosome consists of:
    # 1. A sequence of 'L'/'H' of length max_layers (padded with 'L' if fewer layers used? 
    #    Better: encode length separately? No, DE is continuous.
    #    Let's use a fixed length of max_layers but allow "zero thickness" to effectively remove layers.
    #    However, zero thickness layers are numerically fine (identity matrix) but waste computation.
    #    Actually, the problem says "at most 20 layers". A layer with 0 thickness is technically valid 
    #    if we interpret "layer" as a physical entity, but usually "at most N layers" implies count.
    #    To be safe and efficient, we will evaluate the full 20-layer vector. 
    #    If a thickness is < 1nm, we can treat it as 0.
    #    The sequence is encoded as integers 0 (L) and 1 (H) in the first 20 positions of the chromosome.
    #    The thicknesses are in the next 20 positions.
    
    # Chromosome: [seq_0, ..., seq_19, thk_0, ..., thk_19]
    # seq_i in [0, 1] -> round to 0 or 1
    # thk_i in [0, max_layer]
    
    # Objective function wrapper for DE
    def evaluate(chromosome):
        # Extract sequence
        seq_floats = chromosome[:max_layers]
        seq_ints = np.round(seq_floats).astype(int) # 0 for L, 1 for H
        materials = ['L' if s == 0 else 'H' for s in seq_ints]
        
        # Extract thicknesses
        ths = chromosome[max_layers:]
        
        # Clamp thicknesses
        ths = np.clip(ths, 0.0, max_layer)
        
        # Check total thickness constraint
        total = np.sum(ths)
        if total > max_total:
            # Penalize heavily or scale? Scaling distorts the search.
            # Better: Return a large penalty if constraint violated.
            # But DE works better if we map back. Let's scale to fit if violated.
            scale = max_total / total
            ths = ths * scale
            # Re-clamp after scaling (just in case)
            ths = np.clip(ths, 0.0, max_layer)
            
        # Optimization: If a layer has very small thickness, it's effectively removed.
        # But we still calculate it. This is O(N) per wavelength.
        # To speed up, we could filter out layers with thk < 1e-3.
        # Let's filter for efficiency.
        mask = ths > 1e-3
        if not np.any(mask):
            return _mean_reflectance([], [], problem)
        
        mat_active = [materials[i] for i in range(max_layers) if mask[i]]
        thk_active = ths[mask]
        
        return _mean_reflectance(mat_active, thk_active, problem)

    # Initialize Population
    # Random initialization
    pop = []
    for i in range(pop_size):
        # Random sequence
        seq = np.random.rand(max_layers)
        # Random thicknesses
        # Start with reasonable values, e.g., uniform in [0, max_layer/2]
        ths = np.random.rand(max_layers) * (max_layer * 0.5)
        chrom = np.concatenate([seq, ths])
        pop.append(chrom)
        
    # Evaluate initial population
    scores = [evaluate(chrom) for chrom in pop]
    
    best_idx = np.argmin(scores)
    best_chrom = pop[best_idx].copy()
    best_score = scores[best_idx]
    
    # Differential Evolution Loop
    while time.time() < deadline:
        # Check time for a full generation
        if time.time() + 0.1 > deadline:
            break
            
        next_pop = []
        next_scores = []
        
        for i in range(pop_size):
            # Select 3 distinct individuals different from i
            j, k, l = np.random.choice([x for x in range(pop_size) if x != i], 3, replace=False)
            
            # DE/rand/1 mutation
            mutant = pop[j] + F * (pop[k] - pop[l])
            
            # Crossover
            trial = pop[i].copy()
            cross_points = np.random.rand(len(trial)) < CR
            # Ensure at least one gene is mutated
            if not np.any(cross_points):
                rand_idx = np.random.randint(len(trial))
                cross_points[rand_idx] = True
                
            trial[cross_points] = mutant[cross_points]
            
            # Bounds:
            # Sequence part [0, 1]
            trial[:max_layers] = np.clip(trial[:max_layers], 0.0, 1.0)
            # Thickness part [0, max_layer]
            trial[max_layers:] = np.clip(trial[max_layers:], 0.0, max_layer)
            
            # Evaluate
            trial_score = evaluate(trial)
            
            # Selection
            if trial_score <= scores[i]:
                next_pop.append(trial)
                next_scores.append(trial_score)
            else:
                next_pop.append(pop[i])
                next_scores.append(scores[i])
                
        pop = next_pop
        scores = next_scores
        
        # Update global best
        curr_best_idx = np.argmin(scores)
        if scores[curr_best_idx] < best_score:
            best_score = scores[curr_best_idx]
            best_chrom = pop[curr_best_idx].copy()
            
    # Final Polish: Nelder-Mead on the best found
    # We need to map the best chromosome back to a local optimization space.
    # Since the sequence is discrete, we will fix the sequence from the best DE result
    # and optimize only the thicknesses using Nelder-Mead.
    
    # Extract best sequence
    seq_floats = best_chrom[:max_layers]
    seq_ints = np.round(seq_floats).astype(int)
    best_materials = ['L' if s == 0 else 'H' for s in seq_ints]
    
    # Extract best thicknesses
    best_ths = np.clip(best_chrom[max_layers:], 0.0, max_layer)
    
    # Check if we should trim zero-thickness layers for NM
    # NM works on the full vector, but zero layers are fine.
    
    def constrain_ths(ths):
        ths = np.clip(ths, 0.0, max_layer)
        total = np.sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = ths * scale
            ths = np.clip(ths, 0.0, max_layer)
        return ths

    def nm_objective(ths):
        ths = constrain_ths(ths)
        mask = ths > 1e-3
        if not np.any(mask):
            return _mean_reflectance([], [], problem)
        mat_active = [best_materials[i] for i in range(max_layers) if mask[i]]
        thk_active = ths[mask]
        return _mean_reflectance(mat_active, thk_active, problem)

    # Simple Nelder-Mead implementation
    def nelder_mead(func, x0, max_iters=100):
        n = len(x0)
        # Initial simplex
        simplex = [x0.copy()]
        for i in range(n):
            xi = x0.copy()
            xi[i] += 5.0 if xi[i] == 0 else 0.05 * xi[i] # Adaptive step
            simplex.append(xi)
            
        scores = [func(simplex[i]) for i in range(n + 1)]
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        it = 0
        while it < max_iters and time.time() < deadline:
            it += 1
            order = np.argsort(scores)
            simplex = [simplex[i] for i in order]
            scores = [scores[i] for i in order]
            
            # Check convergence
            if abs(scores[0] - scores[-1]) < 1e-6:
                break
                
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(xr)
            
            if fr < scores[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    scores[-1] = fe
                else:
                    simplex[-1] = xr
                    scores[-1] = fr
            elif fr < scores[-2]:
                simplex[-1] = xr
                scores[-1] = fr
            else:
                # Contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = func(xc)
                if fc < scores[-1]:
                    simplex[-1] = xc
                    scores[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        scores[i] = func(simplex[i])
                        
        return simplex[0], scores[0]

    final_ths, final_score = nelder_mead(nm_objective, best_ths)
    
    if final_score < best_score:
        best_score = final_score
        best_ths = final_ths
        
    # Finalize output
    best_ths = constrain_ths(best_ths)
    
    # Filter out zero thickness layers for the final answer to be clean
    mask = best_ths > 1e-3
    if not np.any(mask):
        return [], []
        
    final_materials = [best_materials[i] for i in range(max_layers) if mask[i]]
    final_thicknesses = best_ths[mask]
    
    return final_materials, list(final_thicknesses)
