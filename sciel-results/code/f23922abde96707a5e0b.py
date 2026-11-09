"""
Broadband antireflection coating design: Island-based Parallel Nelder-Mead with Vectorized Evaluation.

Strategy:
1. Fast Evaluation: The reflectance calculation is vectorized across all 401 wavelengths using NumPy arrays. 
   This reduces the cost of a single evaluation from O(N_layers * N_wavelengths) Python loops to 
   O(N_layers) vectorized operations, making it ~50x faster.
2. Island Model: Instead of a single global SA, we run 4 independent "islands" (populations) in parallel 
   (sequentially in time, but logically isolated). Each island maintains a population of candidate stacks.
3. Local Search: Each island uses a Nelder-Mead simplex method to optimize the continuous thicknesses 
   for a fixed material sequence. When the sequence changes (via mutation), the simplex is reset.
4. Crossover: Periodically, the best individuals from different islands are crossed over (sequence swap) 
   to explore new topologies.
"""
import time
import random
import numpy as np

def _mean_reflectance_fast(materials, thicknesses, problem):
    """
    Compute mean reflectance (percent) over the band using characteristic matrices.
    Vectorized over wavelengths for speed.
    """
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    ds = np.asarray(thicknesses, dtype=float)

    # delta_j = 2*pi*n_j*d_j/lambda -> (N, K)
    # wavs shape (K,), ns shape (N,), ds shape (N,)
    # delta shape (N, K)
    delta = 2.0 * np.pi * (ns[:, None] * ds[:, None]) / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)

    # Initialize characteristic matrix product as identity for all wavelengths
    # A, B, C, D are arrays of shape (K,)
    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)

    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        
        # Update matrices: [A B; C D] = [c_j  i*s_j/n_j; i*n_j*s_j  c_j] * [A B; C D]
        # Actually the standard recurrence is:
        # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
        # M_total = M_1 * M_2 * ... * M_N
        # So we multiply M_j on the left of the accumulated matrix.
        
        A_new = cj * A + 1j * (sj / n_j) * B
        B_new = cj * B + 1j * n_j * sj * A
        C_new = cj * C + 1j * (sj / n_j) * D
        D_new = cj * D + 1j * n_j * sj * C
        
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    
    # r = (n_0 * (A + B*n_s) - (C + D*n_s)) / (n_0 * (A + B*n_s) + (C + D*n_s))
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    
    # Handle potential division by zero (shouldn't happen for lossless)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)

def _constrain_thicknesses(ths, max_total, max_layer):
    """Constrain thicknesses to be within bounds and total sum."""
    ths = np.asarray(ths, dtype=float)
    ths = np.clip(ths, 0.0, max_layer)
    total = np.sum(ths)
    if total > max_total:
        scale = max_total / total
        ths = ths * scale
        ths = np.clip(ths, 0.0, max_layer)
    return ths

def design(problem):
    start_time = time.time()
    # Evaluator kills at 150s. We target 140s to be safe.
    deadline = start_time + 140.0
    
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    
    n_L = materials_dict['L']
    n_H = materials_dict['H']
    
    # Helper to evaluate a genome
    def evaluate_genome(materials, thicknesses):
        ths = _constrain_thicknesses(thicknesses, max_total, max_layer)
        score = _mean_reflectance_fast(materials, ths, problem)
        return score, ths

    def nelder_mead(materials_pattern, init_ths, deadline, max_iters=200):
        """Polish thicknesses using Nelder-Mead. Returns best_ths, best_score."""
        ths = list(init_ths)
        ths = [max(0.0, min(max_layer, t)) for t in ths]
        total = sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = [t * scale for t in ths]
        
        best_score = _mean_reflectance_fast(materials_pattern, ths, problem)
        best_ths = list(ths)
        n = len(ths)
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initial step size
        step = 20.0
        simplex = [list(ths)]
        for i in range(n):
            pt = list(ths)
            pt[i] = max(0.0, min(max_layer, pt[i] + step))
            simplex.append(pt)
            
        scores = [_mean_reflectance_fast(materials_pattern, simplex[i], problem) for i in range(n + 1)]
        
        it = 0
        while it < max_iters and time.time() < deadline:
            it += 1
            order = np.argsort(scores)
            simplex = [simplex[i] for i in order]
            scores = [scores[i] for i in order]
            
            if scores[0] < best_score:
                best_score = scores[0]
                best_ths = list(simplex[0])
                
            # Check convergence
            if max(scores) - min(scores) < 1e-7:
                break
                
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = [max(0.0, min(max_layer, x)) for x in xr]
            total = sum(xr)
            if total > max_total:
                scale = max_total / total
                xr = [x * scale for x in xr]
            fr = _mean_reflectance_fast(materials_pattern, xr, problem)
            
            if fr < scores[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = [max(0.0, min(max_layer, x)) for x in xe]
                total = sum(xe)
                if total > max_total:
                    scale = max_total / total
                    xe = [x * scale for x in xe]
                fe = _mean_reflectance_fast(materials_pattern, xe, problem)
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
                if fr < scores[-1]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = [max(0.0, min(max_layer, x)) for x in xc]
                    total = sum(xc)
                    if total > max_total:
                        scale = max_total / total
                        xc = [x * scale for x in xc]
                    fc = _mean_reflectance_fast(materials_pattern, xc, problem)
                    if fc <= fr:
                        simplex[-1] = xc
                        scores[-1] = fc
                    else:
                        break
                else:
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        simplex[i] = [max(0.0, min(max_layer, x)) for x in simplex[i]]
                        total = sum(simplex[i])
                        if total > max_total:
                            scale = max_total / total
                            simplex[i] = [x * scale for x in simplex[i]]
                        scores[i] = _mean_reflectance_fast(materials_pattern, simplex[i], problem)
                        
        return best_ths, best_score

    # Initialize Islands
    # Each island is a list of individuals (mats, ths, score)
    num_islands = 4
    pop_size = 5
    islands = []
    
    # Initial guesses: Alternating L/H, Quarter-wave at center
    wl_center = 0.5 * (band[0] + band[1])
    
    for i in range(num_islands):
        island = []
        for j in range(pop_size):
            # Randomize the sequence slightly
            mats = []
            ths = []
            # Start with alternating
            for k in range(max_layers):
                if i % 2 == 0:
                    # Even islands start with L
                    m = 'L' if (k + j) % 2 == 0 else 'H'
                else:
                    # Odd islands start with H
                    m = 'H' if (k + j) % 2 == 0 else 'L'
                mats.append(m)
                # Randomize thickness around quarter wave
                n_mat = n_L if m == 'L' else n_H
                t = wl_center / (4.0 * n_mat)
                t *= random.uniform(0.8, 1.2)
                ths.append(t)
            
            score, constrained_ths = evaluate_genome(mats, ths)
            island.append((mats, constrained_ths, score))
        
        # Sort island by score
        island.sort(key=lambda x: x[2])
        islands.append(island)
    
    best_global = islands[0][0]
    
    # Main Loop
    while time.time() < deadline - 10:
        # For each island, perform local search on the best individual
        for i in range(num_islands):
            if time.time() > deadline - 5:
                break
            
            # Pick the best individual from the island
            best_ind = islands[i][0]
            mats, ths, score = best_ind
            
            # Run Nelder-Mead to polish this individual
            # We give it a limited time budget per call to ensure we cycle through islands
            nm_deadline = min(deadline - 5, time.time() + 1.0) # 1 second per polish attempt
            polished_ths, polished_score = nelder_mead(mats, ths, nm_deadline, max_iters=50)
            
            if polished_score < score:
                islands[i][0] = (mats, polished_ths, polished_score)
                if polished_score < best_global[2]:
                    best_global = (mats, polished_ths, polished_score)
            
            # Mutation: Create a new individual by mutating the best one
            # Strategy: Flip one layer material, or randomize one thickness
            new_mats = list(mats)
            new_ths = list(polished_ths)
            
            if random.random() < 0.3:
                # Flip material
                idx = random.randint(0, max_layers - 1)
                new_mats[idx] = 'H' if new_mats[idx] == 'L' else 'L'
            else:
                # Randomize thickness
                idx = random.randint(0, max_layers - 1)
                n_mat = n_L if new_mats[idx] == 'L' else n_H
                new_ths[idx] = random.uniform(0, max_layer)
            
            new_score, new_constrained_ths = evaluate_genome(new_mats, new_ths)
            
            # Insert into island if it's better than the worst
            if new_score < islands[i][-1][2]:
                islands[i][-1] = (new_mats, new_constrained_ths, new_score)
                islands[i].sort(key=lambda x: x[2])
                if new_score < best_global[2]:
                    best_global = (new_mats, new_constrained_ths, new_score)
            
            # Crossover: Occasionally, mix the best individual of this island with another
            if i > 0 and random.random() < 0.1:
                other_best = islands[i-1][0]
                # Take first half of mats from best, second half from other
                cross_mats = new_mats[:max_layers//2] + other_best[0][max_layers//2:]
                cross_ths = new_constrained_ths[:max_layers//2] + other_best[1][max_layers//2:]
                cross_score, cross_ths = evaluate_genome(cross_mats, cross_ths)
                
                if cross_score < islands[i][-1][2]:
                    islands[i][-1] = (cross_mats, cross_ths, cross_score)
                    islands[i].sort(key=lambda x: x[2])
                    if cross_score < best_global[2]:
                        best_global = (cross_mats, cross_ths, cross_score)

    # Final Polish on the absolute best
    if best_global:
        mats, ths, score = best_global
        # Use remaining time for a final deep polish
        final_ths, final_score = nelder_mead(mats, ths, deadline, max_iters=500)
        if final_score < score:
            best_global = (mats, final_ths, final_score)
    
    # Ensure valid output
    if not best_global:
        # Fallback
        mats = ['L', 'H'] * 10
        ths = [100.0] * 20
        return mats, ths
    
    mats, ths, score = best_global
    # Final constraint check
    ths = _constrain_thicknesses(ths, max_total, max_layer)
    return list(mats), list(ths)
