<<<<<<< SEARCH
def construct():
    global _T0
    _T0 = time.time()
    
    M = 512
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    z = np.exp(1j * theta)
    Z = np.power(z[:, None], np.arange(_N)[None, :])
    
    # Phase 1: Partial brute force to get a good starting point
    B = 10000
    total = 1 << _N
    bit_mask = 1 << np.arange(_N, dtype=np.uint32)
    
    best_peak_sq = float('inf')
    best_coeff_idx = 0
    best_peak_phase = 0.0
    
    # Search a subset of candidates (first ~500k to get a good start)
    max_candidates = 500000
    for start in range(0, min(total, max_candidates), B):
        if _time_left() < 10.0:
            break
        
        end = min(start + B, total)
        idxs = np.arange(start, end, dtype=np.uint32)
        bits = (idxs[:, None] & bit_mask[None, :]) > 0
        coeffs = bits.astype(np.float64) * 2.0 - 1.0
        
        vals = coeffs @ Z.T
        mag2 = np.abs(vals) ** 2
        peaks = np.max(mag2, axis=1)
        
        min_idx = np.argmin(peaks)
        min_peak_sq = peaks[min_idx]
        
        if min_peak_sq < best_peak_sq:
            best_peak_sq = min_peak_sq
            best_coeff_idx = start + min_idx
            best_peak_phase = theta[np.argmax(mag2[min_idx])]
    
    # Retrieve best coefficients from brute force
    bits = (best_coeff_idx & (1 << np.arange(_N, dtype=np.uint32))) > 0
    best_coeffs = bits.astype(np.float64) * 2.0 - 1.0
    
    # Refine the peak
    true_peak_sq = _refine_peak(best_coeffs, best_peak_sq, best_peak_phase, M_grid=M)
    best_peak_sq = true_peak_sq
    
    # Phase 2: Local search from the best candidate
    best_coeffs, best_peak_sq = _local_search(best_coeffs, Z, theta, M, max_flips=3, time_budget=30.0)
    
    # Phase 3: Genetic algorithm for broader exploration
    if _time_left() > 20.0:
        ga_coeffs, ga_peak_sq = _genetic_algorithm(best_coeffs, Z, theta, M, pop_size=150, generations=50, time_budget=40.0)
        if ga_peak_sq < best_peak_sq:
            best_coeffs = ga_coeffs
            best_peak_sq = ga_peak_sq
            # Local search from GA result
            best_coeffs, best_peak_sq = _local_search(best_coeffs, Z, theta, M, max_flips=3, time_budget=20.0)
    
    # Final refinement
    _, peak_idx = _eval_peak_sq(best_coeffs, Z, theta, M)
    final_peak_sq = _refine_peak(best_coeffs, best_peak_sq, theta[peak_idx], M_grid=M, iters=40)
    best_peak_sq = final_peak_sq
    
    # Convert back to list of ints
    result = [int(1 if c > 0 else -1) for c in best_coeffs]
    
    return result
=======
def construct():
    global _T0
    _T0 = time.time()
    
    M = 512
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    z = np.exp(1j * theta)
    Z = np.power(z[:, None], np.arange(_N)[None, :])
    
    # Phase 1: Partial brute force to get a good starting point
    B = 10000
    total = 1 << _N
    bit_mask = 1 << np.arange(_N, dtype=np.uint32)
    
    best_peak_sq = float('inf')
    best_coeff_idx = 0
    best_peak_phase = 0.0
    
    # Search a subset of candidates (first ~500k to get a good start)
    max_candidates = 500000
    for start in range(0, min(total, max_candidates), B):
        if _time_left() < 10.0:
            break
        
        end = min(start + B, total)
        idxs = np.arange(start, end, dtype=np.uint32)
        bits = (idxs[:, None] & bit_mask[None, :]) > 0
        coeffs = bits.astype(np.float64) * 2.0 - 1.0
        
        vals = coeffs @ Z.T
        mag2 = np.abs(vals) ** 2
        peaks = np.max(mag2, axis=1)
        
        min_idx = np.argmin(peaks)
        min_peak_sq = peaks[min_idx]
        
        if min_peak_sq < best_peak_sq:
            best_peak_sq = min_peak_sq
            best_coeff_idx = start + min_idx
            best_peak_phase = theta[np.argmax(mag2[min_idx])]
    
    # Retrieve best coefficients from brute force
    bits = (best_coeff_idx & (1 << np.arange(_N, dtype=np.uint32))) > 0
    best_coeffs = bits.astype(np.float64) * 2.0 - 1.0
    
    # Refine the peak
    true_peak_sq = _refine_peak(best_coeffs, best_peak_sq, best_peak_phase, M_grid=M)
    best_peak_sq = true_peak_sq
    
    # Phase 2: Projected gradient descent in continuous space
    if _time_left() > 30.0:
        pgd_coeffs, pgd_peak_sq = _projected_gradient_descent(best_coeffs, theta, M, steps=3000, lr=0.05, beta=10.0, time_budget=25.0)
        if pgd_peak_sq < best_peak_sq:
            best_coeffs = pgd_coeffs
            best_peak_sq = pgd_peak_sq
    
    # Phase 3: Local search from the best candidate
    best_coeffs, best_peak_sq = _local_search(best_coeffs, Z, theta, M, max_flips=3, time_budget=20.0)
    
    # Phase 4: Additional projected gradient descent rounds from multiple restarts
    if _time_left() > 25.0:
        rng = np.random.RandomState(123)
        for restart in range(3):
            if _time_left() < 10.0:
                break
            # Random perturbation of best solution
            perturbed = best_coeffs.copy()
            for _ in range(3):
                idx = rng.randint(0, _N)
                perturbed[idx] = -perturbed[idx]
            pgd_coeffs, pgd_peak_sq = _projected_gradient_descent(perturbed, theta, M, steps=1500, lr=0.03, beta=8.0, time_budget=8.0)
            if pgd_peak_sq < best_peak_sq:
                best_coeffs = pgd_coeffs
                best_peak_sq = pgd_peak_sq
                # Quick local search
                best_coeffs, best_peak_sq = _local_search(best_coeffs, Z, theta, M, max_flips=2, time_budget=5.0)
    
    # Phase 5: Genetic algorithm for broader exploration
    if _time_left() > 15.0:
        ga_coeffs, ga_peak_sq = _genetic_algorithm(best_coeffs, Z, theta, M, pop_size=100, generations=30, time_budget=12.0)
        if ga_peak_sq < best_peak_sq:
            best_coeffs = ga_coeffs
            best_peak_sq = ga_peak_sq
            # Local search from GA result
            best_coeffs, best_peak_sq = _local_search(best_coeffs, Z, theta, M, max_flips=3, time_budget=8.0)
    
    # Final refinement
    _, peak_idx = _eval_peak_sq(best_coeffs, Z, theta, M)
    final_peak_sq = _refine_peak(best_coeffs, best_peak_sq, theta[peak_idx], M_grid=M, iters=40)
    best_peak_sq = final_peak_sq
    
    # Convert back to list of ints
    result = [int(1 if c > 0 else -1) for c in best_coeffs]
    
    return result
>>>>>>> REPLACE
