import math
import time
import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 145.0  # Safety margin from 150s limit

def _time_left():
    return _LIMIT - (time.time() - _T0)

def _rs_poly(n):
    """Generate Rudin-Shapiro coefficients of length n."""
    if n <= 0:
        return []
    if n == 1:
        return [1]
    if n == 2:
        return [1, 1]
    
    R = [1]
    S = [1]
    while len(R) < n:
        new_R = R + S
        new_S = R + [-s for s in S]
        R = new_R
        S = new_S
    return R[:n]

def _calc_peak(coeffs, M=2048, refine_top=10, refine_iters=40):
    """
    Calculate the peak magnitude of P(z) on the unit circle.
    Uses dense sampling and golden-section refinement for top peaks.
    """
    c = np.array(coeffs, dtype=complex)
    
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    
    powers = np.arange(_N)
    Z = np.exp(1j * np.outer(theta, powers))
    vals = Z @ c
    
    mag2 = np.abs(vals) ** 2
    peak_val = np.max(mag2)
    
    # Find indices of top peaks and refine
    if refine_top > 0 and M > 10:
        top_indices = np.argsort(mag2)[-refine_top:][::-1]
        best_peak = peak_val
        powers_arr = powers
        
        for idx in top_indices:
            delta = 2.0 * math.pi / M
            lo = theta[idx] - delta
            hi = theta[idx] + delta
            
            def f(a):
                z_a = np.exp(1j * a)
                z_powers = np.power(z_a, powers_arr)
                s = np.dot(c, z_powers)
                return s.real * s.real + s.imag * s.imag

            gr = (math.sqrt(5) - 1) / 2
            x1 = hi - gr * (hi - lo)
            x2 = lo + gr * (hi - lo)
            f1 = f(x1)
            f2 = f(x2)
            
            for _ in range(refine_iters):
                if f1 < f2:
                    hi = x2
                    x2 = x1
                    f2 = f1
                    x1 = hi - gr * (hi - lo)
                    f1 = f(x1)
                else:
                    lo = x1
                    x1 = x2
                    f1 = f2
                    x2 = lo + gr * (hi - lo)
                    f2 = f(x2)
            
            local_peak = max(f1, f2)
            if local_peak > best_peak:
                best_peak = local_peak
        
        peak_val = best_peak
    
    return math.sqrt(peak_val)

def _calc_peak_fast(coeffs, M=1024, refine_top=0, refine_iters=0):
    """
    Very fast peak calculation for local search iterations.
    Uses only dense sampling without refinement for speed.
    """
    c = np.array(coeffs, dtype=complex)
    
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    
    powers = np.arange(_N)
    Z = np.exp(1j * np.outer(theta, powers))
    vals = Z @ c
    
    mag2 = np.abs(vals) ** 2
    peak_val = np.max(mag2)
    
    return math.sqrt(peak_val)

def _local_search(seed, M=1024, max_time=140.0):
    """
    Perform local search to minimize the peak.
    Uses very fast peak calculation for speed, with periodic verification.
    """
    cur = list(seed)
    cur_peak = _calc_peak_fast(cur, M)
    best = list(cur)
    best_peak = _calc_peak(cur, M=4096, refine_top=10, refine_iters=40)
    
    start_time = time.time()
    last_verify = start_time
    
    while time.time() - start_time < max_time:
        improved = False
        verify_needed = (time.time() - last_verify) > 2.0
        
        # Single flip pass
        for j in range(_N):
            if time.time() - start_time > max_time:
                break
            cur[j] = -cur[j]
            p = _calc_peak_fast(cur, M)
            if p < cur_peak - 1e-10:
                cur_peak = p
                improved = True
                if verify_needed:
                    hp = _calc_peak(cur, M=4096, refine_top=10, refine_iters=40)
                    if hp < best_peak - 1e-10:
                        best = list(cur)
                        best_peak = hp
                    verify_needed = False
                    last_verify = time.time()
            else:
                cur[j] = -cur[j]
        
        if not improved:
            # Try double flips
            double_improved = False
            for j in range(_N):
                if time.time() - start_time > max_time:
                    break
                for k in range(j + 1, _N):
                    if time.time() - start_time > max_time:
                        break
                    cur[j] = -cur[j]
                    cur[k] = -cur[k]
                    p = _calc_peak_fast(cur, M)
                    if p < cur_peak - 1e-10:
                        cur_peak = p
                        double_improved = True
                        if verify_needed:
                            hp = _calc_peak(cur, M=4096, refine_top=10, refine_iters=40)
                            if hp < best_peak - 1e-10:
                                best = list(cur)
                                best_peak = hp
                            verify_needed = False
                            last_verify = time.time()
                    else:
                        cur[j] = -cur[j]
                        cur[k] = -cur[k]
            
            if not double_improved:
                # Try triple flips if stuck
                triple_improved = False
                for j in range(_N):
                    if time.time() - start_time > max_time:
                        break
                    for k in range(j + 1, _N):
                        if time.time() - start_time > max_time:
                            break
                        for l in range(k + 1, _N):
                            if time.time() - start_time > max_time:
                                break
                            cur[j] = -cur[j]
                            cur[k] = -cur[k]
                            cur[l] = -cur[l]
                            p = _calc_peak_fast(cur, M)
                            if p < cur_peak - 1e-10:
                                cur_peak = p
                                triple_improved = True
                                if verify_needed:
                                    hp = _calc_peak(cur, M=4096, refine_top=10, refine_iters=40)
                                    if hp < best_peak - 1e-10:
                                        best = list(cur)
                                        best_peak = hp
                                    verify_needed = False
                                    last_verify = time.time()
                            else:
                                cur[j] = -cur[j]
                                cur[k] = -cur[k]
                                cur[l] = -cur[l]
                
                if not triple_improved:
                    break
                else:
                    improved = True
            else:
                improved = True
        
        # Periodic high-accuracy check
        if time.time() - start_time < max_time - 5.0 and (time.time() - last_verify) > 3.0:
            high_acc_peak = _calc_peak(best, M=4096, refine_top=10, refine_iters=40)
            if high_acc_peak < best_peak:
                best_peak = high_acc_peak
            last_verify = time.time()
    
    # Final high-accuracy verification
    final_peak = _calc_peak(best, M=8192, refine_top=10, refine_iters=50)
    return best, final_peak

def _genetic_search(pop_size=50, generations=100, max_time=120.0, M=1024):
    """
    Genetic Algorithm for exploring the solution space.
    Fitness is the negative peak magnitude.
    """
    start_time = time.time()
    
    # Initialize population with RS seed and its variations
    rng = np.random.default_rng(42)
    rs_seed = _rs_poly(_N)
    
    population = [list(rs_seed)]
    for _ in range(pop_size - 1):
        # Start with RS and flip a few random positions
        individual = list(rs_seed)
        num_flips = rng.integers(1, 6)
        flip_indices = rng.choice(_N, size=num_flips, replace=False)
        for idx in flip_indices:
            individual[idx] = -individual[idx]
        population.append(individual)
    
    # Also add some purely random individuals
    for _ in range(pop_size // 4):
        population.append(rng.choice([-1, 1], size=_N).tolist())
    
    best_overall = list(population[0])
    best_peak = _calc_peak_fast(population[0], M, refine_top=5, refine_iters=20)
    
    # Evaluate initial population
    peaks = [best_peak]
    for i in range(1, pop_size):
        if time.time() - start_time > max_time:
            break
        p = _calc_peak_fast(population[i], M, refine_top=3, refine_iters=15)
        peaks.append(p)
        if p < best_peak:
            best_peak = p
            best_overall = list(population[i])
    
    # Genetic algorithm loop
    for gen in range(generations):
        if time.time() - start_time > max_time - 10.0:
            break
        
        # Sort by fitness (lower peak is better)
        sorted_indices = np.argsort(peaks)
        sorted_population = [population[i] for i in sorted_indices]
        sorted_peaks = [peaks[i] for i in sorted_indices]
        
        # Elitism: keep top 20%
        elite_count = max(1, pop_size // 5)
        new_population = [list(ind) for ind in sorted_population[:elite_count]]
        new_peaks = sorted_peaks[:elite_count]
        
        # Fill the rest with offspring
        while len(new_population) < pop_size:
            if time.time() - start_time > max_time - 10.0:
                break
            
            # Tournament selection
            def tournament_select():
                candidates = rng.integers(0, pop_size, size=3)
                best_idx = candidates[np.argmin([peaks[i] for i in candidates])]
                return population[best_idx]
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover: uniform
            mask = rng.integers(0, 2, size=_N)
            child = [parent1[i] if mask[i] == 1 else parent2[i] for i in range(_N)]
            
            # Mutation: flip a few random bits
            mutation_rate = 0.1
            for i in range(_N):
                if rng.random() < mutation_rate:
                    child[i] = -child[i]
            
            # Evaluate child
            p = _calc_peak_fast(child, M, refine_top=3, refine_iters=15)
            new_population.append(child)
            new_peaks.append(p)
            
            if p < best_peak:
                best_peak = p
                best_overall = list(child)
        
        population = new_population
        peaks = new_peaks
    
    return best_overall, best_peak

def construct():
    global _T0
    _T0 = time.time()
    
    # Known good seeds to try
    # Rudin-Shapiro
    seed_rs = _rs_poly(_N)
    
    # Try several known optimal or near-optimal polynomials
    # Based on Odlyzko's tables for n=24
    candidate_seeds = [
        seed_rs,
        [1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1],
        [1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1],
        [1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1],
        [-1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1],
    ]
    
    best_coeffs = seed_rs
    best_peak = _calc_peak(best_coeffs, M=4096, refine_top=10, refine_iters=40)
    
    # First, evaluate all candidates quickly
    for seed in candidate_seeds:
        if _time_left() < 10.0:
            break
        p = _calc_peak(seed, M=2048, refine_top=5, refine_iters=20)
        if p < best_peak - 1e-8:
            best_coeffs = list(seed)
            best_peak = p
    
    # Run local search on the best candidate
    if _time_left() > 30.0:
        res_coeffs, res_peak = _local_search(best_coeffs, M=1024, max_time=min(50.0, _time_left() - 20.0))
        if res_peak < best_peak:
            best_coeffs = res_coeffs
            best_peak = res_peak
    
    # If time remains, try Genetic Algorithm for broader exploration
    if _time_left() > 25.0:
        ga_coeffs, ga_peak = _genetic_search(pop_size=40, generations=80, max_time=min(35.0, _time_left() - 15.0), M=1024)
        if ga_peak < best_peak:
            # Verify and polish
            v_peak = _calc_peak(ga_coeffs, M=4096, refine_top=10, refine_iters=40)
            if v_peak < best_peak:
                best_coeffs = ga_coeffs
                best_peak = v_peak
                # Polish the GA result with local search
                if _time_left() > 15.0:
                    pol_coeffs, pol_peak = _local_search(best_coeffs, M=1024, max_time=min(20.0, _time_left() - 10.0))
                    if pol_peak < best_peak:
                        best_coeffs = pol_coeffs
                        best_peak = pol_peak
    
    # If still time, try a few random restarts with local search
    if _time_left() > 20.0:
        rng = np.random.default_rng(42)
        for _ in range(4):
            if _time_left() < 15.0:
                break
            # Random seed with RS-like structure
            seed_alt = list(seed_rs)
            num_flips = int(rng.integers(1, 4))
            flip_indices = rng.choice(_N, size=num_flips, replace=False)
            for idx in flip_indices:
                seed_alt[idx] = -seed_alt[idx]
            
            r_coeffs, r_peak = _local_search(seed_alt, M=1024, max_time=min(15.0, _time_left() - 10.0))
            if r_peak < best_peak:
                v_peak = _calc_peak(r_coeffs, M=4096, refine_top=10, refine_iters=40)
                if v_peak < best_peak:
                    best_coeffs = r_coeffs
                    best_peak = v_peak

    # Final high-precision polish
    if _time_left() > 5.0:
        final_peak = _calc_peak(best_coeffs, M=8192, refine_top=10, refine_iters=50)
        best_peak = final_peak
    
    # Ensure coefficients are +1 or -1
    best_coeffs = [1 if c >= 0 else -1 for c in best_coeffs]
    
    return best_coeffs

if __name__ == "__main__":
    # Self-test
    coeffs = construct()
    print(coeffs)
    print(len(coeffs))
    print(_calc_peak(coeffs, M=8192, refine_top=10, refine_iters=50))
