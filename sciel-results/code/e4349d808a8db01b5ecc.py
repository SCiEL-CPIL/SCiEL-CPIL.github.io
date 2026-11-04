"""
Combined approach:
1. Fast FFT-based correlation computation.
2. Multi-start L-BFGS-B with epigraph formulation (auxiliary t variable).
3. Diverse structured initializations.
4. High-precision final polish with coordinate descent.
"""
import numpy as np
import time
import scipy.optimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    target_sum = N / 2.0
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Precompute FFT sizes and indices for fast correlation
    fft_size = 2 * N - 1
    
    def get_correlations_fast(h):
        """Compute linear correlations of h and 1-h for all non-zero shifts using FFT."""
        g = 1.0 - h
        H = np.fft.fft(h, n=fft_size)
        G = np.fft.fft(g, n=fft_size)
        c = np.real(np.fft.ifft(H * np.conj(G)))
        c_pos = c[1:N]
        c_neg = c[fft_size-N+1:fft_size]
        return np.concatenate([c_neg, c_pos])

    def true_score(h):
        """Calculate the raw score (max overlap)."""
        c_all = get_correlations_fast(h)
        return np.max(c_all)

    def project_to_feasible(h):
        """Project h to satisfy sum(h) = N/2 and 0 <= h <= 1."""
        h = np.clip(h, 0.0, 1.0)
        current_sum = np.sum(h)
        diff = target_sum - current_sum
        
        if abs(diff) > 1e-12:
            for _ in range(50):
                if abs(diff) < 1e-14:
                    break
                h += diff / N
                h = np.clip(h, 0.0, 1.0)
                current_sum = np.sum(h)
                diff = target_sum - current_sum
        return h

    def make_objective_epigraph(beta, lambda_sum):
        """
        Epigraph formulation: minimize t + penalty
        subject to c_i <= t for all i (softened with log-sum-exp).
        Variables: [h_0, ..., h_{N-1}, t]
        """
        def obj(x):
            h = x[:N]
            t = x[N]
            
            c_all = get_correlations_fast(h)
            
            # Smooth max: log-sum-exp approximation of max(c_i - t, 0)
            # We want to penalize c_i > t
            excess = c_all - t
            # Use soft-max: log(sum(exp(beta * excess))) / beta
            # If all excess <= 0, this is small; if any excess > 0, grows
            max_excess = np.max(excess)
            if max_excess > 0:
                shifted = beta * (excess - max_excess)
                lse = max_excess + np.log(np.sum(np.exp(shifted))) / beta
            else:
                lse = 0.0
            
            sum_h = np.sum(h)
            sum_pen = lambda_sum * (sum_h - target_sum)**2
            
            return t + beta * lse + sum_pen
        return obj

    def make_objective_lse(beta, lambda_sum):
        """Original LSE formulation for comparison."""
        def obj(h):
            c_all = get_correlations_fast(h)
            max_c = np.max(c_all)
            if beta > 0:
                shifted = beta * (c_all - max_c)
                lse = max_c + np.log(np.sum(np.exp(shifted))) / beta
            else:
                lse = max_c
            
            sum_h = np.sum(h)
            sum_pen = lambda_sum * (sum_h - target_sum)**2
            
            return lse + sum_pen
        return obj

    def coordinate_descent_polish(h, max_iters=500, tol=1e-8):
        """
        Local coordinate descent: try small perturbations of each h_i
        to reduce the max overlap while maintaining feasibility.
        """
        h = h.copy()
        best_score = true_score(h)
        
        for it in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            
            improved = False
            # Randomly permute indices for diversity
            indices = np.random.permutation(N)
            
            for i in indices:
                # Try a few step sizes
                for step in [0.01, 0.005, 0.001, -0.01, -0.005, -0.001]:
                    h_trial = h.copy()
                    h_trial[i] += step
                    
                    # Check bounds
                    if h_trial[i] < 0 or h_trial[i] > 1:
                        continue
                    
                    # Adjust to maintain sum constraint
                    diff = target_sum - np.sum(h_trial)
                    if abs(diff) > 1e-10:
                        # Distribute correction to other elements
                        others = [j for j in range(N) if j != i]
                        h_trial[others] += diff / (N - 1)
                        h_trial = np.clip(h_trial, 0.0, 1.0)
                        # Re-fix sum
                        diff2 = target_sum - np.sum(h_trial)
                        h_trial += diff2 / N
                        h_trial = np.clip(h_trial, 0.0, 1.0)
                    
                    score_trial = true_score(h_trial)
                    if score_trial < best_score - 1e-10:
                        h = h_trial
                        best_score = score_trial
                        improved = True
                        break
            
            if not improved:
                break
        
        return h, best_score

    # Initial candidates
    rng = np.random.default_rng(42)
    candidates = []
    
    # Uniform
    candidates.append(np.full(N, 0.5))
    
    # Random
    for _ in range(8):
        h_rand = rng.uniform(0.1, 0.9, N)
        candidates.append(h_rand)
        
    # Sine waves with various frequencies
    t_arr = np.arange(N)
    for freq in [1, 2, 3, 4, 5, 7, 10, 15]:
        candidates.append(0.5 + 0.4 * np.sin(2 * np.pi * freq * t_arr / N))
        candidates.append(0.5 + 0.4 * np.cos(2 * np.pi * freq * t_arr / N))
    
    # Thue-Morse sequence
    def thue_morse_seq(n):
        tm = np.zeros(n, dtype=int)
        for i in range(1, n):
            tm[i] = tm[i >> 1] ^ (i & 1)
        return tm
    
    tm = thue_morse_seq(N)
    candidates.append(0.5 + 0.45 * (2 * tm - 1))
    
    # Random binary
    h_bin = rng.integers(0, 2, N).astype(float)
    candidates.append(0.5 + 0.45 * (2 * h_bin - 1))
    
    # Van der Corput low-discrepancy sequence
    def van_der_corput(n, base=2):
        vdc = np.zeros(n)
        for i in range(1, n):
            f = 1.0
            j = i
            while j > 0:
                f /= base
                vdc[i] += f * (j % base)
                j //= base
        return vdc
    
    h_vdc = van_der_corput(N)
    h_vdc = (h_vdc - np.min(h_vdc)) / (np.max(h_vdc) - np.min(h_vdc) + 1e-9)
    candidates.append(h_vdc)
    
    # Halving pattern
    h_half = np.zeros(N)
    h_half[::2] = 0.9
    h_half[1::2] = 0.1
    candidates.append(h_half)

    best_h = None
    best_score = float('inf')
    
    # Phase 1: Evaluate candidates
    for h_cand in candidates:
        h_proj = project_to_feasible(h_cand)
        score = true_score(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj.copy()

    # Phase 2: Multi-start optimization using epigraph formulation
    starts = [best_h.copy()]
    
    # Add more diverse starts
    for _ in range(10):
        if rng.random() < 0.5:
            h_start = rng.uniform(0.15, 0.85, N)
        else:
            freq = rng.integers(1, 20)
            phase = rng.uniform(0, 2 * np.pi)
            h_start = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
            h_start += rng.normal(0, 0.05, N)
        starts.append(project_to_feasible(h_start))
    
    for start_idx, h_start in enumerate(starts):
        if time.time() > deadline - 20.0:
            break
        
        # Use epigraph formulation with varying beta
        if start_idx == 0:
            beta = 50.0
        elif start_idx < 5:
            beta = 30.0
        else:
            beta = 20.0
        
        # Initialize t to current max overlap
        t_init = true_score(h_start)
        x0 = np.concatenate([h_start, [t_init]])
        bounds_ext = bounds + [(0.0, 2.0)]  # t in [0, 2]
        
        obj_func = make_objective_epigraph(beta, 500.0)
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds_ext, 
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 40}
            )
            h_opt = project_to_feasible(res.x[:N])
            score_opt = true_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # Phase 3: Coordinate descent polish
    if time.time() < deadline - 10.0:
        h_polished, score_polished = coordinate_descent_polish(best_h, max_iters=200)
        if score_polished < best_score:
            best_score = score_polished
            best_h = h_polished.copy()

    # Phase 4: Final high-precision polish with LSE (very high beta)
    if time.time() < deadline - 5.0:
        obj_func = make_objective_lse(500.0, 5000.0)
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                best_h, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 80}
            )
            h_opt = project_to_feasible(res.x)
            score_opt = true_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Phase 5: Another coordinate descent pass from slightly perturbed best
    if time.time() < deadline - 3.0:
        h_perturbed = best_h + rng.normal(0, 0.005, N)
        h_perturbed = project_to_feasible(h_perturbed)
        h_final, score_final = coordinate_descent_polish(h_perturbed, max_iters=100)
        if score_final < best_score:
            best_score = score_final
            best_h = h_final.copy()

    return best_h
