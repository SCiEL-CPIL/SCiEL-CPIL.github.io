"""
Combined approach:
1. Fast FFT-based correlation computation.
2. Multi-start L-BFGS-B with log-sum-exp smooth surrogate.
3. Diverse structured initializations.
4. High-precision final polish.
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
    # For linear correlation of length N with shift k, we need FFT of size >= 2N-1
    fft_size = 2 * N - 1
    
    def get_correlations_fast(h):
        """Compute linear correlations of h and 1-h for all non-zero shifts using FFT."""
        g = 1.0 - h
        # FFT-based linear correlation: corr = ifft(fft(h) * conj(fft(g)))
        H = np.fft.fft(h, n=fft_size)
        G = np.fft.fft(g, n=fft_size)
        c = np.real(np.fft.ifft(H * np.conj(G)))
        # c has length fft_size; shifts 0..N-1 are c[0..N-1], shifts -(N-1)..-1 are c[fft_size-N+1..fft_size-1]
        # Exclude shift 0 (index 0)
        c_pos = c[1:N]       # shifts +1 to +(N-1)
        c_neg = c[fft_size-N+1:fft_size]  # shifts -(N-1) to -1
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

    def make_objective(beta, lambda_sum):
        def obj(h):
            c_all = get_correlations_fast(h)
            max_c = np.max(c_all)
            if beta > 0:
                # Numerically stable log-sum-exp
                shifted = beta * (c_all - max_c)
                lse = max_c + np.log(np.sum(np.exp(shifted))) / beta
            else:
                lse = max_c
            
            sum_h = np.sum(h)
            sum_pen = lambda_sum * (sum_h - target_sum)**2
            
            return lse + sum_pen
        return obj

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
    t = np.arange(N)
    for freq in [1, 2, 3, 4, 5, 7, 10, 15]:
        candidates.append(0.5 + 0.4 * np.sin(2 * np.pi * freq * t / N))
        candidates.append(0.5 + 0.4 * np.cos(2 * np.pi * freq * t / N))
    
    # Binary-like patterns with low autocorrelation
    # Thue-Morse sequence
    def thue_morse_seq(n):
        tm = np.zeros(n, dtype=int)
        for i in range(1, n):
            tm[i] = tm[i >> 1] ^ (i & 1)
        return tm
    
    tm = thue_morse_seq(N)
    candidates.append(0.5 + 0.45 * (2 * tm - 1))  # Map {0,1} to {0.05, 0.95}
    
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

    # Phase 2: Multi-start L-BFGS-B optimization
    # Use the best found as one start, plus diverse structured starts
    starts = [best_h.copy()]
    
    # Add more diverse starts
    for _ in range(12):
        # Mix of random and structured
        if rng.random() < 0.5:
            h_start = rng.uniform(0.15, 0.85, N)
        else:
            freq = rng.integers(1, 20)
            phase = rng.uniform(0, 2 * np.pi)
            h_start = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
            # Add noise
            h_start += rng.normal(0, 0.05, N)
        starts.append(project_to_feasible(h_start))
    
    for start_idx, h_start in enumerate(starts):
        if time.time() > deadline - 15.0:
            break
        
        # Use progressively higher beta for better max approximation
        if start_idx == 0:
            beta = 200.0
        elif start_idx < 5:
            beta = 100.0
        else:
            beta = 50.0
        
        obj_func = make_objective(beta, 500.0)
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                h_start, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 300, 'ftol': 1e-11, 'gtol': 1e-9, 'maxls': 50}
            )
            h_opt = project_to_feasible(res.x)
            score_opt = true_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # Phase 3: Final high-precision polish with very high beta
    if time.time() < deadline - 8.0:
        obj_func = make_objective(500.0, 5000.0)
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                best_h, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            h_opt = project_to_feasible(res.x)
            score_opt = true_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Phase 4: Another polish from slightly perturbed best
    if time.time() < deadline - 5.0:
        h_perturbed = best_h + rng.normal(0, 0.01, N)
        h_perturbed = project_to_feasible(h_perturbed)
        
        obj_func = make_objective(500.0, 5000.0)
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                h_perturbed, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            h_opt = project_to_feasible(res.x)
            score_opt = true_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    return best_h
