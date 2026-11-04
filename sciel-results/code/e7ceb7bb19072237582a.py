"""
Improved approach for Erdos minimum overlap:
1. Epigraph formulation with auxiliary variable t to handle the max constraint smoothly.
2. Multi-start L-BFGS-B with diverse initializations.
3. Final high-precision polish pass.
"""
import numpy as np
import time
import scipy.optimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    target_sum = N / 2.0
    bounds_h = [(0.0, 1.0) for _ in range(N)]
    bounds_t = [(0.0, 2.0)]
    bounds = bounds_h + bounds_t
    
    def get_correlations(h):
        """Compute linear correlations of h and 1-h for all non-zero shifts."""
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # Exclude center (shift 0)
        c1 = c[:N-1]
        c2 = c[N:]
        return np.concatenate([c1, c2])

    def true_score(h):
        """Calculate the raw score (max overlap)."""
        c_all = get_correlations(h)
        return np.max(c_all)

    def project_to_feasible(h):
        """Project h to satisfy sum(h) = N/2 and 0 <= h <= 1."""
        h = np.clip(h, 0.0, 1.0)
        current_sum = np.sum(h)
        diff = target_sum - current_sum
        
        if abs(diff) > 1e-12:
            for _ in range(50):
                if abs(diff) < 1e-12:
                    break
                # Adjust only elements not at bounds
                not_at_bound = (h > 1e-9) & (h < 1.0 - 1e-9)
                if not np.any(not_at_bound):
                    break
                h[not_at_bound] += diff / np.sum(not_at_bound)
                h = np.clip(h, 0.0, 1.0)
                current_sum = np.sum(h)
                diff = target_sum - current_sum
        return h

    def make_objective(beta, lambda_sum):
        """
        Epigraph formulation:
        Variables: h (heights), t (upper bound on max overlap)
        Objective: t + penalty(sum) + soft_max(C_k(h) - t)
        """
        def obj(h_t):
            h = h_t[:N]
            t = h_t[N]
            
            c_all = get_correlations(h)
            
            # Soft max of (c_all - t) using log-sum-exp
            shifted = c_all - t
            max_shifted = np.max(shifted)
            if beta > 0:
                # log-sum-exp trick for numerical stability
                lse = max_shifted + np.log(np.sum(np.exp(beta * (shifted - max_shifted)))) / beta
            else:
                lse = max_shifted
            
            sum_h = np.sum(h)
            sum_pen = lambda_sum * (sum_h - target_sum)**2
            
            return t + lse + sum_pen
        return obj

    # Initial candidates
    rng = np.random.default_rng(123)
    candidates = []
    
    # Uniform
    candidates.append(np.full(N, 0.5))
    
    # Random
    for _ in range(8):
        h_rand = rng.uniform(0.1, 0.9, N)
        candidates.append(h_rand)
        
    # Sine waves with various frequencies
    t_arr = np.arange(N)
    for freq in [1, 2, 3, 5, 7, 10, 15]:
        for phase in [0, np.pi/2, np.pi]:
            candidates.append(0.5 + 0.4 * np.sin(2 * np.pi * freq * t_arr / N + phase))
            candidates.append(0.5 + 0.4 * np.cos(2 * np.pi * freq * t_arr / N + phase))
    
    # Binary-like patterns
    # Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    candidates.append(h_alt)
    
    # Block patterns
    for block_size in [2, 4, 5, 10]:
        h_block = np.zeros(N)
        for i in range(0, N, block_size * 2):
            h_block[i:i+block_size] = 1.0
        candidates.append(h_block)
    
    # Thue-Morse inspired
    def thue_morse_sequence(n):
        tm = [0]
        for _ in range(n - 1):
            tm.extend([1 - x for x in tm])
        return np.array(tm[:n], dtype=float)
    
    h_tm = thue_morse_sequence(N)
    candidates.append(h_tm)
    
    # Van der Corput sequence
    def van_der_corput(n, base=2):
        vdc = np.zeros(n)
        for i in range(1, n):
            denom = base ** i
            vdc[i] = (i // denom) % base / denom + vdc[i-1] if i > 0 else 0
        # Better implementation
        vdc = np.zeros(n)
        for i in range(n):
            x = 0.0
            f = 1.0 / base
            j = i
            while j > 0:
                x += f * (j % base)
                j //= base
                f /= base
            vdc[i] = x
        return vdc
        
    h_vdc = van_der_corput(N)
    h_vdc = (h_vdc - np.min(h_vdc)) / (np.max(h_vdc) - np.min(h_vdc) + 1e-9)
    candidates.append(h_vdc)
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Evaluate candidates
    for h_cand in candidates:
        h_proj = project_to_feasible(h_cand)
        score = true_score(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj

    # Phase 2: Multi-start L-BFGS-B with epigraph formulation
    num_starts = 20
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 15.0:
            break
            
        # Generate diverse start
        if start_idx == 0:
            h_curr = best_h.copy()
        elif start_idx % 4 == 0:
            # Random uniform
            h_curr = rng.uniform(0.2, 0.8, N)
        elif start_idx % 4 == 1:
            # Sine wave with random parameters
            freq = rng.integers(1, 20)
            phase = rng.uniform(0, 2 * np.pi)
            t_arr = np.arange(N)
            h_curr = 0.5 + 0.35 * np.sin(2 * np.pi * freq * t_arr / N + phase)
        elif start_idx % 4 == 2:
            # Random walk
            h_curr = np.cumsum(rng.normal(0, 0.15, N))
            h_curr = (h_curr - np.min(h_curr)) / (np.max(h_curr) - np.min(h_curr) + 1e-9)
            h_curr = 0.15 + 0.7 * h_curr
        else:
            # Perturbed best
            h_curr = best_h + rng.normal(0, 0.1, N)
            
        h_curr = project_to_feasible(h_curr)
        
        # Initialize t with a reasonable upper bound
        c_all_init = get_correlations(h_curr)
        t_init = np.max(c_all_init) + 0.1
        
        h_t_init = np.concatenate([h_curr, [t_init]])
        
        # Use L-BFGS-B with moderate beta for global-ish search
        obj_func = make_objective(50.0, 200.0)
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                h_t_init, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            h_t_opt = res.x
            h_opt = h_t_opt[:N]
            h_opt = project_to_feasible(h_opt)
            score_opt = true_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
                
        except Exception:
            pass

    # Phase 3: Final high-precision polish on the best found
    if time.time() < deadline - 8.0:
        # Use a very high beta to approximate max closely
        obj_func = make_objective(500.0, 5000.0)
        c_all_best = get_correlations(best_h)
        t_best = np.max(c_all_best) + 0.01
        h_t_best = np.concatenate([best_h, [t_best]])
        
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                h_t_best, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            h_t_opt = res.x
            h_opt = h_t_opt[:N]
            h_opt = project_to_feasible(h_opt)
            score_opt = true_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
        except Exception:
            pass
            
        # One more polish with even tighter tolerance if time permits
        if time.time() < deadline - 4.0:
            try:
                res = scipy.optimize.minimize(
                    obj_func, 
                    np.concatenate([best_h, [np.max(get_correlations(best_h)) + 0.001]]), 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                h_t_opt = res.x
                h_opt = h_t_opt[:N]
                h_opt = project_to_feasible(h_opt)
                score_opt = true_score(h_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt
            except Exception:
                pass

    # Ensure the result is valid
    best_h = project_to_feasible(best_h)
    return best_h
