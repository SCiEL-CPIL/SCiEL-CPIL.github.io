import numpy as np
import scipy.optimize
import time

def construct():
    """
    Minimize the maximum overlap of h with shifted (1-h).
    
    Uses linear (non-circular) correlation: C(k) = sum_i h[i]*(1-h[i+k])
    Optimized via L-BFGS-B on log-sum-exp smooth max + simulated annealing + coordinate descent polish.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator kills at 150s)
    
    N = 100
    target_sum = N / 2.0
    
    def get_linear_correlations(h):
        """
        Returns array of length 2*(N-1) with:
          indices 0..N-2: C_neg(k) = sum_i h[i]*(1-h[i-k]) for k=1..N-1 (negative shifts)
          indices N-1..2N-3: C_pos(k) = sum_i h[i]*(1-h[i+k]) for k=1..N-1 (positive shifts)
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # c[N-1] is shift 0
        # c[N-1-k] for k=1..N-1 gives negative shifts
        # c[N-1+k] for k=1..N-1 gives positive shifts
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero
    
    def max_overlap(h):
        c = get_linear_correlations(h)
        return np.max(c)
    
    def project(h):
        """Project to 0<=h<=1, sum(h)=N/2 using bisection for exact feasibility."""
        h = np.clip(h, 0, 1)
        
        # If sum is already correct, return
        if abs(np.sum(h) - target_sum) < 1e-12:
            return h
        
        # Bisection on offset to find exact feasible point
        def f(offset):
            return np.sum(np.clip(h + offset, 0, 1)) - target_sum
        
        # Find bracket
        lo, hi = -1.0, 1.0
        flo, fhi = f(lo), f(hi)
        
        if flo * fhi > 0:
            # No bracket, fall back to iterative
            for _ in range(50):
                diff = target_sum - np.sum(h)
                if abs(diff) < 1e-14:
                    break
                mask = (h > 1e-10) & (h < 1.0 - 1e-10)
                if not np.any(mask):
                    break
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
            return h
        
        # Bisection
        for _ in range(60):
            mid = (lo + hi) / 2
            fm = f(mid)
            if abs(fm) < 1e-14:
                return np.clip(h + mid, 0, 1)
            if flo * fm <= 0:
                hi = mid
                fhi = fm
            else:
                lo = mid
                flo = fm
        
        return np.clip(h + (lo + hi) / 2, 0, 1)
    
    def lse_objective(h, beta):
        """
        LogSumExp of correlations for smooth approximation of max.
        """
        c_all = get_linear_correlations(h)
        max_c = np.max(c_all)
        if max_c > 500:
            max_c = 500.0
        # Stable LSE
        lse = max_c + np.log(np.sum(np.exp(beta * (c_all - max_c)))) / beta
        return lse
    
    def optimize_instance(h0, beta, max_iter, ftol=1e-10, gtol=1e-10):
        """
        Optimize using L-BFGS-B on the smooth surrogate.
        """
        bounds = [(0, 1) for _ in range(N)]
        
        def obj(h):
            h_proj = project(h)
            return lse_objective(h_proj, beta)
        
        try:
            res = scipy.optimize.minimize(
                obj, 
                h0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0
        
        h_opt = project(h_opt)
        return h_opt
    
    # Initialize with multiple random starts
    rng = np.random.default_rng(42)
    
    best_h = None
    best_score = np.inf
    
    def find_clusters(h, threshold=0.08):
        clusters = []
        i = 0
        while i < N:
            j = i
            base_val = h[i]
            while j < N and abs(h[j] - base_val) < threshold:
                j += 1
            if j > i:
                clusters.append((i, j))
            i = j
        return clusters
    
    def simulated_annealing(h_init, n_iter, T_init, T_final, rng):
        h_curr = h_init.copy()
        local_best_h = h_curr.copy()
        local_best_score = max_overlap(local_best_h)
        
        for it in range(n_iter):
            if time.time() > deadline - 20:
                break
            
            progress = it / n_iter
            T = T_init * (T_final / T_init) ** progress
            
            move_type = rng.random()
            if move_type < 0.30:
                # Cluster shift
                clusters = find_clusters(h_curr, threshold=0.06)
                if clusters:
                    idx = rng.integers(0, len(clusters))
                    i, j = clusters[idx]
                    h_prop = h_curr.copy()
                    shift = rng.uniform(-0.2, 0.2)
                    h_prop[i:j] += shift
                    h_prop = project(h_prop)
                else:
                    h_prop = h_curr.copy()
                    i = rng.integers(0, N)
                    h_prop[i] += rng.uniform(-0.15, 0.15)
                    h_prop = project(h_prop)
            elif move_type < 0.55:
                # Antiferromagnetic flip on a block
                i = rng.integers(0, N - 3)
                length = rng.integers(2, 20)
                j = min(i + length, N)
                h_prop = h_curr.copy()
                h_prop[i:j] = 1.0 - h_curr[i:j]
                h_prop = project(h_prop)
            elif move_type < 0.80:
                # Pair swap
                i, j = rng.integers(0, N, size=2)
                h_prop = h_curr.copy()
                delta = rng.uniform(-0.12, 0.12)
                h_prop[i] += delta
                h_prop[j] -= delta
                h_prop = project(h_prop)
            elif move_type < 0.95:
                # Random restart of a segment
                i = rng.integers(0, N - 5)
                length = rng.integers(3, 25)
                j = min(i + length, N)
                h_prop = h_curr.copy()
                h_prop[i:j] = rng.uniform(0.1, 0.9, size=j-i)
                h_prop = project(h_prop)
            else:
                # Large perturbation
                i = rng.integers(0, N - 10)
                length = rng.integers(10, 30)
                j = min(i + length, N)
                h_prop = h_curr.copy()
                h_prop[i:j] = rng.uniform(0.0, 1.0, size=j-i)
                h_prop = project(h_prop)
            
            score_curr = max_overlap(h_curr)
            score_prop = max_overlap(h_prop)
            
            delta = score_prop - score_curr
            if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                h_curr = h_prop
            
            curr_score = max_overlap(h_curr)
            if curr_score < local_best_score:
                local_best_score = curr_score
                local_best_h = h_curr.copy()
        
        return local_best_h, local_best_score
    
    # Stage 1: L-BFGS-B multi-restart with different initializations
    n_restarts = 6
    restart_configs = [
        (np.full(N, 0.5), 5.0, 300),
        (project(np.tile([0.3, 0.7], N // 2)), 10.0, 300),
        (project(np.linspace(0.2, 0.8, N)), 15.0, 300),
        (project(np.sin(np.linspace(0, 2*np.pi, N)) * 0.3 + 0.5), 20.0, 300),
        (project(rng.uniform(0.1, 0.9, N)), 25.0, 300),
        (project(np.where(rng.random(N) > 0.5, 0.8, 0.2)), 30.0, 300),
    ]
    
    for r in range(n_restarts):
        if time.time() > deadline - 40:
            break
        
        h_init, beta, max_iter = restart_configs[r]
        h_init = h_init.copy()
        
        h_opt = optimize_instance(h_init, beta, max_iter)
        score_opt = max_overlap(h_opt)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
    
    # Stage 2: Simulated annealing from best L-BFGS-B result
    if time.time() < deadline - 25 and best_h is not None:
        h_sa, score_sa = simulated_annealing(best_h, 800, 0.15, 0.0003, rng)
        
        if score_sa < best_score:
            best_score = score_sa
            best_h = h_sa.copy()
    
    # Stage 3: Coordinate descent polish
    if time.time() < deadline - 10 and best_h is not None:
        h = best_h.copy()
        current_score = max_overlap(h)
        
        for epoch in range(3):
            if time.time() > deadline - 5:
                break
            
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                
                old_val = h[i]
                best_local = old_val
                best_local_score = current_score
                
                # Try a range of values
                for trial in np.linspace(max(0, old_val - 0.3), min(1, old_val + 0.3), 11):
                    h_trial = h.copy()
                    h_trial[i] = trial
                    h_trial = project(h_trial)
                    s = max_overlap(h_trial)
                    if s < best_local_score:
                        best_local_score = s
                        best_local = trial
                
                if best_local != old_val:
                    h[i] = best_local
                    h = project(h)
                    current_score = max_overlap(h)
                    if current_score < best_score:
                        best_score = current_score
                        best_h = h.copy()
    
    # Stage 4: High-precision L-BFGS-B polish with very high beta
    if time.time() < deadline - 5 and best_h is not None:
        polish_beta = 80.0
        h_polished = optimize_instance(best_h, polish_beta, 500, ftol=1e-12, gtol=1e-12)
        polished_score = max_overlap(h_polished)
        
        if polished_score < best_score:
            best_score = polished_score
            best_h = h_polished.copy()
    
    # Final: ensure validity
    best_h = project(best_h)
    
    # Verify sum constraint with high precision
    diff = target_sum - best_h.sum()
    if abs(diff) > 1e-10:
        def f(offset):
            return np.sum(np.clip(best_h + offset, 0, 1)) - target_sum
        
        lo, hi = -0.1, 0.1
        for _ in range(50):
            mid = (lo + hi) / 2
            if f(lo) * f(mid) <= 0:
                hi = mid
            else:
                lo = mid
        
        best_h = np.clip(best_h + (lo + hi) / 2, 0, 1)
    
    # Final clip to ensure bounds
    best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
