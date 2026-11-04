import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 100
TARGET_SUM = N / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    def get_correlations_linear(h):
        """
        Compute linear cross-correlation of h with (1-h) for all non-zero shifts.
        Returns array of C_k for k != 0.
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # c[N-1] is zero-shift. We exclude it.
        # c[0:N-1] are negative shifts, c[N:2N-1] are positive shifts.
        return np.concatenate([c[:N-1], c[N:]])

    def max_overlap(h):
        """Return the maximum overlap value."""
        c = get_correlations_linear(h)
        if len(c) == 0:
            return 0.0
        return np.max(c)

    def project(h):
        """
        Project h onto feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
        Uses bisection on a Lagrange multiplier for exact projection.
        """
        h = np.clip(h, 0.0, 1.0).copy()
        current = np.sum(h)
        
        if abs(current - TARGET_SUM) < 1e-12:
            return h
        
        # Bisection on lambda: find lambda such that sum(clip(h - lambda, 0, 1)) = TARGET_SUM
        # When lambda < 0, we're adding; when lambda > 0, we're subtracting
        lo, hi = -max(h) - 1, 1 - min(h) + 1
        
        def sum_at_lambda(lam):
            return np.sum(np.clip(h - lam, 0, 1))
        
        for _ in range(60):
            mid = (lo + hi) / 2
            s = sum_at_lambda(mid)
            if abs(s - TARGET_SUM) < 1e-12:
                break
            if s > TARGET_SUM:
                lo = mid
            else:
                hi = mid
        
        lam = (lo + hi) / 2
        h = np.clip(h - lam, 0, 1)
        
        # Final adjustment for any residual error
        diff = TARGET_SUM - np.sum(h)
        if abs(diff) > 1e-14:
            # Distribute residual to elements not at bounds
            mask = (h > 1e-10) & (h < 1 - 1e-10)
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
        
        return h

    def objective_lse(x, beta, lam_sum):
        """
        Objective function using LogSumExp approximation of the max overlap.
        """
        h = np.clip(x, 0.0, 1.0)
        overlaps = get_correlations_linear(h)
        
        if len(overlaps) == 0:
            max_ov = 0.0
            lse = 0.0
        else:
            max_ov = np.max(overlaps)
            shifted = overlaps - max_ov
            lse = max_ov + logsumexp(beta * shifted) / beta
        
        # Penalty for sum constraint
        sum_err = np.sum(h) - TARGET_SUM
        sum_penalty = lam_sum * (sum_err ** 2)
        
        return lse + sum_penalty

    def objective_epigraph(x, lam_sum, t_init=0.5):
        """
        Epigraph formulation: minimize t + penalty, subject to overlaps <= t.
        x = [h_1, ..., h_N, t]
        """
        h = np.clip(x[:N], 0.0, 1.0)
        t = x[N]
        
        overlaps = get_correlations_linear(h)
        
        if len(overlaps) == 0:
            constraint_violation = 0.0
        else:
            # Smooth approximation of max(0, overlaps - t)
            violation = np.maximum(0, overlaps - t)
            constraint_violation = np.sum(violation ** 2)
        
        sum_err = np.sum(h) - TARGET_SUM
        sum_penalty = lam_sum * (sum_err ** 2)
        
        return t + 100.0 * constraint_violation + sum_penalty

    def run_optimization_lse(x0, beta, lam_sum, maxiter=200):
        """
        Run L-BFGS-B optimization from x0 with given parameters.
        """
        bounds = [(0.0, 1.0) for _ in range(N)]
        
        def obj_func(x):
            return objective_lse(x, beta, lam_sum)
        
        try:
            result = minimize(
                obj_func, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            return result.x
        except Exception:
            return x0

    def run_optimization_epigraph(x0, lam_sum, t_init=0.5, maxiter=200):
        """
        Run L-BFGS-B optimization with epigraph formulation.
        """
        # Extend x0 with initial t
        x0_ext = np.concatenate([x0, [t_init]])
        
        bounds = [(0.0, 1.0) for _ in range(N)] + [(0.0, 2.0)]
        
        def obj_func(x):
            return objective_epigraph(x, lam_sum, t_init)
        
        try:
            result = minimize(
                obj_func, x0_ext, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            return result.x[:N]  # Return only h, not t
        except Exception:
            return x0

    def local_search_coordinate(h, max_iters=500):
        """
        Perform coordinate descent local search.
        Tries adjusting individual coordinates and re-projecting.
        """
        current_h = h.copy()
        current_score = max_overlap(current_h)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            improved = False
            # Try adjusting each coordinate slightly
            for i in range(N):
                # Try a few small steps
                for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                    new_h = current_h.copy()
                    new_h[i] += delta
                    new_h = project(new_h)
                    
                    new_score = max_overlap(new_h)
                    if new_score < current_score - 1e-12:
                        current_h = new_h
                        current_score = new_score
                        improved = True
                        break
                        
            if not improved:
                break
                
        return current_h, current_score

    def local_search_swap(h, max_iters=300):
        """
        Local search using swap moves: exchange values between pairs of coordinates.
        This can escape local minima that coordinate descent gets stuck in.
        """
        current_h = h.copy()
        current_score = max_overlap(current_h)
        rng_local = np.random.default_rng(123)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            improved = False
            # Try random pairs of swaps
            for _ in range(50):
                i, j = rng_local.choice(N, size=2, replace=False)
                if i == j:
                    continue
                    
                new_h = current_h.copy()
                # Swap with some mixing
                mix = rng_local.uniform(0.1, 0.9)
                temp = new_h[i]
                new_h[i] = mix * temp + (1 - mix) * new_h[j]
                new_h[j] = (1 - mix) * temp + mix * new_h[j]
                new_h = project(new_h)
                
                new_score = max_overlap(new_h)
                if new_score < current_score - 1e-12:
                    current_h = new_h
                    current_score = new_score
                    improved = True
                    break
                    
            if not improved:
                break
                
        return current_h, current_score

    # --- Initialization ---
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.full(N, 0.5))
    
    # 2. Alternating binary-like (smoothed)
    h_alt = np.zeros(N)
    h_alt[::2] = 0.6
    h_alt[1::2] = 0.4
    candidates.append(h_alt)
    
    # 3. Stronger alternating patterns
    h_alt2 = np.zeros(N)
    h_alt2[::2] = 0.8
    h_alt2[1::2] = 0.2
    candidates.append(h_alt2)
    
    h_alt3 = np.zeros(N)
    h_alt3[::2] = 0.9
    h_alt3[1::2] = 0.1
    candidates.append(h_alt3)
    
    # 4. Block patterns
    for block_size in [2, 3, 4, 5]:
        h_block = np.zeros(N)
        for i in range(0, N, block_size * 2):
            h_block[i:i+block_size] = 0.7
            h_block[i+block_size:i+2*block_size] = 0.3
        candidates.append(h_block)
        
    # 5. Smooth "blob" patterns (to encourage positive autocorrelation)
    t = np.arange(N)
    for center in [0, N//4, N//2, 3*N//4]:
        for width in [5, 10, 20]:
            h_blob = 0.5 + 0.3 * np.exp(-((t - center)**2) / (2 * width**2))
            # Shift to mean 0.5? The projection will handle the sum.
            candidates.append(h_blob)

    # 6. Sinusoidal patterns
    for freq in range(1, 10):
        for amp in [0.1, 0.2, 0.3, 0.4]:
            for phase in [0, np.pi/2]:
                trial = 0.5 + amp * np.sin(2*np.pi*freq*t/N + phase)
                candidates.append(trial.copy())
                
    # 7. Random
    for _ in range(20):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        candidates.append(h_rand)
        
    # 8. Sturmian-like sequences (quasi-periodic)
    def sturmian_sequence(alpha, N_pts):
        """Generate a Sturmian sequence with slope alpha."""
        seq = np.zeros(N_pts)
        x = 0.0
        for i in range(N_pts):
            x += alpha
            if x >= 1.0:
                x -= 1.0
                seq[i] = 1.0
            else:
                seq[i] = 0.0
        return seq
    
    for alpha in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        s = sturmian_sequence(alpha, N)
        # Smooth it a bit
        candidates.append(0.3 + 0.4 * s)
        
    # Evaluate and select best initial candidate
    best_h = None
    best_score = np.inf
    
    for h_cand in candidates:
        h_proj = project(h_cand)
        score = max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
            
    # --- Optimization Phases ---
    # We increase beta to sharpen the max approximation.
    # We also increase the sum penalty lambda to enforce constraint more strictly as we converge.
    
    beta_schedule = [5, 10, 20, 50, 100, 200]
    
    current_h = best_h.copy()
    
    for beta in beta_schedule:
        if time.time() > deadline - 15.0:
            break
            
        # Increase lambda as beta increases to keep sum constraint tight
        lam_sum = 10.0 * (beta / 5.0)
        
        # Run LSE optimization from current best
        new_h = run_optimization_lse(current_h, beta, lam_sum, maxiter=300)
        
        # Project to ensure strict feasibility for evaluation
        new_h_proj = project(new_h)
        score_new = max_overlap(new_h_proj)
        
        if score_new < best_score:
            best_score = score_new
            best_h = new_h_proj
            current_h = new_h_proj
        
        # Also try epigraph formulation from current best
        if time.time() < deadline - 10.0:
            t_init = max_overlap(current_h) + 0.1
            new_h_epi = run_optimization_epigraph(current_h, lam_sum, t_init, maxiter=200)
            new_h_epi_proj = project(new_h_epi)
            score_epi = max_overlap(new_h_epi_proj)
            
            if score_epi < best_score:
                best_score = score_epi
                best_h = new_h_epi_proj
                current_h = new_h_epi_proj
        
        # Try a few random restarts at lower beta to escape local minima
        if beta <= 20:
            for _ in range(3):
                if time.time() > deadline - 15.0:
                    break
                x0 = rng.uniform(0.2, 0.8, size=N)
                x0 = project(x0)
                new_h = run_optimization_lse(x0, beta, lam_sum, maxiter=100)
                new_h_proj = project(new_h)
                score_new = max_overlap(new_h_proj)
                if score_new < best_score:
                    best_score = score_new
                    best_h = new_h_proj
                    current_h = new_h_proj

    # --- Local Search Phase ---
    # Use coordinate descent to polish the gradient-optimized solution
    if time.time() < deadline - 15.0:
        ls_h, ls_score = local_search_coordinate(best_h, max_iters=1000)
        if ls_score < best_score:
            best_score = ls_score
            best_h = ls_h

    # Try swap-based local search
    if time.time() < deadline - 10.0:
        swap_h, swap_score = local_search_swap(best_h, max_iters=500)
        if swap_score < best_score:
            best_score = swap_score
            best_h = swap_h

    # --- Final Polish ---
    # Use very high beta and high lambda for final precision
    if time.time() < deadline - 5.0:
        beta_final = 500
        lam_final = 1000.0
        new_h = run_optimization_lse(best_h, beta_final, lam_final, maxiter=500)
        new_h_proj = project(new_h)
        score_new = max_overlap(new_h_proj)
        
        if score_new < best_score:
            best_score = score_new
            best_h = new_h_proj
        
        # Also try epigraph for final polish
        if time.time() < deadline - 3.0:
            t_init = max_overlap(best_h) + 0.05
            new_h_epi = run_optimization_epigraph(best_h, lam_final, t_init, maxiter=300)
            new_h_epi_proj = project(new_h_epi)
            score_epi = max_overlap(new_h_epi_proj)
            
            if score_epi < best_score:
                best_score = score_epi
                best_h = new_h_epi_proj

    # Final Validation
    best_h = project(best_h)
    
    # Ensure strict constraints
    assert np.all(best_h >= 0) and np.all(best_h <= 1), "h out of bounds"
    
    # Re-project one more time with tighter tolerance if needed
    diff = TARGET_SUM - np.sum(best_h)
    if abs(diff) > 1e-9:
        mask = (best_h > 1e-6) & (best_h < 1 - 1e-6)
        if np.any(mask):
            adj = diff / np.sum(mask)
            best_h[mask] += adj
            best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
