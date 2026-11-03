import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """
    Compute the score: L2^2 / (L1 * Sup(autoconv)) / DELTA^2
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max < 1e-15:
        return 0.0
    return (l2_sq / (l1 * autoconv_max)) / (DELTA**2)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    # 1. Initialization: Uniform, Random, and Sidon-like
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Try a few random initializations
    for _ in range(20):
        h_rand = np.random.exponential(1.0, N)
        h_rand /= np.sum(h_rand)
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand

    # 2. Optimization Strategy:
    # We want to maximize J(h) = sum(h^2) / max_conv(h) subject to sum(h)=1, h>=0.
    # Note: The ratio R = (L2^2 / (L1 * Sup)) / DELTA^2. With L1=1, R = (L2^2 / Sup) / DELTA^2.
    # Maximizing L2^2 / Sup is equivalent to minimizing Sup / L2^2.
    # Sup(h*h) = max_k sum_i h_i h_{i+k}.
    # This is a non-convex optimization problem.
    
    # Approach A: Smoothed Max with increasing temperature (Annealing)
    # f(h) = - (sum(h^2) / soft_max(beta, h*h))
    # We minimize this.
    
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
    bounds = [(0.0, 1.0)] * N

    current_h = best_h.copy()
    
    # Schedule of betas (inverse temperature)
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
            
        def smooth_obj(h):
            h = np.clip(h, 0, None)
            s = np.sum(h)
            if s < 1e-12: return 0.0
            h = h / s
            l2_sq = np.sum(h**2)
            hc = np.convolve(h, h, mode='full')
            # Soft max
            scaled = beta * hc
            max_scaled = np.max(scaled)
            soft_max_val = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) / beta
            if soft_max_val < 1e-12: return 0.0
            return -(l2_sq / soft_max_val)

        # Use L-BFGS-B or SLSQP. SLSQP handles constraints well.
        # For speed, we might use L-BFGS-B with a penalty for constraint violation if needed, 
        # but SLSQP is more robust for equality constraints.
        # Given N=50, SLSQP with numerical gradient is okay.
        
        try:
            res = minimize(smooth_obj, current_h, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 300, 'ftol': 1e-10})
            if res.success:
                h_candidate = np.clip(res.x, 0, None)
                s = np.sum(h_candidate)
                if s > 0:
                    h_candidate /= s
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
                    current_h = h_candidate
                else:
                    current_h = best_h
        except Exception:
            pass

    # Approach B: Local search / Nelder-Mead on the exact objective
    # The exact objective is: - (sum(h^2) / max_conv(h))
    # This function is continuous but not differentiable at points where multiple 
    # convolution coefficients achieve the maximum. Nelder-Mead can handle this.
    
    def exact_obj(h):
        h = np.clip(h, 0, None)
        s = np.sum(h)
        if s < 1e-12: return 0.0
        h = h / s
        l2_sq = np.sum(h**2)
        hc = np.convolve(h, h, mode='full')
        mc = np.max(hc)
        if mc < 1e-12: return 0.0
        return -(l2_sq / mc)

    if time.time() < deadline - 5:
        try:
            # Nelder-Mead doesn't handle bounds or constraints directly.
            # We can project back to the simplex if needed, but it's tricky.
            # Alternative: Use SLSQP on the exact objective.
            res = minimize(exact_obj, best_h, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
            if res.success:
                h_candidate = np.clip(res.x, 0, None)
                s = np.sum(h_candidate)
                if s > 0:
                    h_candidate /= s
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    # Approach C: Coordinate Descent / Hill Climbing on the exact objective
    # This is often very effective for this type of problem.
    # We fix all but one variable, and optimize that variable analytically or via 1D search.
    # Since sum(h)=1, if we change h_i, we must adjust others or just do a local move.
    # Let's try a simple stochastic hill climbing with restarts.
    
    if time.time() < deadline - 2:
        h_curr = best_h.copy()
        step_size = 0.05
        no_improve_count = 0
        
        while time.time() < deadline - 1:
            # Pick a random index to perturb
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            
            # Move mass from j to i
            # h_i' = h_i + delta
            # h_j' = h_j - delta
            # Ensure non-negativity
            max_delta = h_curr[j]
            min_delta = -h_curr[i]
            delta = np.random.uniform(min_delta, max_delta) * step_size
            
            h_new = h_curr.copy()
            h_new[i] += delta
            h_new[j] -= delta
            
            if np.any(h_new < -1e-12):
                continue
                
            s_new = _score_raw(h_new)
            if s_new > best_score:
                best_score = s_new
                best_h = h_new
                h_curr = h_new
                no_improve_count = 0
            else:
                no_improve_count += 1
                # Decay step size if stuck
                if no_improve_count > 100:
                    step_size *= 0.9
                    no_improve_count = 0
                    if step_size < 1e-5:
                        # Restart from best_h with larger step
                        h_curr = best_h.copy()
                        step_size = 0.05

    return list(best_h)
