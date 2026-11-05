"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _eval_sup_norm(coeffs):
    """
    Compute the sup norm of P(z) = sum c_k z^k over |z|=1.
    Uses dense sampling + golden-section refinement of top peaks.
    Returns (max_abs_value, theta_at_max).
    """
    n = len(coeffs)
    c = np.array(coeffs, dtype=float)
    
    # Dense sampling
    N = 8192
    theta = 2.0 * np.pi * np.arange(N) / N
    z = np.exp(1j * theta)
    
    # Evaluate polynomial at all points
    # P(z_j) = sum_k c_k * z_j^k
    # Use matrix: V[k, j] = z_j^k
    # For efficiency with small n, compute powers iteratively
    values = np.zeros(N, dtype=complex)
    z_pow = np.ones(N, dtype=complex)
    for k in range(n):
        values += c[k] * z_pow
        z_pow *= z
    
    abs_vals = np.abs(values)
    
    # Find top peaks
    top_k = 10
    top_indices = np.argpartition(abs_vals, -top_k)[-top_k:]
    
    best_val = 0.0
    best_theta = 0.0
    
    for idx in top_indices:
        # Golden section refinement around theta[idx]
        t0 = theta[idx]
        # Determine local neighborhood
        if idx == 0:
            t_left = t0 - 2*np.pi/N
            t_right = t0 + 2*np.pi/N
        elif idx == N-1:
            t_left = t0 - 2*np.pi/N
            t_right = t0 + 2*np.pi/N
        else:
            t_left = theta[idx-1]
            t_right = theta[idx+1]
        
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2  # golden ratio ~0.618
        a = t_left
        b = t_right
        c_pt = b - gr * (b - a)
        d_pt = a + gr * (b - a)
        
        def f(t):
            z_pt = np.exp(1j * t)
            val = 0.0 + 0.0j
            z_pow_pt = 1.0 + 0.0j
            for k in range(n):
                val += c[k] * z_pow_pt
                z_pow_pt *= z_pt
            return np.abs(val)
        
        fc = f(c_pt)
        fd = f(d_pt)
        
        for _ in range(50):  # ~50 iterations gives accuracy ~ (b-a)*gr^50
            if fc < fd:
                b = d_pt
                d_pt = c_pt
                fd = fc
                c_pt = b - gr * (b - a)
                fc = f(c_pt)
            else:
                a = c_pt
                c_pt = d_pt
                fc = fd
                d_pt = a + gr * (b - a)
                fd = f(d_pt)
        
        t_star = (a + b) / 2
        f_star = f(t_star)
        
        if f_star > best_val:
            best_val = f_star
            best_theta = t_star
    
    return best_val, best_theta

def _score(coeffs):
    """Compute the score: max |P(z)| / sqrt(24)."""
    max_val, _ = _eval_sup_norm(coeffs)
    return max_val / math.sqrt(24)

def construct():
    """
    Find 24 coefficients in {+1, -1} minimizing the sup norm.
    Strategy: 
    1. Start with Golay-Rudin-Shapiro as initial guess.
    2. Use local search (hill climbing) with random restarts.
    3. Use a continuous relaxation with gradient-based optimization as a guide.
    """
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 10s safety margin
    
    # Initial guess: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    initial = p[:24]
    
    best_coeffs = initial[:]
    best_score = _score(best_coeffs)
    
    # Local search with random restarts
    rng = np.random.RandomState(42)
    
    def _local_search(coeffs, max_iter=5000):
        """Hill climbing: flip one bit at a time if it improves the score."""
        current = coeffs[:]
        current_score = _score(current)
        improved = True
        iters = 0
        
        while improved and iters < max_iter:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            
            # Try flipping each bit
            best_flip = -1
            best_new_score = current_score
            
            for k in range(24):
                trial = current[:]
                trial[k] = -trial[k]
                s = _score(trial)
                if s < best_new_score:
                    best_new_score = s
                    best_flip = k
            
            if best_flip >= 0:
                current[best_flip] = -current[best_flip]
                current_score = best_new_score
                improved = True
        
        return current, current_score
    
    # Phase 1: Local search from initial guess
    best_coeffs, best_score = _local_search(initial)
    
    # Phase 2: Random restarts
    num_restarts = 0
    while time.time() < deadline - 10:  # Leave 10s margin
        # Generate random starting point
        random_coeffs = [1 if rng.random() < 0.5 else -1 for _ in range(24)]
        
        # Local search from random start
        candidate, score = _local_search(random_coeffs, max_iter=2000)
        
        if score < best_score:
            best_coeffs = candidate
            best_score = score
            num_restarts += 1
            
            # If we found a very good solution, do more intensive local search
            if best_score < 1.26:
                # Try more aggressive local search with longer max_iter
                candidate, score = _local_search(best_coeffs, max_iter=10000)
                if score < best_score:
                    best_coeffs = candidate
                    best_score = score
        else:
            num_restarts += 1
    
    # Phase 3: Continuous relaxation to guide discrete search
    if time.time() < deadline - 15:
        # Use scipy minimize to find a continuous solution, then project
        def objective(x):
            """Smooth surrogate for sup norm using log-sum-exp."""
            c = np.tanh(x)  # Map real values to [-1, 1]
            # Evaluate at many points
            N = 2048
            theta = 2.0 * np.pi * np.arange(N) / N
            z = np.exp(1j * theta)
            
            values = np.zeros(N, dtype=complex)
            z_pow = np.ones(N, dtype=complex)
            for k in range(24):
                values += c[k] * z_pow
                z_pow *= z
            
            abs_vals = np.abs(values)
            # Soft max using log-sum-exp
            M = np.max(abs_vals)
            if M == 0:
                return 0.0
            log_sum = M + np.log(np.sum(np.exp((abs_vals - M) / 1.0)))
            return log_sum / math.sqrt(24)
        
        # Start from current best, mapped to real line via inverse tanh
        x0 = np.zeros(24)
        for k in range(24):
            x0[k] = math.atanh(0.99 * best_coeffs[k])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.success or result.fun < best_score:
                # Project to binary
                c_cont = np.tanh(result.x)
                proj = [1 if c_cont[k] > 0 else -1 for k in range(24)]
                
                # Local search from projection
                candidate, score = _local_search(proj, max_iter=5000)
                if score < best_score:
                    best_coeffs = candidate
                    best_score = score
        except Exception:
            pass
    
    # Final check: make sure we return valid coefficients
    assert len(best_coeffs) == 24
    assert all(c in (1, -1) for c in best_coeffs)
    
    return best_coeffs
# EVOLVE-BLOCK-END
