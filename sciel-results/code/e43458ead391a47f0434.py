"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, linprog
import time

def construct():
    """
    Construct step heights h[0..N-1] in [0,1] with sum = N/2
    to minimize the maximum overlap of h with shifted (1-h).
    
    The overlap for shift k is:
      sum_{i=0}^{N-1-k} h[i] * (1 - h[i+k])
    (assuming zero-padding outside [0, N-1])
    
    We want to minimize max_k overlap_k.
    
    Strategy:
    1. Start with a reasonable initial guess (e.g., alternating 0 and 1, or 0.5).
    2. Use a smooth surrogate for the max: t + mu * log(sum(exp((overlap_k - t)/mu)))
       or simply minimize t subject to overlap_k <= t for all k, using an epigraph form.
    3. Since the problem is small (N=100, ~100 constraints), we can use scipy.optimize.minimize
       with SLSQP or trust-constr, or formulate as a convex program if possible.
    
    Actually, let's think more carefully. The overlap for shift k is:
      O_k = sum_{i=0}^{N-1-k} h_i (1 - h_{i+k})
          = sum_{i=0}^{N-1-k} h_i - sum_{i=0}^{N-1-k} h_i h_{i+k}
    
    This is quadratic in h. Minimizing the maximum of these quadratics is non-convex in general.
    
    However, note that if we set h_i = 0.5 for all i, then:
      O_k = sum_{i=0}^{N-1-k} 0.5 * 0.5 = (N-k) * 0.25
      max_k O_k = (N-1) * 0.25 ≈ 24.75 for N=100.
    
    Can we do better? If we alternate 0 and 1:
      h = [1, 0, 1, 0, ...]
      For shift k=1: O_1 = sum_{i=0}^{N-2} h_i (1-h_{i+1})
        i=0: h_0=1, h_1=0 -> 1*1 = 1
        i=1: h_1=0, h_2=1 -> 0*0 = 0
        i=2: h_2=1, h_3=0 -> 1*1 = 1
        ...
        So O_1 = N/2 (approximately) = 50. That's worse.
    
    What about a smoother pattern? Let's try to use optimization.
    
    Given the 120s time limit, we need an efficient approach. Let's try:
    1. Use a parameterized family of solutions.
    2. Or use scipy.optimize.minimize with a good initial guess.
    
    Let's try a different approach: note that the problem is symmetric. 
    Let's use a smooth optimization with the epigraph formulation:
      minimize t
      subject to O_k <= t for all k
                 0 <= h_i <= 1
                 sum h_i = N/2
    
    This is a non-convex QP (quadratic constraints). We can use SLSQP.
    
    To speed things up, we can:
    - Reduce the number of shifts to consider (only shifts up to some max, since for large k, the overlap is small).
    - Use a good initial guess.
    
    Let's implement this.
    """
    deadline = time.time() + 110  # 10-second safety margin
    
    # Initial guess: 0.5 everywhere
    h0 = np.full(N, 0.5)
    
    # Objective: minimize t + penalty for constraint violations
    # We'll use the epigraph form with t as an additional variable
    # x = [h_0, ..., h_{N-1}, t]
    
    def objective(x):
        h = x[:N]
        t = x[N]
        # Compute max overlap approximately
        max_ov = 0.0
        for k in range(1, N):
            if time.time() > deadline:
                break
            # overlap for shift k
            ov = np.sum(h[:N-k] * (1 - h[k:]))
            if ov > max_ov:
                max_ov = ov
        return t + 0.1 * max(0, max_ov - t) ** 2  # soft penalty
    
    # Constraints
    def constraint_sum(x):
        return np.sum(x[:N]) - N / 2.0
    
    # Bounds: h_i in [0,1], t >= 0
    bounds = [(0, 1)] * N + [(0, None)]
    
    # Try SLSQP
    try:
        res = minimize(
            objective,
            h0,
            method='SLSQP',
            bounds=bounds,
            constraints=[{'type': 'eq', 'fun': constraint_sum}],
            options={'maxiter': 200, 'ftol': 1e-6}
        )
        if res.success and time.time() < deadline:
            h = res.x[:N]
            # Clamp to [0,1]
            h = np.clip(h, 0, 1)
            # Adjust to ensure sum is exactly N/2
            # Simple projection: scale and clip, then adjust
            diff = N / 2.0 - np.sum(h)
            if abs(diff) > 1e-10:
                # Distribute the difference evenly, respecting bounds
                h += diff / N
                h = np.clip(h, 0, 1)
                # Second correction
                diff2 = N / 2.0 - np.sum(h)
                if abs(diff2) > 1e-10:
                    # Adjust the element closest to 0.5
                    idx = np.argmin(np.abs(h - 0.5))
                    h[idx] += diff2
                    h = np.clip(h, 0, 1)
                    # Final tiny correction
                    diff3 = N / 2.0 - np.sum(h)
                    if abs(diff3) > 1e-12:
                        h[np.argmin(np.abs(h - 0.5))] += diff3
            return h
    except Exception:
        pass
    
    # Fallback: return the initial guess
    return h0

# EVOLVE-BLOCK-END
