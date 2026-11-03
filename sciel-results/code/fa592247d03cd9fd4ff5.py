import numpy as np
from scipy.optimize import linprog
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Choose 50 nonnegative heights on equal intervals of [-1/4, 1/4]
    to maximize ||h||_2^2 / (||h||_1 * ||h * h||_inf).
    
    By homogeneity, we can fix ||h||_1 = 1 and minimize ||h * h||_inf.
    The autoconvolution (h * h)[k] = sum_i h_i h_{i+k} for k in [-49, 49].
    We want to minimize max_k (h*h)[k].
    
    Let t be the peak autoconvolution. Constraints:
      sum_k (h*h)[k] <= t  for all k
      sum_i h_i = 1
      h_i >= 0
    
    This is a linear program:
      Minimize t
      Subject to:
        For each lag k in [-49, 49]: sum_i h_i h_{i+k} <= t  -> NOT linear in h!
    
    Wait, the autoconvolution is quadratic in h. So this is NOT a linear program.
    
    Let me reconsider. The autoconvolution (h*h)[k] = sum_i h_i * h_{i+k}.
    This is a quadratic form. Minimizing the maximum of quadratic forms is not LP.
    
    However, we can use a different approach. Let's think about this differently.
    
    Actually, let me re-read the task. "squared L2 norm to L1 norm times supremum norm of the autoconvolution"
    
    Ratio = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    
    By homogeneity, fix ||h||_1 = 1. Then maximize ||h||_2^2 / ||h*h||_inf.
    
    ||h||_2^2 = sum h_i^2
    ||h*h||_inf = max_k |sum_i h_i h_{i+k}|
    
    Since h_i >= 0, all autoconvolution values are non-negative, so ||h*h||_inf = max_k sum_i h_i h_{i+k}.
    
    This is still a non-convex problem because we're maximizing a ratio of a convex function (sum h_i^2) to a convex function (max of linear... wait, no, the autoconvolution terms are quadratic).
    
    Let me try a different approach: use scipy.optimize.minimize with a good initial guess and see if we can find a local optimum that beats the uniform seed.
    
    Actually, let's think about what happens with the uniform distribution:
    h_i = 1/N for all i.
    ||h||_1 = 1
    ||h||_2^2 = N * (1/N)^2 = 1/N
    (h*h)[k] = sum_i (1/N)(1/N) = (N-|k|)/N^2 for |k| < N
    ||h*h||_inf = 1/N (at k=0)
    Ratio = (1/N) / (1 * 1/N) = 1
    
    So the uniform distribution gives a ratio of 1. We need to do better.
    
    Let me try a concentrated distribution. If h is concentrated at a single point:
    h_0 = 1, rest 0.
    ||h||_1 = 1
    ||h||_2^2 = 1
    (h*h)[0] = 1, all others 0
    ||h*h||_inf = 1
    Ratio = 1/1 = 1
    
    Same ratio! Interesting.
    
    Let me try two points: h_0 = a, h_k = b, a+b=1.
    ||h||_2^2 = a^2 + b^2
    (h*h)[0] = a^2 + b^2
    (h*h)[k] = ab
    (h*h)[-k] = ab
    ||h*h||_inf = max(a^2+b^2, ab)
    Ratio = (a^2+b^2) / max(a^2+b^2, ab)
    
    If a^2+b^2 >= ab (which is always true since a^2+b^2-ab = (a-b/2)^2 + 3b^2/4 >= 0), then ratio = 1.
    
    Hmm, it seems hard to beat 1. Let me try a broader distribution.
    
    Actually, I think the optimal might be related to spreading out the mass. Let me use numerical optimization.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Initialize with uniform
    h = np.ones(N) / N
    
    # Define the objective: maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # By fixing ||h||_1 = 1, maximize ||h||_2^2 / ||h*h||_inf
    
    def autoconv_peak(h):
        """Compute max_k sum_i h_i h_{i+k}"""
        # Full convolution
        conv = np.convolve(h, h, mode='full')
        # The peak is the max of the convolution
        return np.max(conv)
    
    def objective(h):
        """Negative of the ratio (for minimization)"""
        norm1 = np.sum(h)
        if norm1 < 1e-10:
            return 1e10
        norm2_sq = np.sum(h**2)
        peak = autoconv_peak(h)
        if peak < 1e-10:
            return -1e10
        return -(norm2_sq / (norm1 * peak))
    
    # Use SLSQP with constraint sum(h) = 1, h >= 0
    from scipy.optimize import minimize
    
    best_h = h.copy()
    best_val = -objective(h)  # positive ratio
    
    # Try a few different initializations
    initializations = [
        np.ones(N) / N,  # uniform
        np.array([1.0] + [0.0]*(N-1)),  # delta
        np.array([0.5, 0.5] + [0.0]*(N-2)),  # two points adjacent
        np.array([0.5, 0.0, 0.5] + [0.0]*(N-3)),  # two points separated
        np.linspace(0, 2/N, N),  # ramp
        np.sin(np.pi * np.arange(N) / N) / np.sum(np.sin(np.pi * np.arange(N) / N)),  # sine bump
    ]
    
    for h0 in initializations:
        if time.time() > deadline:
            break
        
        # Normalize
        h0 = h0 / np.sum(h0)
        
        try:
            res = minimize(
                objective,
                h0,
                method='SLSQP',
                bounds=[(0, None)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}],
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.success and res.fun < -best_val + 1e-10:
                # res.fun is negative ratio, so -res.fun is the ratio
                if -res.fun > best_val:
                    best_val = -res.fun
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Final polish with more iterations on the best solution found
    if time.time() < deadline - 5:
        try:
            res = minimize(
                objective,
                best_h,
                method='SLSQP',
                bounds=[(0, None)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}],
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if res.success and -res.fun > best_val:
                best_val = -res.fun
                best_h = res.x.copy()
        except Exception:
            pass
    
    # Ensure nonnegativity and normalize
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) > 1e-10:
        best_h = best_h / np.sum(best_h)
    
    return list(best_h)
# EVOLVE-BLOCK-END
