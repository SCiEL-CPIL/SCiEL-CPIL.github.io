# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_prob(r1, r2, p, n=128):
    """
    Compute P[X1+X2+X3 < 2*X4] for a mixture of two geometric distributions.
    r1, r2: decay parameters (0 < r < 1)
    p: weight for component 1
    n: number of support points
    """
    support = np.arange(n, dtype=np.float64)
    
    # Geometric PMF: (1-r) * r^i
    # Ensure numerical stability
    w1 = (1 - r1) * (r1 ** support)
    w2 = (1 - r2) * (r2 ** support)
    
    weights = p * w1 + (1 - p) * w2
    
    # Normalize just in case, though geometric sums to 1
    total_w = np.sum(weights)
    if total_w <= 1e-9:
        return 0.0
    weights = weights / total_w
    
    # Compute CDF of Z = X1 + X2
    # Z takes values in [0, 2*(n-1)]
    # We need P[Z < 2*X4 - X3]
    # Let S = X1 + X2. We need sum over k,l w_k w_l P[S < 2*s_l - s_k]
    
    # Efficiently compute CDF of S
    # S is sum of two i.i.d. discrete variables on [0, n-1]
    # Max sum is 2n-2.
    # We can compute the probability mass function of S via convolution or direct counting.
    # Direct counting is O(n^2).
    
    # Create a histogram of sums
    # Since support is integers 0..n-1, sums are integers 0..2n-2
    sum_probs = np.zeros(2 * n - 1, dtype=np.float64)
    
    # Loop is O(n^2). For n=128, 16k ops. Fast.
    # But we can vectorize?
    # i, j in 0..n-1
    # sum = i + j
    # prob = w_i * w_j
    # We can use np.add.outer and then bincount?
    # Or just a loop is fine for n=128 inside an optimizer that might run 100-1000 iterations.
    # 1000 * 16k = 16M ops. Very fast in Python/NumPy.
    
    # Let's use a vectorized approach for speed
    # i = np.arange(n)
    # j = np.arange(n)
    # sums = i[:, None] + j[None, :]
    # probs = weights[:, None] * weights[None, :]
    # This creates 128x128 arrays. 
    # Then bincount(sums, weights=probs)
    
    i = np.arange(n)
    j = np.arange(n)
    sums = i[:, None] + j[None, :]
    probs = weights[:, None] * weights[None, :]
    
    # sums is int array, probs is float
    # bincount requires non-negative ints
    sum_probs = np.bincount(sums.flatten(), weights=probs.flatten(), minlength=2*n-1)
    
    # CDF of S: CDF[t] = P[S <= t]
    # We need P[S < x] = P[S <= x-1] for integer x.
    # If x is not integer, P[S < x] = P[S <= floor(x-epsilon)]
    # Here x = 2*s_l - s_k. Since s_l, s_k are integers, x is integer.
    # So P[S < x] = sum_probs[0:x].sum() = CDF[x-1]
    
    cdf = np.cumsum(sum_probs)
    
    # Now sum over k, l
    # term = w_k * w_l * P[S < 2*s_l - s_k]
    # Let x = 2*s_l - s_k
    # If x <= 0, P[S < x] = 0
    # If x >= 2n-1, P[S < x] = 1 (since max sum is 2n-2)
    
    # We can vectorize this too.
    # k, l in 0..n-1
    # x = 2*l - k
    # prob_val = cdf[x-1] if x > 0 else 0
    
    # Create matrices
    k = np.arange(n)
    l = np.arange(n)
    x = 2 * l[:, None] - k[None, :] # Shape (n, n)
    
    # Clip x to valid indices for cdf lookup
    # We need index = x - 1
    # If x <= 0, index < 0, prob 0
    # If x > 2n-1, index >= 2n-1, prob 1 (sum_probs sums to 1)
    
    indices = x - 1
    
    # Initialize result matrix
    prob_s_less_x = np.zeros((n, n), dtype=np.float64)
    
    # Mask for valid indices (0 <= index <= 2n-2)
    mask = (indices >= 0) & (indices < 2*n-1)
    prob_s_less_x[mask] = cdf[indices[mask]]
    
    # Mask for x > 2n-1 (i.e. index >= 2n-1)
    mask_high = indices >= 2*n-1
    prob_s_less_x[mask_high] = 1.0
    
    # Final sum
    # Sum over k, l of w_k * w_l * prob_s_less_x[l, k]
    # Note: x matrix is l (rows), k (cols). So prob_s_less_x is indexed by [l, k].
    # Weights are w_k (cols) and w_l (rows).
    # So we need sum_{l,k} w_l * w_k * prob_s_less_x[l, k]
    
    # Outer product of weights
    w_outer = weights[None, :] * weights[:, None] # Shape (n, n), [l, k]
    
    p_val = np.sum(w_outer * prob_s_less_x)
    
    return p_val

def _objective(params):
    r1, r2, p = params
    # Clip to valid ranges to avoid numerical issues
    r1 = np.clip(r1, 0.01, 0.999)
    r2 = np.clip(r2, 0.01, 0.999)
    p = np.clip(p, 0.01, 0.99)
    
    prob = _compute_prob(r1, r2, p)
    return -prob # Minimize negative probability

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    n = 128
    support = np.arange(n, dtype=np.float64)
    
    # Initial guess
    x0 = np.array([0.85, 0.98, 0.7])
    
    # Bounds
    bounds = [(0.01, 0.999), (0.01, 0.999), (0.01, 0.99)]
    
    # Optimize
    # Use Nelder-Mead for robustness without gradients
    res = minimize(_objective, x0, method='Nelder-Mead', bounds=bounds, options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    r1, r2, p = res.x
    r1 = np.clip(r1, 0.01, 0.999)
    r2 = np.clip(r2, 0.01, 0.999)
    p = np.clip(p, 0.01, 0.99)
    
    w1 = (1 - r1) * (r1 ** support)
    w2 = (1 - r2) * (r2 ** support)
    weights = p * w1 + (1 - p) * w2
    
    # Normalize
    weights = weights / np.sum(weights)
    
    return support, weights
# EVOLVE-BLOCK-END
