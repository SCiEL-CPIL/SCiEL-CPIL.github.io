import numpy as np
import time
import sys
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.linalg import eigh

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * Sup(AutoConv))"""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max <= 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _build_toeplitz_conv_matrix(N):
    """
    Build the matrix T such that h^T T h = sum_k conv(h,h)[k] * w_k for some weights.
    Actually, for the peak autoconvolution, we need max_k h^T T_k h.
    For nonnegative h, the peak is often at k=0 or k=N-1.
    
    Alternative: use the fact that for the discrete autoconvolution,
    max_conv(h) = max_k sum_i h[i]*h[i+k]
    
    We can formulate: minimize max_k h^T T_k h / (h^T h)
    This is a min-max problem. Use bisection on lambda:
    Check if exists h>=0, sum=1 with h^T h >= lambda * max_k h^T T_k h
    """
    # T_k[i,j] = 1 if j-i = k (for k>=0) or i-j = -k (for k<0)
    # For k=0: T_0 = I, h^T T_0 h = sum h_i^2
    # For k>0: T_k has 1s on the k-th superdiagonal
    # For k<0: T_k = T_{-k}^T
    # max_conv = max_{k=-N+1}^{N-1} h^T T_k h
    # But T_k for k and -k give same value for real h (by symmetry of convolution)
    # So max_conv = max_{k=0}^{N-1} h^T T_k h where T_k is the k-th shift
    
    # For the eigenvalue approach, consider the average or the max.
    # Let's build T_0 = I (for lag 0)
    # The key insight: for nonnegative h, h^T T_0 h = sum(h^2) is always a candidate
    # for the max. Other lags can be larger only if h has structure.
    
    # For the bisection: we want max h^T I h / max_k h^T T_k h
    # Since h^T I h <= max_k h^T T_k h, the ratio is <= 1.
    # Achieved when h^T I h = max_k h^T T_k h, i.e., lag 0 is the unique max.
    
    return None  # Placeholder

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Theoretical max: 1/DELTA = 2N = 100, achieved by delta functions.
    # Structural approach: use generalized eigenvalue / bisection on spectral radius.
    
    # Start with delta function (theoretical optimum)
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_score = _score_raw(best_h)
    
    # Try all delta positions
    for pos in range(N):
        h_delta = np.zeros(N)
        h_delta[pos] = 1.0
        s = _score_raw(h_delta)
        if s > best_score:
            best_score = s
            best_h = h_delta.copy()
    
    # Bisection approach on the ratio:
    # We want to find h that maximizes R(h) = sum(h^2) / max_conv(h)
    # subject to h >= 0, sum(h) = 1
    # R(h) <= 1 always, and R(h) = 1 for delta functions.
    # 
    # Alternative structural approach: use the power method / eigenvalue iteration
    # on the autoconvolution operator to find h that concentrates mass where
    # the autoconvolution peak is minimized relative to L2^2.
    
    # Let's try a different family: functions with minimal autoconvolution spread
    # This means we want h such that conv(h,h) is as "peaky" as possible at lag 0.
    
    # Try near-delta with slight spread (might help numerically)
    for spread in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]:
        h_spread = np.zeros(N)
        center = N // 2
        for i in range(N):
            d = abs(i - center)
            h_spread[i] = np.exp(-d**2 / (2 * max(spread, 0.1)**2))
        if np.sum(h_spread) > 0:
            h_spread /= np.sum(h_spread)
            s = _score_raw(h_spread)
            if s > best_score:
                best_score = s
                best_h = h_spread.copy()
    
    # Use SLSQP from delta to see if it can find anything better
    def neg_obj(h):
        h = np.maximum(0, h)
        l1 = np.sum(h)
        if l1 <= 0:
            return 0.0
        h_norm = h / l1
        l2_sq = np.sum(h_norm**2)
        h_conv = np.convolve(h_norm, h_norm, mode='full')
        max_conv = np.max(h_conv)
        if max_conv <= 0:
            return 0.0
        return -(l2_sq / max_conv)
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
        {'type': 'ineq', 'fun': lambda h: h}
    ]
    bounds = [(0.0, 1.0)] * N
    
    # Optimize from delta
    try:
        res = minimize(neg_obj, best_h.copy(), method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': 300, 'ftol': 1e-14})
        if res.success:
            h_candidate = np.maximum(0, res.x)
            if np.sum(h_candidate) > 0:
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
    except Exception:
        pass
    
    # Random sparse restarts
    np.random.seed(123)
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        h_rand = np.zeros(N)
        k = np.random.randint(1, 6)
        indices = np.random.choice(N, k, replace=False)
        h_rand[indices] = np.random.rand(k)
        h_rand /= np.sum(h_rand)
        
        try:
            res = minimize(neg_obj, h_rand, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 150, 'ftol': 1e-12})
            if res.success:
                h_candidate = np.maximum(0, res.x)
                if np.sum(h_candidate) > 0:
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
        except Exception:
            pass
    
    return list(best_h)

# EVOLVE-BLOCK-END
