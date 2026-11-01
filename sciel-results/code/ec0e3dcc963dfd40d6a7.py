"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_c3_bound(h):
    """Compute the actual C3 upper bound for a given height vector."""
    h = np.asarray(h)
    n_fft = 2**int(np.ceil(np.log2(2*N-1)))
    H = np.fft.rfft(h, n=n_fft)
    C = np.fft.irfft(H * H, n=n_fft)
    conv = C[:2*N-1]
    
    # Exact metric: max |conv| / (sum |h|)^2
    max_abs = np.max(np.abs(conv))
    sum_abs = np.sum(np.abs(h))
    return max_abs / (sum_abs**2)


def _local_refine(h, max_time=10.0):
    """Local refinement via discrete sign flips."""
    h = np.array(h, dtype=float)
    start = time.time()
    best_score = _compute_c3_bound(h)
    
    while time.time() - start < max_time:
        improved = False
        # Try flipping each sign
        for i in range(N):
            h[i] = -h[i]
            score = _compute_c3_bound(h)
            if score < best_score - 1e-10:
                best_score = score
                improved = True
            else:
                h[i] = -h[i]  # revert
        
        if not improved:
            break
    
    return h, best_score


def _c3_upper_bound(h):
    """
    Compute the upper bound on C3 for a signed step function with heights h.
    Metric: max_k |conv(h, h)[k]| / (sum(|h|))^2
    """
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin for safety
    
    def objective(x):
        h = np.tanh(x)
        # Use FFT for convolution
        n_fft = 2**int(np.ceil(np.log2(2*N-1)))
        H = np.fft.rfft(h, n=n_fft)
        C = np.fft.irfft(H * H, n=n_fft)
        conv = C[:2*N-1]
        
        # Tighter smooth max
        eps = 1e-12
        abs_conv = np.sqrt(conv**2 + eps)
        T = 1.0  # Tighter than 5.0
        max_abs = np.max(abs_conv)
        # Numerically stable log-sum-exp
        shifted = (abs_conv - max_abs) / T
        log_sum_exp = np.log(np.sum(np.exp(shifted))) + max_abs / T
        smooth_max = T * log_sum_exp
        
        sum_abs_h = np.sum(np.sqrt(h**2 + eps))
        value = smooth_max / (sum_abs_h**2)
        return value

    # Better initial conditions: random signs to break symmetry
    rng = np.random.default_rng(42)
    x0 = rng.standard_normal(N) * 2.0
    
    # Run optimization
    h_opt = None
    best_score = float('inf')
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False})
        h_opt = np.tanh(res.x)
        best_score = _compute_c3_bound(h_opt)
    except Exception:
        h_opt = np.ones(N)
        best_score = _compute_c3_bound(h_opt)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return h_opt
    
    # Local refinement with remaining time
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 5.0:
        h_refined, refined_score = _local_refine(h_opt, max_time=min(remaining_time - 2.0, 30.0))
        if refined_score < best_score:
            h_opt = h_refined
            best_score = refined_score
    
    return h_opt


def construct():
    return _c3_upper_bound(np.full(N, 5.0))
# EVOLVE-BLOCK-END
