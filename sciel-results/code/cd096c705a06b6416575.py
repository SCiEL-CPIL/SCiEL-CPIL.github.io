"""Rewritten program: spectral initialization + continuous relaxation + discrete refinement."""
import numpy as np
from scipy.optimize import minimize
import time

N = 400

def _autocorr_sidelobes(x):
    """Compute autocorrelation sidelobes (excluding lag 0) using FFT."""
    n = len(x)
    # Zero-pad to 2n for linear autocorrelation
    size = 2 * n
    X = np.fft.rfft(x, size)
    acf = np.fft.irfft(X * np.conj(X), size)
    # acf has length size; lags 0..n-1 are in acf[0:n]
    # Sidelobes are lags 1..n-1
    return acf[1:n]

def _soft_max_sidelobe(x, beta=10.0):
    """Soft approximation of max absolute autocorrelation sidelobe."""
    sidelobes = _autocorr_sidelobes(x)
    abs_sidelobes = np.abs(sidelobes)
    # Log-sum-exp for max
    m = np.max(abs_sidelobes)
    if m == 0:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (abs_sidelobes - m)))) / beta

def _objective_and_grad(x, beta=10.0):
    """Objective: soft max of absolute autocorrelation sidelobes."""
    sidelobes = _autocorr_sidelobes(x)
    abs_sidelobes = np.abs(sidelobes)
    m = np.max(abs_sidelobes)
    if m == 0:
        return 0.0, np.zeros_like(x)
    
    # Soft max value
    exp_vals = np.exp(beta * (abs_sidelobes - m))
    sum_exp = np.sum(exp_vals)
    val = m + np.log(sum_exp) / beta
    
    # Gradient via chain rule
    # d/dx of sum of |acf[k]| approximated by soft max
    # acf[k] = sum_i x[i]*x[i+k]
    # For soft max: grad = (1/beta) * sum_j (exp(beta*(|acf[j]|-m))/sum_exp) * sign(acf[j]) * d(acf[j])/dx
    weights = exp_vals / sum_exp / beta
    # d|acf[j]|/dx = sign(acf[j]) * d(acf[j])/dx
    # acf[j] = sum_i x[i]*x[i+j], so d(acf[j])/dx[i] = x[i+j] + x[i-j] (for valid indices)
    grad = np.zeros_like(x)
    for j in range(1, len(x)):
        if abs_sidelobes[j] == 0:
            continue
        s = np.sign(sidelobes[j])
        w = weights[j] * s
        # Contribution to gradient from acf[j]
        # d(acf[j])/dx[i] = x[i+j] if i+j < n, plus x[i-j] if i-j >= 0
        # This is the convolution structure
        # acf[j] = sum_{i=0}^{n-1-j} x[i] * x[i+j]
        # d/dx[i] = x[i+j] for i < n-j, and d/dx[i] = x[i-j] for i >= j
        # So grad[i] += w * (x[i+j] if i+j<n else 0) + w * (x[i-j] if i-j>=0 else 0)
        # Vectorized:
        upper = x[j:]  # x[i+j] for i=0..n-1-j
        lower = x[:n-j]  # x[i-j] for i=j..n-1, i.e., x[0..n-1-j]
        grad[:n-j] += w * upper
        grad[j:] += w * lower
    
    return val, grad

def _project_to_sign(x):
    """Project to ±1."""
    return np.sign(x)

def _local_refine(bits, max_iterations=50000, time_limit=None):
    """Local bit-flip refinement to minimize max autocorrelation sidelobe."""
    n = len(bits)
    best_bits = bits.copy()
    best_score = None
    
    def _score(b):
        sidelobes = _autocorr_sidelobes(b.astype(float))
        return np.max(np.abs(sidelobes))
    
    best_score = _score(best_bits)
    
    start_time = time.time()
    iters = 0
    while iters < max_iterations:
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        improved = False
        # Random order for diversity
        indices = np.random.permutation(n)
        for i in indices:
            if time_limit is not None and (time.time() - start_time) > time_limit:
                break
            new_bits = best_bits.copy()
            new_bits[i] = -new_bits[i]
            new_score = _score(new_bits)
            if new_score < best_score:
                best_bits = new_bits
                best_score = new_score
                improved = True
        iters += 1
        if not improved:
            break
    
    return best_bits, best_score

def construct():
    global N
    n = N
    start_time = time.time()
    time_budget = 110.0  # Leave 10s margin
    
    # Phase 1: Spectral initialization
    # Start with a random ±1 sequence, or use a structured initialization
    np.random.seed(42)
    
    # Try multiple random initializations and pick the best
    best_bits = None
    best_score = np.inf
    
    # Also try a structured initialization based on low-peak spectrum
    # Initialize from inverse FFT of flat magnitude spectrum
    # This gives a sequence with flat autocorrelation in expectation
    
    n_trials = 10
    for trial in range(n_trials):
        if (time.time() - start_time) > time_budget * 0.3:
            break
        
        # Random ±1
        bits = np.random.choice([-1.0, 1.0], size=n)
        score = np.max(np.abs(_autocorr_sidelobes(bits)))
        if score < best_score:
            best_score = score
            best_bits = bits.copy()
    
    # Phase 2: Continuous relaxation with L-BFGS-B
    # Use the best discrete solution as starting point
    x0 = best_bits.astype(float)
    
    beta = 5.0  # Softness parameter
    
    def obj(x):
        val, _ = _objective_and_grad(x, beta=beta)
        return val
    
    def grad(x):
        _, g = _objective_and_grad(x, beta=beta)
        return g
    
    # Run L-BFGS-B
    try:
        result = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000, 'disp': False}
        )
        x_continuous = result.x
    except Exception:
        x_continuous = x0
    
    # Project to ±1
    bits_proj = np.sign(x_continuous)
    score_proj = np.max(np.abs(_autocorr_sidelobes(bits_proj)))
    
    if score_proj < best_score:
        best_bits = bits_proj
        best_score = score_proj
    
    # Phase 3: Local refinement
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 5:
        best_bits, best_score = _local_refine(
            best_bits.astype(float), 
            max_iterations=100000,
            time_limit=max(0, remaining_time - 2)
        )
    
    # Return as sequence of signed step heights
    return best_bits

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
