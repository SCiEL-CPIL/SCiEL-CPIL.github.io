import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START
def _compute_overlap_values(h):
    """
    Compute the overlap of h with shifted 1-h for all shifts k in 0..N-1.
    Uses FFT for efficiency.
    h is a vector of length N.
    Overlap O(k) = sum_i h[i] * (1 - h[(i-k) % N])
    This is equivalent to 1/2 - sum_i h[i] h[(i-k) % N] if sum(h) = N/2.
    Wait, sum_i h[i](1-h[i-k]) = sum_i h[i] - sum_i h[i]h[i-k].
    sum_i h[i] is N/2.
    So O(k) = N/2 - R(k), where R(k) is the cyclic autocorrelation of h.
    We want to minimize max O(k), which is equivalent to maximizing min R(k).
    """
    # Autocorrelation via FFT
    # R(k) = ifft(fft(h) * conj(fft(h)))
    # Note: scipy ifft returns complex, take real part.
    # Also need to handle the scaling.
    # numpy fft: ifft(fft(x) * conj(fft(x))) gives circular correlation.
    # The result is scaled by N? No, numpy ifft includes 1/N factor.
    # Let's verify: x = [1, 0]. fft = [1, 1]. conj = [1, 1]. prod = [1, 1]. ifft = [1, 0].
    # Correlation of [1, 0] with itself: [1, 0]. Correct.
    
    H = np.fft.fft(h)
    power_spectrum = H * np.conj(H)
    R = np.fft.ifft(power_spectrum).real
    
    # O(k) = N/2 - R(k)
    # We want to minimize max O(k)
    # This is equivalent to minimizing max (N/2 - R(k))
    # Which is equivalent to maximizing min R(k)
    # But the objective for minimizer is max O(k).
    
    return N/2 - R

def _objective_and_grad(h):
    """
    Objective: max overlap.
    We use a smooth approximation (log-sum-exp) to the max to allow gradients.
    """
    overlaps = _compute_overlap_values(h)
    # Soft max: log(sum(exp(beta * x))) / beta
    beta = 10.0
    # Avoid overflow
    max_val = np.max(overlaps)
    soft_max = max_val + np.log(np.sum(np.exp(beta * (overlaps - max_val)))) / beta
    
    # Gradient calculation is tricky with FFT.
    # We can use finite differences or analytic gradient.
    # For simplicity and robustness within time limit, we might use finite differences 
    # or just provide the value and let L-BFGS-B use finite diff.
    # However, analytic gradient is faster.
    # dO/dh_i = - dR/dh_i.
    # R = ifft(|FFT(h)|^2).
    # This is a quadratic form.
    # R = h @ C h, where C is the correlation matrix?
    # Actually, R(k) = sum_i h[i] h[i-k].
    # O(k) = N/2 - R(k).
    # grad O(k) wrt h is -2 * shift(h, k) ?
    # Let's just use finite difference gradient to be safe, or simple analytic.
    # Given N=100, finite diff is fast enough.
    
    return soft_max, None # None triggers finite difference in L-BFGS-B if not provided

def construct():
    start_time = time.time()
    deadline = 120.0
    margin = 10.0
    
    # Initial guess: Uniform
    h0 = np.full(N, 0.5)
    
    # Multi-start strategy
    best_h = h0
    best_score = _compute_overlap_values(h0).max()
    
    # Define constraints
    # Sum(h) = N/2
    # 0 <= h <= 1
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - N/2, 'jac': lambda h: np.ones(N)}
    ]
    bounds = [(0, 1)] * N
    
    num_starts = 20
    rng = np.random.default_rng(42)
    
    for i in range(num_starts):
        if time.time() > deadline - margin:
            break
            
        if i == 0:
            h_init = h0
        else:
            # Add random Fourier modes
            h_init = h0.copy()
            # Add noise
            noise = rng.normal(0, 0.1, N)
            h_init += noise
            # Project onto constraint sum = N/2
            h_init -= np.sum(h_init) - N/2
            # Clip to bounds
            h_init = np.clip(h_init, 0, 1)
            # Re-project? Clipping breaks sum. 
            # Simple re-normalization
            if np.sum(h_init) > 0:
                h_init *= (N/2) / np.sum(h_init)
            # Clip again
            h_init = np.clip(h_init, 0, 1)
            # Final fix for sum if needed (simple iterative water filling)
            # For now, just use the clipped version as start, L-BFGS-B will handle bounds.
            # But L-BFGS-B requires starting point to satisfy linear constraints?
            # No, it handles bounds, but linear equality constraints should be satisfied initially or close.
            
        # Run L-BFGS-B
        # We minimize max overlap.
        # Using soft-max objective.
        
        # To avoid finite difference cost, we can compute gradient analytically.
        # But let's try without first.
        
        res = minimize(
            _objective_and_grad, 
            h_init, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        h_res = res.x
        # Ensure constraints are met
        # Sum correction
        h_res -= np.sum(h_res) - N/2
        h_res = np.clip(h_res, 0, 1)
        
        score = _compute_overlap_values(h_res).max()
        
        if score < best_score:
            best_score = score
            best_h = h_res.copy()
            
    return best_h
# EVOLVE-BLOCK-END
