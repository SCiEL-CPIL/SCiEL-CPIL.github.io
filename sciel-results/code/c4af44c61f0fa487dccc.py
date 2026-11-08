"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_cost(f):
    """
    Compute the cost J = max_k |AC(k)| / (sum(f))^2.
    Returns infinity if sum(f) is 0.
    """
    s = np.sum(f)
    if s == 0:
        return float('inf')
    
    # Compute autocorrelation
    # Using FFT for efficiency, though N=400 is small enough for direct
    # AC(k) = sum_{i} f[i] * f[i+k]
    # Zero-pad to avoid circular convolution effects if we used FFT directly on N
    # But standard autocorrelation is linear.
    # scipy.signal.correlate(f, f, mode='full') gives linear autocorrelation.
    # However, we want to minimize the MAX absolute value over all lags.
    # The lags range from -(N-1) to N-1.
    
    # Direct calculation is O(N^2) = 160k ops, very fast.
    # Let's use np.correlate
    ac = np.correlate(f, f, mode='full')
    
    max_abs_ac = np.max(np.abs(ac))
    
    return max_abs_ac / (s * s)

def construct():
    """
    Construct a signed step function that minimizes the maximum absolute
    autoconvolution divided by the squared integral.
    
    Strategy:
    Use random initialization and hill climbing (local search) to minimize
    the cost function J = max_k |AC(k)| / (sum(f))^2.
    We restrict f to +/- 1 values.
    """
    np.random.seed(42)
    
    # Start with random +/- 1
    f = np.random.choice([-1.0, 1.0], size=N)
    
    # Ensure sum is not zero initially, if it is, flip a random bit
    while np.sum(f) == 0:
        idx = np.random.randint(N)
        f[idx] *= -1
    
    current_cost = _compute_cost(f)
    
    # Hill climbing
    max_iterations = 50
    no_improvement_count = 0
    start_time = time.time()
    
    for _ in range(max_iterations):
        if time.time() - start_time > 100: # Leave time for other evaluations
            break
            
        improved = False
        # Try flipping each bit
        # Optimization: We can update AC incrementally, but for N=400,
        # recalculating AC is cheap (~0.1ms). 400 flips * 0.1ms = 40ms per iteration.
        # 50 iterations = 2s. This is safe.
        
        for i in range(N):
            # Flip bit i
            f[i] *= -1.0
            new_cost = _compute_cost(f)
            
            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
            else:
                # Revert
                f[i] *= -1.0
                
        if not improved:
            break
            
    return f
# EVOLVE-BLOCK-END
