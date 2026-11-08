"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _c3_upper_bound(x):
    """
    Compute the upper bound on C3 for a signed step function.
    The task states: 'score is the upper bound on C3' and 'lower is better'.
    The autoconvolution of x with x is (x * x)[k] = sum_i x[i] x[i+k].
    The L-infinity norm of the autoconvolution is a common upper bound for such constants.
    Specifically, if f is a step function with heights x_i on intervals of width h,
    the autoconvolution integral is related to the discrete convolution.
    Let's use the L_infinity norm of the discrete autoconvolution as the metric.
    """
    # Efficient convolution using FFT
    # We need the linear convolution, so pad to length 2N-1
    # Using np.convolve with mode='full' is O(N^2) for small N, but FFT is faster for N=400.
    # Actually, for N=400, O(N^2) is 160,000 ops, which is very fast.
    # Let's use np.convolve for simplicity and accuracy, or FFT if needed.
    # Let's try FFT for speed.
    n_fft = 2 * N
    f_x = np.fft.rfft(x, n=n_fft)
    f_conv = f_x * f_x
    conv = np.fft.irfft(f_conv, n=n_fft)[:2*N-1]
    
    # The autoconvolution of a function on [-1/4, 1/4] with 400 intervals.
    # The peak value of the autoconvolution is bounded by the max absolute value of the discrete convolution.
    # Note: The scaling factor depends on the interval width.
    # Interval width h = (1/4 - (-1/4)) / 400 = 0.5 / 400 = 1/800.
    # The integral autoconvolution at lag k is approx h * sum x_i x_{i+k}.
    # So the L_inf norm of the integral autoconvolution is h * max |conv|.
    h = 0.5 / N
    return h * np.max(np.abs(conv))

def _local_search(x, time_limit):
    """
    Local search with single and multi-flip moves.
    """
    x = x.copy()
    best_score = _c3_upper_bound(x)
    start_time = time.time()
    
    # Precompute convolution for fast updates?
    # Updating convolution after a flip is O(N) if we do it naively, but we can use the structure.
    # However, for N=400, recomputing via FFT is ~10us. 
    # Let's just recompute for simplicity first, then optimize if needed.
    # Actually, let's use a direct update method for speed.
    # conv[k] = sum_i x[i] x[i+k]
    # If we flip x[j], the change in conv[k] is:
    # For each k, the terms involving x[j] are x[j]x[j+k] and x[j]x[j-k] (if within bounds).
    # This is complex to update incrementally without errors.
    # Given N=400, np.convolve is O(N^2) ~ 160k ops. In Python/NumPy, this is ~1ms.
    # 150s / 1ms = 150,000 iterations. That's plenty.
    
    # Let's use a smarter update.
    # conv = np.convolve(x, x, mode='full')
    # If we flip x[i] -> -x[i]:
    # The new conv is conv - 2 * (contribution of x[i])
    # The contribution of x[i] to conv[k] is x[i]*x[k-i] (roughly).
    # Let's stick to recomputation for safety, it's fast enough for N=400.
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try single flips
        for i in range(N):
            old_val = x[i]
            x[i] = -old_val
            new_score = _c3_upper_bound(x)
            if new_score < best_score - 1e-12:
                best_score = new_score
                improved = True
            else:
                x[i] = old_val
        
        # Try double flips (random pairs) to escape local minima
        for _ in range(10): # 10 random double flips per pass
            i, j = random.sample(range(N), 2)
            old_i, old_j = x[i], x[j]
            x[i] = -old_i
            x[j] = -old_j
            new_score = _c3_upper_bound(x)
            if new_score < best_score - 1e-12:
                best_score = new_score
                improved = True
            else:
                x[i] = old_i
                x[j] = old_j
        
        if not improved:
            # No improvement in this full pass, break or do a more aggressive search?
            # Let's break to save time for other restarts or polish.
            break
            
    return x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_x = None
    best_score = float('inf')
    
    # Strategy:
    # 1. Start with random initializations.
    # 2. Run local search on each.
    # 3. Keep the best result.
    
    # Number of restarts depends on time available.
    # Estimate time per restart:
    # Local search: ~50-100 passes * (N single flips + 10 double flips) * cost of eval.
    # Cost of eval: ~1ms (FFT) or ~0.5ms (convolve).
    # N=400.
    # Pass cost: 400 * 1ms = 0.4s.
    # 50 passes = 20s.
    # So we can do maybe 5 restarts in 100s.
    
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        x = np.random.choice([-1.0, 1.0], size=N)
        
        # Run local search with a time limit slightly less than time_per_restart
        # to allow for overhead
        x_opt, score = _local_search(x, time_per_restart * 0.9)
        
        if score < best_score:
            best_score = score
            best_x = x_opt
            
        # If we have plenty of time left, do more restarts?
        # Or refine the best solution further?
        
    # Final polish on the best solution
    if best_x is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            # Do a final high-precision polish
            # Maybe try triple flips or longer search
            best_x, best_score = _local_search(best_x, remaining_time * 0.8)
    
    # Ensure output is a sequence of signed step heights
    # The task says "sequence of signed step heights".
    # The current program returns np.ones(N).
    # So we return an array of -1 and 1.
    
    if best_x is None:
        # Fallback
        return np.ones(N)
        
    return best_x
# EVOLVE-BLOCK-END
