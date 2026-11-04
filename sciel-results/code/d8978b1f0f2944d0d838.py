"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _thue_morse(n):
    """Generate the Thue-Morse sequence of length n."""
    # The Thue-Morse sequence t_n is defined by t_n = popcount(n) mod 2.
    # For efficiency with numpy, we can use bit operations or recursive construction.
    # Recursive: T_0 = 0, T_{k+1} = T_k, 1-T_k
    # Length needs to be power of 2 for simple recursion, then truncate.
    if n == 0:
        return np.array([], dtype=np.int8)
    
    # Find next power of 2
    k = 1
    while k < n:
        k *= 2
    
    t = np.zeros(1, dtype=np.int8)
    while len(t) < k:
        # t_next = [t, 1-t]
        t = np.concatenate([t, 1 - t])
    
    return t[:n]

def _autoconv_peak(x):
    """Calculate the maximum absolute value of the autoconvolution (correlation) of x."""
    # x is a vector of length N.
    # The autoconvolution (or autocorrelation depending on definition) 
    # for a finite sequence is often defined via FFT.
    # The task asks for "autoconvolution upper bound".
    # For a step function on [-1/4, 1/4], the autoconvolution (f*f) is a continuous function.
    # However, the "upper bound on C3" usually refers to the maximum of the autoconvolution.
    # Let's compute the discrete autoconvolution (circular or linear?).
    # "signed step function with 400 equal intervals"
    # The continuous autoconvolution of a step function is a piecewise quadratic function.
    # The peak is typically at lag 0 (if sum x_i^2 is maximized) or near 0.
    # For binary {-1, 1}, sum x_i^2 = N. The peak at lag 0 is N (if normalized) or N*h^2.
    # Wait, the "Third autocorrelation inequality" usually involves the integral of |f*f|.
    # But the score is "upper bound on C3".
    # Let's assume the metric is the max of the absolute value of the autoconvolution (continuous).
    # For a step function f(t) = x_i on interval i, the autoconvolution (f*f)(t) is the sum of x_i x_j * overlap_length.
    # The maximum value of the autoconvolution for a binary sequence is related to the "maximum partial sum" or similar.
    # Actually, for f in L2, ||f*f||_inf <= ||f||_2^2.
    # If we normalize ||f||_2 = 1, then ||f*f||_inf <= 1.
    # The "seed" program returns ones. If we normalize, ones has peak 1.
    # If we don't normalize, ones has peak N.
    # The task says "score is the upper bound on C3".
    # Inspiration 2 mentions `_autoconvolution_max_abs`.
    # Let's calculate the max absolute value of the continuous autoconvolution.
    # For a step function with heights x_i, the autoconvolution at lag 0 is sum x_i^2 * h = sum x_i^2 * (1/200).
    # If x_i in {-1, 1}, sum x_i^2 = 400. Peak = 400 * 0.005 = 2.
    # If we use the discrete approximation (sum x_i x_j), the peak is N=400.
    # Let's look at the "seed" score 1.0.
    # If the metric is the max autoconvolution value, and seed is ones, score is 1.0.
    # This implies the metric is normalized such that the all-ones sequence gives 1.0?
    # Or maybe the metric is the *minimum* possible peak?
    # "score is the upper bound on C3... lower is better".
    # "1.0 equals the seed program".
    # So if I return ones, score is 1.0.
    # If I return a sequence with lower peak autoconvolution, score < 1.0?
    # But the contract says "1.0 equals the seed program and above 1.0 is an improvement".
    # This is contradictory to "lower is better" unless the score is defined as 1.0 / (metric) or similar?
    # "The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
    # So Score = Seed_Value / My_Value.
    # If My_Value < Seed_Value, Score > 1.0.
    # So I need to MINIMIZE the raw value (autoconvolution peak).
    
    # Let's compute the continuous autoconvolution peak.
    # f(t) is constant on intervals [i*h, (i+1)*h].
    # (f*f)(tau) = integral f(t) f(t-tau) dt.
    # For discrete steps, this is equivalent to the triangular convolution of the sequence x.
    # c_k = sum_{i} x_i x_{i+k} for k=0...N-1 (linear correlation).
    # The continuous peak is max_k c_k * h? No.
    # The overlap length for lag k (integer steps) is h * (1 - |k|/N) ? No, that's for continuous shift.
    # If we shift by integer steps, the overlap is exactly h for each pair.
    # The peak is at lag 0: sum x_i^2 * h.
    # For binary x, sum x_i^2 = N. Peak = N*h = N * (1/200) = 2.
    # If the seed (ones) gives score 1.0, and Score = Seed/Value, then Seed_Value = 1.0 * Value?
    # Or is the "Value" reported by the evaluator the raw peak?
    # If Seed_Value = 2 (raw peak for ones), and Score = 2/2 = 1.0.
    # Then I need to minimize the raw peak.
    
    # Let's compute the max of the linear autoconvolution (correlation) of the sequence x.
    # To be safe, I will compute the max absolute value of the continuous autoconvolution.
    # The continuous autoconvolution of a step function is a piecewise quadratic function.
    # Its maximum occurs at one of the "corners" (integer lags) or in between?
    # For a step function, the derivative of the autoconvolution is piecewise linear (from the step changes).
    # The maximum is typically at a point where the derivative changes sign, which corresponds to an integer lag for binary sequences?
    # Actually, for binary sequences, the autoconvolution is symmetric.
    # Let's just compute the discrete autoconvolution (sum x_i x_{i+k}) and multiply by h?
    # Or maybe the evaluator just uses the discrete sum?
    # Given "signed step function", the physical quantity is the continuous integral.
    # Let's calculate the continuous peak.
    
    # Efficient way:
    # The continuous autoconvolution g(tau) for step function is:
    # g(k*h + delta) = h * ( sum_{i=0}^{N-1-k} x_i x_{i+k} * (1 - delta) + sum_{i=0}^{N-1-(k+1)} x_i x_{i+k+1} * delta )
    # This is linear between integer lags. Thus the max is at an integer lag.
    # So we just need max_k | sum_{i} x_i x_{i+k} | * h.
    # Wait, is it absolute value? "absolute value differently" in paper.
    # "execution must explicitly select a metric".
    # Usually "upper bound on C3" (constant in inequality) is positive.
    # Let's assume we minimize max_k | (x*x)_k | * h.
    
    x = np.asarray(x)
    # Compute linear correlation
    # Using FFT for speed
    n_fft = 2 ** int(np.ceil(np.log2(2 * N)))
    X = np.fft.fft(x, n=n_fft)
    corr = np.fft.ifft(X * np.conj(X)).real
    # corr[0:N] corresponds to lags 0 to N-1
    # corr[N:2N-1] corresponds to negative lags
    # We want max absolute value
    peak = np.max(np.abs(corr[:N]))
    # Scale by h = 1/200
    return peak * (1.0 / 200.0)

def _local_search(x, max_iters=10000):
    """Perform local search by flipping bits."""
    current_x = x.copy()
    current_val = _autoconv_peak(current_x)
    
    for i in range(max_iters):
        improved = False
        # Try flipping each bit
        # To speed up, we can estimate the change in autoconvolution for a flip.
        # But for N=400, calculating the new peak is O(N log N) with FFT.
        # 400 * 10000 = 4M FFTs of size 1024. Might be too slow in Python.
        # Let's use a heuristic or smaller steps.
        
        # Alternative: Randomly pick a few indices to flip
        # Or use the gradient of the squared peak?
        
        # Let's try a simple hill climbing with a limit on evaluations per step.
        # We can approximate the change.
        # Let's just do a limited number of full sweeps if time permits.
        
        # For this implementation, let's do a few random restarts or steps.
        # Given the 120s limit, we need to be careful.
        # Let's try flipping one bit at a time, but only if it improves the peak significantly.
        
        # Optimization: Precompute the current correlation vector?
        # If we flip x_j, the new correlation c'_k = c_k - 2*x_j*(x_{j+k} + x_{j-k}) ... roughly.
        # This is complex to update the MAX efficiently.
        
        # Let's just do a few iterations of "flip random bit, if better keep, else revert".
        # This is a stochastic local search.
        
        idx = np.random.randint(N)
        new_x = current_x.copy()
        new_x[idx] *= -1
        
        new_val = _autoconv_peak(new_x)
        if new_val < current_val:
            current_x = new_x
            current_val = new_val
            improved = True
            
    return current_x, current_val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Generate Thue-Morse sequence
    # Thue-Morse has low autocorrelation.
    x_tm = _thue_morse(N)
    # Map 0->1, 1->-1 for signed step function
    x_tm_signed = 2 * x_tm - 1
    x_tm_signed = x_tm_signed.astype(np.int8)
    
    best_x = x_tm_signed
    best_val = _autoconv_peak(best_x)
    
    # Also try the inverse (just in case)
    x_tm_inv = -x_tm_signed
    val_inv = _autoconv_peak(x_tm_inv)
    if val_inv < best_val:
        best_x = x_tm_inv
        best_val = val_inv
        
    # 2. Local Search
    # We have ~110s.
    # Each eval is ~1ms (FFT 1024).
    # We can do ~100,000 evals.
    # Let's run a local search loop until deadline.
    
    current_x = best_x
    current_val = best_val
    
    # Use a simple random walk local search
    # To avoid getting stuck, we might need multiple restarts or simulated annealing,
    # but let's stick to simple hill climbing first.
    
    # To speed up, we can check if we have time for another iteration.
    # We'll iterate while time < deadline - 2s.
    
    # Optimization: Instead of full FFT for every single flip, we can batch or use a smarter metric?
    # No, let's just run as many as we can.
    
    # We'll use a loop that picks a random index to flip.
    # If it improves, we keep it. If not, we revert.
    # To escape local minima, we could allow some worsening moves (annealing), 
    # but simple hill climbing is safer for "refinement".
    
    # Let's add a "kick" every 1000 steps to break symmetry/stuck states?
    # Or just rely on the random nature.
    
    # We'll run until time is up.
    while time.time() < deadline - 2.0:
        idx = np.random.randint(N)
        old_val = current_x[idx]
        current_x[idx] *= -1
        
        new_val = _autoconv_peak(current_x)
        
        if new_val < current_val:
            current_val = new_val
        else:
            # Revert
            current_x[idx] = old_val
            # Optional: If stuck, try a random kick?
            # For now, just continue.
            
    return current_x

# EVOLVE-BLOCK-END
