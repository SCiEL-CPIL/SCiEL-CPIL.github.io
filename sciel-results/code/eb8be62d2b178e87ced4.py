import numpy as np
import time

N = 400
N_FFT = 1024

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=N_FFT)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=N_FFT)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _calc_delta_acf(s, i):
    """
    Compute the change in autocorrelation when flipping s[i].
    Returns delta array where A_new[k] = A[k] + delta[k].
    """
    old_si = s[i]
    delta = np.zeros(N)
    
    # For k in 0..N-1:
    # A_new[k] = A[k] - 2*old_si*(s[i+k] + s[i-k]) (with boundary handling)
    
    # Part 1: s[i+k] is defined for k in 0..N-1-i
    # s[i+k] corresponds to s[i:N]
    # delta[k] -= 2*old_si * s[i+k]
    if N - i > 0:
        delta[:N-i] -= 2.0 * old_si * s[i:N]
        
    # Part 2: s[i-k] is defined for k in 0..i
    # s[i-k] corresponds to s[i::-1] (which is s[i], s[i-1], ..., s[0])
    # delta[k] -= 2*old_si * s[i-k]
    if i >= 0:
        delta[:i+1] -= 2.0 * old_si * s[i::-1]
        
    return delta

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Number of restarts to try
    # We want to balance restart count with local search depth.
    # Let's use a fixed number of restarts if time permits, or time-based.
    # Given 140s budget, and local search taking ~0.1-0.5s per restart, we can do ~100-1000 restarts.
    # But local search is much faster with O(N) updates.
    
    num_restarts = 50 # Start with a reasonable number, adjust based on time
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search
        improved = True
        passes = 0
        max_passes = 20 # Limit passes to prevent infinite loops or slow convergence
        
        while improved and passes < max_passes and time.time() < deadline - 5.0:
            improved = False
            passes += 1
            # Random order to avoid bias
            indices = np.random.permutation(N)
            
            for i in indices:
                if time.time() > deadline - 5.0:
                    break
                
                # Calculate delta
                delta = _calc_delta_acf(s, i)
                
                # Check if flipping improves the max abs ACF
                # A_new = A + delta
                # We want to minimize max(|A_new[1:]|)
                
                # Optimization: Instead of creating A_new, we can check conditions.
                # But creating A_new is O(N) and simple.
                A_new = A + delta
                val_new = _max_abs_acf_from_array(A_new)
                
                if val_new < current_val:
                    # Accept flip
                    s[i] = -s[i]
                    A = A_new
                    current_val = val_new
                    improved = True
                    # Note: We don't break here, we continue to next i
                    # This is a "first improvement" or "best improvement" within the pass?
                    # Current logic is first improvement (greedy).
                # else: reject flip, s and A remain unchanged
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
            
        # Optional: If we have time, do a second phase of finer search?
        # For now, stick to multiple restarts.

    return best_seq
