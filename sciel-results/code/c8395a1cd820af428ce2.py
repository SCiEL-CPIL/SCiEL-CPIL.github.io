import numpy as np
import time

N = 400
DEADLINE = 115.0  # Safety margin below 150s evaluator limit

def _compute_r3(h):
    """
    Compute R3[k] = sum_i h[i] h[i+k] h[i+2k] for k = 0 .. N//2
    Using FFT for initialization speed.
    """
    h_float = h.astype(np.float64)
    size = 2 * N + 1
    h_fft = np.fft.fft(h_float, n=size)
    r3_fft = h_fft * h_fft * h_fft
    r3_full = np.fft.ifft(r3_fft).real
    # r3_full corresponds to circular convolution? No, linear convolution of h,h,h.
    # The lag k term in autocorrelation is at index k in the result?
    # Let's verify.
    # conv(h, h, h)[k] = sum_i h[i] h[i+k] h[i+2k] ?
    # Actually, standard autocorrelation is defined with positive lags.
    # For k=0, sum h[i]^3 = sum h[i] (since h[i]=±1).
    # The FFT method computes the cyclic autocorrelation if not padded, or linear if padded.
    # With padding size 2N+1, the result at index k (0 to N) should be the sum for lag k.
    # Let's double check the index mapping.
    # np.fft.ifft(fft(h)**3) gives the circular convolution.
    # With zero padding to 2N+1, the first N+1 elements contain the linear convolution.
    # Specifically, r3_full[k] for k=0..N should be the desired R3[k].
    return r3_full[:N+1]

def _score(h):
    r3 = _compute_r3(h)
    return np.max(np.abs(r3))

def construct():
    start_time = time.time()
    
    # Initialize with a random pattern or a specific structured pattern
    # A random pattern is a good starting point for coordinate descent.
    # We can also try a few different initializations.
    
    best_h = np.random.choice([-1.0, 1.0], size=N)
    best_score = _score(best_h)
    best_r3 = _compute_r3(best_h)
    
    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE

    # Coordinate Descent
    # We will perform multiple passes.
    # In each pass, we try to flip each bit.
    # We maintain the current R3 array and update it incrementally.
    
    h = best_h.copy()
    r3 = best_r3.copy()
    current_score = best_score
    
    max_passes = 2000 # Limit passes just in case, but time is the real constraint
    
    for p in range(max_passes):
        if not elapsed_ok():
            break
            
        improved = False
        # Randomize order to avoid systematic bias
        order = np.random.permutation(N)
        
        for i in order:
            if not elapsed_ok():
                return best_h
            
            # Calculate the change in r3 for all k if we flip h[i]
            # h[i] becomes -h[i]
            # Delta for a specific k:
            # The terms in R3[k] involving h[i] are:
            # 1. i is the first index: h[i] h[i+k] h[i+2k]. Valid if i+2k < N.
            # 2. i is the second index: h[i-k] h[i] h[i+k]. Valid if i-k >= 0 and i+k < N.
            # 3. i is the third index: h[i-2k] h[i-k] h[i]. Valid if i-2k >= 0.
            
            # We need to update r3[k] for all k where any of these terms exist.
            # Max k is roughly N/2.
            # We can just loop k from 0 to N//2 and check validity.
            # This is O(N) per flip.
            
            delta_r3 = np.zeros_like(r3)
            h_i = h[i]
            
            # Pre-compute terms that depend on k
            # Term 1: h[i+k] h[i+2k]
            # Term 2: h[i-k] h[i+k]
            # Term 3: h[i-2k] h[i-k]
            
            # To make it fast, we can vectorize the calculation of the sum of terms.
            # But the validity conditions make it tricky.
            # Let's stick to a simple loop over k, but optimized with numpy slicing if possible.
            # Actually, N=400 is small. A Python loop over 200 k's is fine.
            
            for k in range(N // 2 + 1):
                term_sum = 0.0
                
                # Term 1: h[i] is at pos i. Others at i+k, i+2k.
                if i + 2*k < N:
                    term_sum += h[i+k] * h[i+2*k]
                
                # Term 2: h[i] is at pos i. Others at i-k, i+k.
                if i - k >= 0 and i + k < N:
                    term_sum += h[i-k] * h[i+k]
                
                # Term 3: h[i] is at pos i. Others at i-2k, i-k.
                if i - 2*k >= 0:
                    term_sum += h[i-2*k] * h[i-k]
                
                if term_sum != 0.0:
                    # Delta is -2 * h_i * term_sum
                    delta_r3[k] = -2.0 * h_i * term_sum
            
            # Check if flipping improves the score
            # New r3 = r3 + delta_r3
            # New score = max(abs(r3 + delta_r3))
            
            # Optimization: Only check if delta is non-zero
            if np.any(delta_r3 != 0):
                new_r3 = r3 + delta_r3
                new_score = np.max(np.abs(new_r3))
                
                if new_score < current_score - 1e-12:
                    # Accept flip
                    h[i] = -h_i
                    r3 = new_r3
                    current_score = new_score
                    improved = True
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_h = h.copy()
                        best_r3 = r3.copy()
        
        if not improved:
            # No improvement in this pass, try a random restart or break
            # Let's do a few random restarts if time allows
            if elapsed_ok():
                 # Generate a new random pattern
                 h_new = np.random.choice([-1.0, 1.0], size=N)
                 score_new = _score(h_new)
                 if score_new < best_score:
                     best_score = score_new
                     best_h = h_new.copy()
                     best_r3 = _compute_r3(best_h)
                     h = best_h.copy()
                     r3 = best_r3.copy()
                     current_score = best_score
                     # Continue with coordinate descent on this new pattern
                 else:
                     # If no improvement from restart, maybe break to save time
                     # Or continue with current best if it was a local min
                     pass
            else:
                break

    # Final check
    final_score = _score(best_h)
    if final_score < best_score:
        best_score = final_score
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print("Constructed:", h[:20])
    print("Score:", _score(h))
