import numpy as np
import time

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin
    
    N = 600
    # Initial guess: random binary
    h = np.random.randint(0, 2, N).astype(float)
    
    def get_score(h):
        # Ensure non-negative
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s == 0:
            return np.inf
        conv = np.convolve(h, h, mode='full')
        # The peak is likely near the center, but let's take max
        peak = np.max(conv)
        return peak / s

    best_h = h.copy()
    best_score = get_score(best_h)
    
    # Local search
    # We can use a simple bit-flipping strategy
    # To make it faster, we can batch operations or use a smarter search.
    
    # Let's try a simple hill climbing with random restarts if stuck
    # But 600 dimensions is small enough for a few thousand iterations of bit flips.
    
    # Optimization: 
    # Instead of recalculating full convolution every step, 
    # we can update the convolution incrementally? 
    # conv[k] = sum h[i] h[k-i].
    # If we change h[j], conv changes by h[j]*(h_new - h_old) convolved with delta at j?
    # Actually, d(conv)/dh[j] = h convolved with delta_j + delta_j convolved with h = h[j:] shifted + h[:j] shifted.
    # This allows O(N) update per step instead of O(N^2) or O(N log N).
    
    # Let's implement the incremental update for speed.
    
    h = best_h.copy()
    conv = np.convolve(h, h, mode='full')
    s = np.sum(h)
    peak = np.max(conv)
    score = peak / s
    best_score = score
    best_h = h.copy()
    
    # Precompute indices for convolution update
    # conv[m] = sum_{i} h[i] h[m-i]
    # If we change h[j] by delta, 
    # new_conv[m] = conv[m] + delta * (h[m-j] + h[j-m])  (with boundary checks)
    # Note: for m-j and j-m, we need to handle indices carefully.
    
    N_conv = 2 * N - 1
    
    def update_conv(conv, h, j, delta):
        # h[j] changes by delta
        # For each m in conv:
        # term 1: h[j] * h[m-j] -> delta * h[m-j]
        # term 2: h[m-j] * h[j] -> h[m-j] * delta  (same if symmetric? No, conv is sum h[i]h[m-i])
        # Actually, conv[m] = sum_i h[i] h[m-i].
        # If h[j] changes, the terms involving h[j] are:
        # i=j: h[j] h[m-j]
        # m-i=j => i=m-j: h[m-j] h[j]
        # So the change is delta * h[m-j] + h[m-j] * delta = 2 * delta * h[m-j] (if m-j != j)
        # If m-j == j (i.e., m=2j), the term is h[j]^2. Change is (h[j]+delta)^2 - h[j]^2 = 2 h[j] delta + delta^2.
        # The formula 2 * delta * h[m-j] works for m != 2j.
        # For m=2j, we need to be careful.
        
        # Let's just use the vectorized update.
        # indices of m such that m-j is valid index in h (0 to N-1)
        # m ranges from 0 to 2N-2.
        # m-j ranges from -j to 2N-2-j.
        # Valid h indices are 0 to N-1.
        
        # This is getting complex to optimize manually. 
        # Given N=600, np.convolve is fast (C implementation). 
        # 1000 iterations * 600 convs = 600k convs. 
        # Each conv is 600x600? No, np.convolve is O(N^2) or O(N log N).
        # N=600 is small. 600*600 = 360,000 operations.
        # 1000 steps * 360k ops = 360M ops. In Python/NumPy, this might take a few seconds.
        # We can do 10,000 steps easily within 150s.
        
        pass

    # Let's stick to full recalculation for simplicity and robustness, as N is small.
    # We will do a random restart local search.
    
    for restart in range(10): # 10 restarts
        if time.time() - start_time > timeout:
            break
            
        if restart == 0:
            h = best_h.copy()
        else:
            h = np.random.randint(0, 2, N).astype(float)
            # Mix with best?
            h = np.where(np.random.rand(N) < 0.1, best_h, h)
            
        conv = np.convolve(h, h, mode='full')
        s = np.sum(h)
        if s == 0: continue
        peak = np.max(conv)
        score = peak / s
        
        improved = True
        while improved and (time.time() - start_time < timeout):
            improved = False
            # Pick a random index to flip
            j = np.random.randint(N)
            old_val = h[j]
            h[j] = 1.0 - old_val # Flip binary
            
            new_conv = np.convolve(h, h, mode='full')
            new_s = np.sum(h)
            if new_s == 0:
                h[j] = old_val
                continue
            new_peak = np.max(new_conv)
            new_score = new_peak / new_s
            
            if new_score < score:
                score = new_score
                conv = new_conv
                s = new_s
                peak = new_peak
                improved = True
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
            else:
                h[j] = old_val
                
        # After local search, maybe do a "kick" if time remains
        if time.time() - start_time < timeout:
             # Randomly flip a block
             start_idx = np.random.randint(N-10)
             h[start_idx:start_idx+10] = 1 - h[start_idx:start_idx+10]
             # Re-evaluate and continue? 
             # For now, just let the restart handle it.
             
    return best_h.tolist()
