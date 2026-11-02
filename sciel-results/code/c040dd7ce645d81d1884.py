"""Improved program for the third autocorrelation inequality task."""
import numpy as np
import time

N = 400
DEADLINE = 110  # safety margin before 120s kill

def _score(s):
    """Upper bound on C3; lower is better."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    acf = acf / N
    return float(np.max(np.abs(acf[1:])))

def _block_flip_local_search(s, block_size, max_passes=100):
    """
    Perform local search by flipping contiguous blocks of a given size.
    This is more effective for autocorrelation than single-bit flips.
    """
    cur_score = _score(s)
    s = s.copy()
    
    for _ in range(max_passes):
        improved = False
        # Try all starting positions for the block
        # We can wrap around or just stay in bounds. Let's stay in bounds for simplicity,
        # but wrapping is often better. Let's do non-wrapping first for speed, 
        # or just iterate 0 to N-1.
        for i in range(N):
            # Define the block indices
            # If we don't wrap, we only flip if i+block_size <= N
            if i + block_size > N:
                continue
                
            # Flip the block
            s[i:i+block_size] *= -1
            
            new_score = _score(s)
            
            if new_score < cur_score:
                cur_score = new_score
                improved = True
            else:
                # Revert
                s[i:i+block_size] *= -1
                
        if not improved:
            break
            
    return s, cur_score

def construct():
    t_start = time.time()
    
    # Generate diverse initial candidates
    rng = np.random.default_rng(42)
    candidates = []
    
    # Random sequences
    for _ in range(5):
        candidates.append(rng.choice([-1, 1], size=N).astype(np.int8))
        
    # Chirp-like
    n = np.arange(N)
    phase = np.pi * n * n / N
    candidates.append(np.sign(np.cos(phase)).astype(np.int8))
    
    # Evaluate all candidates and pick the best initial one
    scores = [_score(c) for c in candidates]
    best_idx = np.argmin(scores)
    best_seq = candidates[best_idx].copy()
    best_score = scores[best_idx]
    
    # Define block sizes to try. Smaller blocks are more local, larger blocks are more global.
    # Knowledge suggests 4-32 is effective.
    block_sizes = [2, 4, 8, 16, 32]
    
    # Perform local search with each block size
    # We alternate between sizes to escape local minima specific to one size.
    current_seq = best_seq
    current_score = best_score
    
    while time.time() - t_start < DEADLINE:
        for b_size in block_sizes:
            if time.time() - t_start > DEADLINE:
                break
                
            # Perform a few passes of local search with this block size
            # The number of passes can be adjusted, but 10-20 is usually enough for convergence
            new_seq, new_score = _block_flip_local_search(current_seq, b_size, max_passes=20)
            
            if new_score < best_score:
                best_score = new_score
                best_seq = new_seq.copy()
            
            current_seq = new_seq
            current_score = new_score
            
    return best_seq
