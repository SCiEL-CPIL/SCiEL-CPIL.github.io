"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute FFT size
_n_fft = 1
while _n_fft < 2 * N:
    _n_fft *= 2

def _compute_objective(x):
    """Compute the upper bound on C3 for a given sequence.
    
    The task states: "Paper and published result notebook place the absolute value differently; 
    execution must explicitly select a metric."
    
    Based on the "Third autocorrelation inequality" context, the standard bound for C3 
    often involves the sum of squares of the non-zero autocorrelations (related to the 
    L2 norm of the side-lobes) or the maximum absolute value. 
    
    However, looking at the inspirations, they all target `max(|A[1:]|)`. 
    Let's verify which metric yields the "seed" score of 1.0 for `np.ones(N)`.
    
    If metric is `max(|A[1:]|)`:
    For x = [1, 1, ..., 1] (N=400):
    Autocorrelation A[k] = N - k for k=0..N-1.
    A[0] = 400, A[1] = 399, ..., A[399] = 1.
    max(|A[1:]|) = 399.
    Seed value = 399.
    Score = 399 / 399 = 1.0. This matches the inspiration analysis.
    
    If metric is `sum(|A[1:]|)`:
    Sum = 399 + 398 + ... + 1 = 399 * 400 / 2 = 79800.
    Score = 79800 / 79800 = 1.0. This also matches.
    
    However, the "Third Autocorrelation Inequality" in signal processing (e.g., related to 
    Barker codes or low autocorrelation sequences) typically aims to minimize the peak 
    sidelobe level (PSL), which is `max(|A[k]| for k != 0)`. 
    Minimizing the PSL is a more standard "design" goal for such sequences. 
    The sum of squares (energy) is also common. 
    Let's stick with `max(|A[1:]|)` as it is the most common "inequality" bound for 
    correlation peaks, and the inspirations explicitly use it.
    """
    x_f = x.astype(np.float64)
    F = np.fft.rfft(x_f, n=_n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=_n_fft)
    A = np.real(acf[:N])
    # Exclude lag 0 (index 0)
    return float(np.max(np.abs(A[1:])))

def _eval_fast(s):
    """Fast evaluation of the objective for a sequence s."""
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=_n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=_n_fft)
    A = np.real(acf[:N])
    return float(np.max(np.abs(A[1:])))

def _local_search_block(s, best_val, deadline, block_sizes):
    """
    Perform local search using block flips.
    This explores larger regions of the sign space than single-bit flips.
    """
    current_val = best_val
    improved = True
    passes = 0
    max_passes = 50
    
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        # Random order of block sizes and positions
        # We try flipping blocks of various sizes
        for _ in range(20): # Try 20 random block flips per pass
            if time.time() > deadline:
                break
            
            # Pick a random block size from the allowed set
            b_size = np.random.choice(block_sizes)
            # Pick a random start position
            if b_size >= N:
                start = 0
            else:
                start = np.random.randint(0, N - b_size + 1)
            
            # Flip the block
            s[start:start+b_size] *= -1
            
            new_val = _eval_fast(s)
            
            if new_val < current_val:
                current_val = new_val
                improved = True
            else:
                # Revert
                s[start:start+b_size] *= -1
                
    return s, current_val

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We return by 140s.
    deadline = start_time + 140.0
    
    # Initialize with random sequence
    np.random.seed(42)
    best_x = np.random.choice([-1, 1], size=N)
    best_obj = _compute_objective(best_x)
    
    # Block sizes for local search
    block_sizes = [4, 8, 16, 32]
    
    # Multi-start strategy
    # We will perform several restarts. Each restart starts from a random sequence
    # or the current best, and performs local search.
    
    # Since we have a time budget, we can do a few full passes of local search
    # on the initial random sequence, then try some restarts if time permits.
    
    # Phase 1: Local search on the initial random sequence
    current_x = best_x.copy()
    current_x, current_obj = _local_search_block(current_x, best_obj, deadline, block_sizes)
    
    if current_obj < best_obj:
        best_obj = current_obj
        best_x = current_x.copy()
    
    # Phase 2: If time remains, try a few more restarts with different random seeds
    # to escape potential local minima.
    while time.time() < deadline - 5.0: # Leave 5s safety buffer
        # Generate a new random sequence
        new_x = np.random.choice([-1, 1], size=N)
        new_obj = _compute_objective(new_x)
        
        # If this random start is already good, or we want to explore, do local search
        # We only do local search if we have enough time for it
        # Estimate time for local search: ~20 block flips * 50 passes * FFT cost
        # FFT cost is small. The loop overhead is the main cost.
        
        # Let's try local search on this new random start
        local_x, local_obj = _local_search_block(new_x, new_obj, deadline, block_sizes)
        
        if local_obj < best_obj:
            best_obj = local_obj
            best_x = local_x.copy()
            
        # If we found a significantly better solution, we might want to 
        # continue searching from there? The function already returns best.
        # We just keep trying new random starts to see if we find a better basin.
        
    return best_x
# EVOLVE-BLOCK-END
