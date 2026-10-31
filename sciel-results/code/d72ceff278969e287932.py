"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _calc_max_side_lobe(x):
    """Calculate the maximum absolute value of the linear autocorrelation at non-zero lags."""
    # Zero-pad to length 2N-1 to compute linear autocorrelation via FFT
    # Using n=1024 (next power of 2 >= 2*400=800) for faster FFT
    x_pad = np.zeros(1024)
    x_pad[:N] = x
    X = np.fft.fft(x_pad)
    # Autocorrelation is IFFT of |FFT|^2
    # We only need the first N lags (0 to N-1). 
    # Due to symmetry, C[k] = C[N-k] for circular, but for linear padded it's different.
    # Actually, linear autocorrelation via FFT:
    # C = ifft(fft(x_pad) * conj(fft(x_pad)))
    # The first N values correspond to lags 0..N-1.
    C = np.fft.ifft(X * np.conj(X)).real[:N]
    # Exclude lag 0
    return np.max(np.abs(C[1:]))

def _greedy_local_search(x, max_iterations=1000):
    """
    Perform greedy local search. Flip bit i if it reduces the max side lobe.
    Returns improved x and best value found.
    """
    current_val = _calc_max_side_lobe(x)
    best_x = x.copy()
    best_val = current_val
    x = x.copy()
    
    # We iterate through indices multiple times to allow for non-monotonic improvements
    # though greedy is monotonic, the order matters.
    # Simple strategy: sweep indices 0..N-1, flip if better.
    # Repeat until no change or max iterations.
    
    for _ in range(max_iterations):
        improved = False
        for i in range(N):
            # Try flipping i
            old_val = x[i]
            x[i] = -old_val
            
            new_val = _calc_max_side_lobe(x)
            
            if new_val < current_val:
                current_val = new_val
                improved = True
                if new_val < best_val:
                    best_val = new_val
                    best_x = x.copy()
            else:
                # Revert
                x[i] = old_val
        
        if not improved:
            break
            
    return best_x, best_val

def construct():
    start_time = time.time()
    # Strict deadline: 110s to be safe against 120s limit
    deadline = start_time + 110.0
    
    # Initial guess: Thue-Morse sequence is a good starting point
    # But we will also try random restarts.
    
    # Thue-Morse
    t = np.array([1])
    while len(t) < N:
        t = np.concatenate([t, -t])
    x_init = t[:N].astype(np.int8)
    
    best_x, best_val = _greedy_local_search(x_init, max_iterations=50)
    
    # Iterated Local Search with Random Restarts
    # We will run several restarts.
    # Each restart: Random init -> Greedy Search.
    # We allocate time budget to each restart.
    
    # Estimate time per greedy search step.
    # Let's just run as many as we can within the deadline.
    
    np.random.seed(42) # For reproducibility in evaluation if needed, but usually not required.
    
    while time.time() < deadline:
        # Check if we have enough time for another round
        # A greedy search on N=400 takes some time.
        # Let's do a quick time check.
        t_start_round = time.time()
        
        # Random initialization
        x_rand = np.random.choice([-1, 1], size=N).astype(np.int8)
        
        # Run greedy search with a limited number of sweeps to keep it fast
        # If a sweep doesn't improve, it stops.
        # Limit sweeps to avoid getting stuck in a long, unproductive sweep.
        x_new, val_new = _greedy_local_search(x_rand, max_iterations=20)
        
        if val_new < best_val:
            best_val = val_new
            best_x = x_new
        
        # Also try perturbing the current best
        # This is ILS: take best, perturb, search.
        x_perturb = best_x.copy()
        num_flips = max(1, N // 20) # Flip ~5% of bits
        flip_indices = np.random.choice(N, size=num_flips, replace=False)
        x_perturb[flip_indices] *= -1
        
        x_ils, val_ils = _greedy_local_search(x_perturb, max_iterations=20)
        
        if val_ils < best_val:
            best_val = val_ils
            best_x = x_ils

    return best_x
# EVOLVE-BLOCK-END
