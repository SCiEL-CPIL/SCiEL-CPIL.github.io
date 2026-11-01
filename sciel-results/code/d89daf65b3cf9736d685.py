"""Rewritten program for the suite task; uses simulated annealing with FFT-based objective."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _compute_sidelobe(h):
    """Compute max |autocorr| at non-zero lags using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    return np.max(np.abs(acf[1:]))


def _simulated_annealing(h, deadline):
    """Simulated annealing to minimize max autocorrelation sidelobe."""
    h = h.copy().astype(np.float64)
    current_val = _compute_sidelobe(h)
    best_h = h.copy()
    best_val = current_val
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.001
    # We'll use a cooling schedule based on time
    start_time = time.time()
    
    while time.time() < deadline:
        # Choose a random position to flip
        i = np.random.randint(N)
        
        # Flip the bit
        h[i] = -h[i]
        
        # Compute new objective
        new_val = _compute_sidelobe(h)
        
        # Accept or reject
        if new_val < current_val:
            # Always accept improvement
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Accept with probability exp(-delta/T)
            # Estimate temperature based on elapsed time
            elapsed = time.time() - start_time
            total_time = deadline - start_time
            progress = min(elapsed / total_time, 1.0)
            T = T_start * (T_end / T_start) ** progress
            
            delta = new_val - current_val
            if np.random.rand() < np.exp(-delta / (T + 1e-10)):
                current_val = new_val
            else:
                # Revert
                h[i] = -h[i]
        
        # Check time
        if time.time() >= deadline:
            break
    
    return best_h, best_val


def _multi_start_sa(deadline):
    """Multiple random starts with simulated annealing."""
    best_h = None
    best_val = np.inf
    
    start_time = time.time()
    
    while time.time() < deadline - 2.0:  # Keep 2s buffer
        # Random initialization
        np.random.seed()  # Use system time for randomness
        h = np.random.choice([-1.0, 1.0], size=N)
        
        # Run SA with a sub-deadline
        sub_deadline = min(deadline - 1.0, time.time() + 30.0)  # Max 30s per restart
        h_local, val_local = _simulated_annealing(h, sub_deadline)
        
        if val_local < best_val:
            best_val = val_local
            best_h = h_local
        
        # If we're getting close to deadline, stop
        if time.time() > deadline - 5.0:
            break
    
    if best_h is None:
        # Fallback: return a random sequence
        best_h = np.random.choice([-1.0, 1.0], size=N)
        best_val = _compute_sidelobe(best_h)
    
    return best_h, best_val


def construct():
    """Construct a signed step function that minimizes the autoconvolution upper bound."""
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # First, try a few quick random starts with local search to get a good initial point
    h_init = np.random.choice([-1.0, 1.0], size=N)
    
    # Quick greedy local search to get a decent starting point
    def _greedy_local_search(h, deadline):
        h = h.copy()
        current_val = _compute_sidelobe(h)
        best_h = h.copy()
        best_val = current_val
        
        while time.time() < deadline:
            improved = False
            # Try flipping each bit (in random order for diversity)
            order = np.random.permutation(N)
            for i in order:
                if time.time() >= deadline:
                    break
                h[i] = -h[i]
                val = _compute_sidelobe(h)
                h[i] = -h[i]  # revert
                
                if val < best_val:
                    h[i] = -h[i]  # keep flip
                    best_val = val
                    best_h = h.copy()
                    improved = True
            
            if not improved:
                break
        
        return best_h, best_val
    
    # Run quick local search from initial random start
    h_init, val_init = _greedy_local_search(h_init, min(deadline - 60.0, start_time + 20.0))
    
    # Then use simulated annealing from this point and other random starts
    if time.time() < deadline - 10.0:
        h_best, val_best = _multi_start_sa(deadline)
    else:
        h_best = h_init
        val_best = val_init
    
    return h_best
# EVOLVE-BLOCK-END
