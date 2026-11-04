"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_peak(h):
    """Compute the max absolute autocorrelation at non-zero lags."""
    n = len(h)
    # Full autocorrelation via FFT
    fft_size = 2 * n
    H = np.fft.fft(h, n=fft_size)
    acf = np.fft.ifft(np.abs(H)**2).real
    # acf[0] is the DC term (sum of squares), we want non-zero lags
    # The autocorrelation at lag k is acf[k] for k>0 and acf[fft_size-k] for k<0
    max_val = 0.0
    for k in range(1, n):
        val = abs(acf[k])
        if val > max_val:
            max_val = val
    return max_val


def _score(h):
    """Score = upper bound on C3 (lower is better)."""
    return _autocorr_peak(h)


def construct():
    """
    Construct a sparse signed step function with N=400 intervals
    that minimizes the autoconvolution upper bound (C3).
    
    Strategy: Select a small subset of intervals to be non-zero (±1),
    and optimize which positions are active.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_h = np.zeros(N)
    best_score = np.inf
    
    # Try different sparsity levels
    for sparsity in [20, 50, 100, 150, 200]:
        if time.time() > deadline:
            break
            
        # Random initialization: choose sparsity positions
        rng = np.random.default_rng(42)
        for trial in range(50):
            if time.time() > deadline:
                break
                
            # Select active positions
            active_idx = rng.choice(N, size=sparsity, replace=False)
            h = np.zeros(N)
            h[active_idx] = rng.choice([-1, 1], size=sparsity)
            
            # Local search: try flipping individual bits or swapping positions
            current_score = _score(h)
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
            
            # Local optimization: try flipping signs and moving positions
            for iteration in range(200):
                if time.time() > deadline:
                    break
                    
                improved = False
                # Try flipping a random active position's sign
                if sparsity > 0:
                    flip_idx = active_idx[rng.integers(sparsity)]
                    old_val = h[flip_idx]
                    h[flip_idx] = -old_val
                    new_score = _score(h)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        h[flip_idx] = old_val
                
                # Try swapping an active position with an inactive one
                if sparsity < N:
                    active_set = set(active_idx.tolist())
                    inactive_idx = np.array([i for i in range(N) if i not in active_set])
                    if len(inactive_idx) > 0:
                        from_active = active_idx[rng.integers(sparsity)]
                        to_inactive = inactive_idx[rng.integers(len(inactive_idx))]
                        
                        old_val = h[from_active]
                        h[from_active] = 0
                        h[to_inactive] = old_val
                        
                        new_score = _score(h)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            active_idx = active_idx.copy()
                            active_idx[active_idx == from_active] = to_inactive
                        else:
                            h[from_active] = old_val
                            h[to_inactive] = 0
                
                if not improved:
                    break
            
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
