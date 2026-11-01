"""Optimized program for third autocorrelation inequality task."""
import numpy as np
import time

N = 400
_DEADLINE = 150.0
_START_TIME = None

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _compute_c3_bound(x):
    """Compute the upper bound on C3 from the signed step function.
    
    The autoconvolution of x with itself gives the autocorrelation.
    C3 bound is typically related to the peak sidelobe or a specific norm.
    For a signed step function, we consider the discrete autoconvolution.
    """
    # Compute autoconvolution (linear, not circular)
    # For efficiency, use FFT-based convolution
    n = len(x)
    # Zero-pad for linear convolution
    size = 2 * n - 1
    f = np.fft.fft(x, size=size)
    ac = np.fft.ifft(f * np.conj(f)).real
    # The main lobe is at the center
    # Sidelobes are everything else
    center = n - 1
    # Exclude the main lobe (center element)
    sidelobes = np.concatenate([ac[:center], ac[center+1:]])
    # The upper bound on C3 is related to the max absolute sidelobe
    # relative to the main lobe, or an L-infinity norm
    main_lobe = ac[center]
    if main_lobe == 0:
        return np.max(np.abs(sidelobes))
    # Return the peak sidelobe level as the bound
    return np.max(np.abs(sidelobes)) / abs(main_lobe)

def _flip_and_evaluate(x, i):
    """Flip sign at position i and return new C3 bound."""
    x_new = x.copy()
    x_new[i] = -x_new[i]
    return _compute_c3_bound(x_new)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Initialize with alternating signs (good initial autocorrelation properties)
    x = np.ones(N)
    x[1::2] = -1.0
    
    best_x = x.copy()
    best_score = _compute_c3_bound(best_x)
    
    deadline = _DEADLINE - 5.0  # 5 second safety margin
    
    # Local search: try flipping individual bits
    improved = True
    while improved and _get_elapsed() < deadline:
        improved = False
        for i in range(N):
            if _get_elapsed() > deadline:
                break
            
            # Try flipping bit i
            x[i] = -x[i]
            new_score = _compute_c3_bound(x)
            
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
                improved = True
            else:
                x[i] = -x[i]  # revert
    
    # If we have time, try a few random restarts with local search
    while _get_elapsed() < deadline - 10.0:
        # Random restart
        x = np.random.choice([-1.0, 1.0], size=N)
        local_best = x.copy()
        local_score = _compute_c3_bound(local_best)
        
        # Quick local search on this restart
        for _ in range(3):  # a few passes
            if _get_elapsed() > deadline - 10.0:
                break
            for i in range(N):
                if _get_elapsed() > deadline - 10.0:
                    break
                x[i] = -x[i]
                new_score = _compute_c3_bound(x)
                if new_score < local_score:
                    local_score = new_score
                    local_best = x.copy()
                else:
                    x[i] = -x[i]
        
        if local_score < best_score:
            best_score = local_score
            best_x = local_best.copy()
    
    return best_x

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
