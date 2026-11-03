import numpy as np
import time

N = 400

def _autoconv_lin(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    conv = np.fft.irfft(X * X.conj(), size)
    return conv[:2 * n - 1]

def _metric_offdiag(x):
    """Max |c_k| for k != 0, divided by (sum x)^2."""
    c = _autoconv_lin(x)
    n = len(x)
    # Off-diagonal coefficients are c[1:n] and c[n+1:2*n-1]
    # Note: c[n] is the zero-lag (peak) if we consider the full linear conv length 2N-1
    # Wait, linear conv of length N with itself has length 2N-1.
    # Indices: 0 to 2N-2.
    # Zero lag is at index N-1.
    # So off-diagonal are [0, N-2] and [N, 2N-2].
    off1 = c[:n-1]
    off2 = c[n:]
    off = np.concatenate([off1, off2])
    max_abs = np.max(np.abs(off)) if len(off) > 0 else 0.0
    s = np.sum(x)
    if abs(s) < 1e-12:
        return max_abs
    return max_abs / (s * s)

def construct():
    deadline = time.time() + 110.0  # Safety margin
    
    # Initial best
    # Start with a random vector
    rng = np.random.default_rng(42)
    best_x = rng.choice([-1, 1], size=N)
    best_score = _metric_offdiag(best_x)
    
    # Simulated Annealing parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.9995
    
    # We will run multiple independent SA runs
    # Each run will have a budget of time
    # Let's say we allocate 50% of time to SA, 50% to local polishing
    
    # Actually, let's just run SA until time runs out, keeping the global best.
    
    current_x = best_x.copy()
    current_score = best_score
    T = T_start
    
    # Precompute indices for faster access? No, just use numpy.
    
    while time.time() < deadline - 5.0:
        # Random move: flip one bit
        i = rng.integers(0, N)
        current_x[i] *= -1
        
        new_score = _metric_offdiag(current_x)
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = current_x.copy()
        else:
            # Revert move
            current_x[i] *= -1
            
        # Cool down
        T *= cooling_rate
        if T < T_end:
            # Reset temperature to restart the search from the current best?
            # Or just stop cooling and keep flipping?
            # Let's restart the temperature to allow larger jumps again
            T = T_start * 0.5
            
    # Final local search polish on the best solution found
    remaining = deadline - time.time()
    if remaining > 1.0:
        cur_x = best_x.copy()
        cur_score = best_score
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            # Try flipping each bit once
            for i in range(N):
                if time.time() > deadline - 0.5:
                    break
                cur_x[i] *= -1
                new_score = _metric_offdiag(cur_x)
                if new_score < cur_score:
                    cur_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_x = cur_x.copy()
                else:
                    cur_x[i] *= -1

    return best_x.astype(float)
