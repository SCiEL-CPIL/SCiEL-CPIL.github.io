"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_autocorr(f):
    """Compute linear autocorrelation of f."""
    # f is array of length N with values ±1
    # autocorr[k] = sum_{i=0}^{N-1-k} f[i] * f[i+k] for k >= 0
    # Use FFT for efficiency
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)
    # acf[0:N] gives positive lags, acf[n_fft-N+1:] gives negative lags
    # For real signal, acf is symmetric
    return acf[:N]  # lags 0 to N-1

def _peak_sidelobe(f):
    """Return peak absolute autocorrelation sidelobe (excluding lag 0)."""
    acf = _compute_autocorr(f)
    # acf[0] is the main lobe, look at acf[1:]
    return np.max(np.abs(acf[1:]))

def _score(f):
    """The raw value to minimize: peak sidelobe of autocorrelation."""
    return _peak_sidelobe(f)

def construct():
    deadline = time.time() + 140  # safety margin
    
    # Start with random ±1
    np.random.seed(42)
    f = np.random.choice([-1.0, 1.0], size=N)
    
    best_f = f.copy()
    best_score = _score(best_f)
    
    # Local search: flip individual bits
    n_improve = 0
    max_iters = 500000
    iter_count = 0
    
    while iter_count < max_iters:
        if time.time() > deadline:
            break
        iter_count += 1
        
        # Try flipping a random position
        idx = np.random.randint(N)
        old_val = f[idx]
        f[idx] = -f[idx]
        
        new_score = _score(f)
        
        if new_score < best_score - 1e-12:
            best_score = new_score
            best_f = f.copy()
            n_improve += 1
        else:
            f[idx] = old_val  # revert
        
        # Occasionally try multiple flips
        if iter_count % 1000 == 0:
            # Try 2-bit flip
            i1 = np.random.randint(N)
            i2 = np.random.randint(N)
            while i2 == i1:
                i2 = np.random.randint(N)
            v1, v2 = f[i1], f[i2]
            f[i1] = -f[i1]
            f[i2] = -f[i2]
            new_score = _score(f)
            if new_score < best_score - 1e-12:
                best_score = new_score
                best_f = f.copy()
            else:
                f[i1] = v1
                f[i2] = v2
    
    return best_f
# EVOLVE-BLOCK-END
