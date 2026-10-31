import numpy as np
import time

def _autoconv_fft(f):
    """Compute full autoconvolution via FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(f, size)
    C = np.fft.irfft(F * np.conj(F), size)
    return C[:2*n-1]

def _objective(f):
    """Compute max|C_k| / S^2."""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return np.inf
    C = _autoconv_fft(f)
    return np.max(np.abs(C)) / (S * S)

def _autoconv_full(f):
    """Full autoconvolution (all lags from -(N-1) to N-1)."""
    n = len(f)
    C = _autoconv_fft(f)
    # C has lags 0..2n-2, but autoconv is symmetric: C[k] for k=0..n-1 is lag k
    # For full: lags -(n-1)..(n-1), use symmetry
    return C

def construct():
    deadline = time.time() + 115.0  # 115s safety margin
    rng = np.random.default_rng(42)
    
    best_f = None
    best_obj = np.inf
    
    # Try multiple N values and random seeds
    for N in [2000, 5000, 10000, 20000, 50000]:
        if time.time() > deadline - 5.0:
            break
        
        for seed in range(3):
            if time.time() > deadline - 5.0:
                break
            
            f = rng.choice([-1.0, 1.0], size=N)
            
            # Ensure S is not too small; if |S| < sqrt(N), flip some signs
            S = np.sum(f)
            if abs(S) < np.sqrt(N):
                # Flip signs of elements to increase S
                idx = np.where(f < 0)[0]
                if len(idx) > 0:
                    # Flip a subset to boost S
                    flip_count = min(len(idx), N // 2)
                    flip_idx = rng.choice(idx, size=flip_count, replace=False)
                    f[flip_idx] *= -1
            
            obj = _objective(f)
            if obj < best_obj:
                best_obj = obj
                best_f = f.copy()
            
            # Local sign-flip optimization
            if time.time() > deadline - 3.0:
                break
            
            improved = True
            while improved and time.time() < deadline - 3.0:
                improved = False
                C = _autoconv_fft(f)
                max_abs_C = np.max(np.abs(C))
                S = np.sum(f)
                S2 = S * S
                
                # Try flipping each sign (batched for speed)
                # For each position i, compute change in objective if f[i] -> -f[i]
                # This changes C_k for all k involving position i
                # Efficient: precompute contribution
                
                # Simpler: random subset of flips
                n_tries = min(N, 5000)
                for _ in range(n_tries):
                    if time.time() > deadline - 3.0:
                        break
                    
                    i = rng.integers(0, N)
                    f[i] *= -1
                    new_obj = _objective(f)
                    if new_obj < obj:
                        obj = new_obj
                        improved = True
                        if obj < best_obj:
                            best_obj = obj
                            best_f = f.copy()
                    else:
                        f[i] *= -1  # revert
            
            # Magnitude optimization: scale to optimal
            # For fixed signs, optimal magnitudes minimize max|C|/S^2
            # This is non-trivial; try simple scaling
            # Actually, for ±1 sequences, scaling doesn't help since ratio is scale-invariant
            # Wait: ratio = max|C|/S^2. If we scale f by c, C scales by c^2, S scales by c,
            # so ratio = c^2 * max|C| / (c^2 * S^2) = max|C|/S^2. Scale invariant!
            # So ±1 is fine.
    
    if best_f is None:
        # Fallback: constant vector
        N = 10000
        best_f = np.ones(N)
    
    return best_f
