import numpy as np
import time

N = 400
DEADLINE = 140.0  # Safety margin before 150s kill

def _autocorr(h):
    """Compute full autocorrelation of h."""
    n = len(h)
    # Use FFT-based autocorrelation
    f = np.fft.fft(h, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    return acf

def _score(h):
    """Compute the upper bound on C3: peak sidelobe / main lobe."""
    acf = _autocorr(h)
    main = acf[0]
    if main <= 0:
        return np.inf
    sidelobe = np.max(np.abs(acf[1:]))
    return sidelobe / main

def construct():
    start = time.time()
    
    # Initial: random signed signal
    np.random.seed(42)
    best_h = np.random.choice([-1.0, 1.0], size=N)
    best_score = _score(best_h)
    
    # Coordinate descent: flip individual bits
    improved = True
    while improved and (time.time() - start) < DEADLINE:
        improved = False
        for i in range(N):
            if (time.time() - start) > DEADLINE:
                break
            # Try flipping bit i
            old_val = best_h[i]
            best_h[i] = -old_val
            new_score = _score(best_h)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best_h[i] = old_val
    
    # If time remains, try block flips of size 2-4
    if (time.time() - start) < DEADLINE:
        for block_size in [2, 3, 4]:
            if (time.time() - start) > DEADLINE:
                break
            for i in range(0, N - block_size + 1):
                if (time.time() - start) > DEADLINE:
                    break
                old_block = best_h[i:i+block_size].copy()
                best_h[i:i+block_size] = -old_block
                new_score = _score(best_h)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_h[i:i+block_size] = old_block
    
    # Return the best heights found
    return best_h
