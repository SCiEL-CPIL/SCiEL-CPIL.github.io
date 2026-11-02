"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(f):
    """Compute max|autoconv| / (sum f)^2"""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return float('inf')
    # Autoconvolution via FFT
    n = len(f)
    fft_size = 1
    while fft_size < 2 * n - 1:
        fft_size *= 2
    F = np.fft.rfft(f, fft_size)
    autoconv = np.fft.irfft(F * np.conj(F), fft_size)[:n]
    max_abs = np.max(np.abs(autoconv))
    return max_abs / (S * S)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_f = np.ones(N)
    best_score = _score(best_f)
    
    # Try various initializations
    initializations = [
        np.ones(N),
        np.random.choice([-1, 1], size=N),
        np.sign(np.sin(np.linspace(0, 10*np.pi, N))),
        np.sign(np.cos(np.linspace(0, 5*np.pi, N))),
    ]
    
    for init in initializations:
        s = _score(init)
        if s < best_score:
            best_score = s
            best_f = init.copy()
    
    # Local search: flip individual elements
    f = best_f.copy()
    current_score = best_score
    step_count = 0
    
    while time.time() < deadline:
        # Try flipping a random position
        i = np.random.randint(N)
        old_val = f[i]
        f[i] = -f[i]
        new_score = _score(f)
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_f = f.copy()
        else:
            f[i] = old_val
        step_count += 1
    
    return best_f
# EVOLVE-BLOCK-END
