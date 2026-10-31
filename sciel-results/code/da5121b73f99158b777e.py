import numpy as np
import time

N = 400

def autoconv(f):
    """Compute full autoconvolution."""
    n = len(f)
    # Use FFT for efficiency
    size = 2 * n
    F = np.fft.fft(f, size)
    ac = np.fft.ifft(F * F).real
    return ac[:n]  # Only the first n lags (0 to n-1)

def score(f):
    """Compute the score: max|autoconv| / (sum f)^2."""
    ac = autoconv(f)
    s = np.sum(f)
    if abs(s) < 1e-12:
        return float('inf')
    return np.max(np.abs(ac)) / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_f = np.ones(N)
    best_score = score(best_f)
    
    # Try various parametric families
    
    # Family 1: alternating with parameter a
    # f[j] = 1 if j even, -a if j odd
    for a in np.linspace(0, 2, 50):
        f = np.ones(N)
        f[1::2] = -a
        s = score(f)
        if s < best_score:
            best_score = s
            best_f = f.copy()
    
    # Family 2: two-level with different ratio
    # First half = 1, second half = -a
    for a in np.linspace(0, 2, 50):
        f = np.ones(N)
        f[N//2:] = -a
        s = score(f)
        if s < best_score:
            best_score = s
            best_f = f.copy()
    
    # Family 3: sinusoidal modulation
    for freq in np.linspace(0.01, 0.5, 30):
        for amp in np.linspace(0, 0.9, 20):
            f = 1 - amp * np.sin(2 * np.pi * freq * np.arange(N))
            s = score(f)
            if s < best_score:
                best_score = s
                best_f = f.copy()
    
    # Family 4: random search with local improvement
    rng = np.random.default_rng(42)
    for _ in range(200):
        if time.time() > deadline:
            break
        f = rng.choice([-1, 1], size=N)
        # Add a small bias to make sum non-zero
        f[0] += 0.1
        s = score(f)
        if s < best_score:
            best_score = s
            best_f = f.copy()
    
    # Local optimization on best found
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            old_val = best_f[i]
            for delta in [-0.1, -0.01, 0.01, 0.1]:
                new_val = old_val + delta
                test_f = best_f.copy()
                test_f[i] = new_val
                s = score(test_f)
                if s < best_score - 1e-12:
                    best_f = test_f
                    best_score = s
                    improved = True
                    break
        if not improved:
            break
    
    return best_f
