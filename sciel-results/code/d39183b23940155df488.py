"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from numpy.fft import fft, ifft
import time


def _autoconv(x):
    """Compute linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    F = fft(x, size2)
    A = ifft(F * np.conj(F)).real
    return A[:size]


def _score(x):
    """Compute max(|autoconv|) / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    A = _autoconv(x)
    m = np.max(np.abs(A))
    return m / (s * s)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Helper: try a candidate and update best
    def try_candidate(x):
        nonlocal best_x, best_score
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
            
    rng = np.random.default_rng(42)

    # Phase 1: Sparse random signs (Bernoulli sampling)
    # Hypothesis: Sparse random signs have autoconvolution that decays faster 
    # relative to the squared integral, reducing the peak-to-average ratio.
    # We test various densities and amplitudes.
    
    densities = [0.05, 0.1, 0.2, 0.3, 0.5]
    amplitudes = [1.0, 2.0, 5.0, 10.0]
    
    for d in densities:
        for a in amplitudes:
            if time.time() > deadline:
                break
            # Try multiple random instances of this structure
            for _ in range(100):
                if time.time() > deadline:
                    break
                mask = rng.random(N) < d
                signs = rng.choice([-1, 1], size=N)
                x = np.zeros(N)
                x[mask] = signs[mask] * a
                try_candidate(x)
                
                # Also try with positive bias to ensure non-zero sum
                # If sum is near zero, score is inf, so we need to balance signs or add a constant
                # Let's try adding a small constant to ensure sum != 0
                x = x + 0.1 * a
                try_candidate(x)
                
    # Phase 2: Structured sparse patterns (e.g., periodic sparse)
    for period in range(2, 20):
        if time.time() > deadline:
            break
        for offset in range(period):
            if time.time() > deadline:
                break
            x = np.zeros(N)
            x[offset::period] = 1.0
            # Add some random noise
            x += 0.1 * rng.normal(0, 1, N)
            try_candidate(x)
            
            # Invert signs randomly
            signs = rng.choice([-1, 1], size=N)
            x = np.zeros(N)
            x[offset::period] = signs[offset::period]
            x += 0.1
            try_candidate(x)

    # Phase 3: Local search on the best sparse candidate
    # We can refine the sparse structure by adjusting amplitudes or adding/removing entries
    if best_score < _score(np.ones(N)) * 0.99:
        x = best_x.copy()
        sc = best_score
        for iteration in range(50):
            if time.time() > deadline:
                break
            improved = False
            # Try perturbing random subsets
            for _ in range(200):
                if time.time() > deadline:
                    break
                n_perturb = rng.integers(1, 10)
                perturb_idx = rng.choice(N, size=n_perturb, replace=False)
                x_new = x.copy()
                # Randomly change values in perturb_idx
                x_new[perturb_idx] = rng.normal(0, 1, n_perturb)
                sc_new = _score(x_new)
                if sc_new < sc - 1e-10:
                    x = x_new
                    sc = sc_new
                    improved = True
                    best_x = x.copy()
                    best_score = sc
            if not improved:
                break

    # Phase 4: Systematic local search on best
    if time.time() < deadline - 10.0:
        x = best_x.copy()
        sc = best_score
        # Try flipping/perturbing each position
        for iteration in range(10):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                # Try small perturbation
                orig_val = x[i]
                for delta in [0.01, -0.01, 0.1, -0.1]:
                    x[i] = orig_val + delta
                    sc_new = _score(x)
                    if sc_new < sc - 1e-10:
                        sc = sc_new
                        best_x = x.copy()
                        best_score = sc
                        improved = True
                        break
                else:
                    x[i] = orig_val  # revert if no improvement
            if not improved:
                break

    return best_x
# EVOLVE-BLOCK-END
