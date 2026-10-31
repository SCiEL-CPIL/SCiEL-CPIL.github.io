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

    # Phase 1: Deterministic candidates
    # All ones (already done)
    # Alternating
    alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    try_candidate(alt)

    # Periodic blocks of various lengths
    for period in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 16, 20, 25, 30, 40, 50, 60, 80, 100, 125, 133, 160, 200]:
        if time.time() > deadline:
            break
        for num_neg in range(0, period):
            if time.time() > deadline:
                break
            # Create one period pattern with num_neg negative entries
            pattern = np.ones(period)
            if num_neg > 0:
                # Place negatives at evenly spaced positions
                neg_pos = np.linspace(0, period - 1, num_neg, dtype=int)
                if num_neg > 1:
                    # Avoid duplicates
                    neg_pos = np.array(sorted(set(neg_pos.tolist())))
                    if len(neg_pos) > num_neg:
                        neg_pos = neg_pos[:num_neg]
                pattern[neg_pos] = -1.0
            # Repeat pattern
            repeats = N // period + 1
            x = np.tile(pattern, repeats)[:N]
            try_candidate(x)

    # Zadoff-Chu sequences (from inspiration 2)
    for u in [1, 2, 3, 4, 5, 7, 11, 13, 17, 19]:
        if time.time() > deadline:
            break
        n = np.arange(N)
        zc = np.exp(-1j * np.pi * u * n * (n + 1) / N)
        x = np.where(zc.real > 0, 1.0, -1.0)
        try_candidate(x)

    # Phase 2: Random ±1 search
    rng = np.random.default_rng(42)
    num_random = 5000
    for i in range(num_random):
        if time.time() > deadline:
            break
        x = rng.choice([-1.0, 1.0], size=N)
        try_candidate(x)

    # Phase 3: Local optimization (bit flips)
    # Start from best candidate and try flipping bits
    x = best_x.copy()
    sc = best_score
    # Multiple passes of local search
    for iteration in range(20):
        if time.time() > deadline:
            break
        improved = False
        # Try flipping random subsets of bits
        for _ in range(100):
            if time.time() > deadline:
                break
            # Flip a small random set of bits
            n_flips = rng.integers(1, 5)
            flip_idx = rng.choice(N, size=n_flips, replace=False)
            x_new = x.copy()
            x_new[flip_idx] *= -1.0
            sc_new = _score(x_new)
            if sc_new < sc - 1e-10:
                x = x_new
                sc = sc_new
                improved = True
                best_x = x.copy()
                best_score = sc
        if not improved:
            break

    # Phase 4: Smooth random patterns (Gaussian noise, thresholded)
    for i in range(2000):
        if time.time() > deadline:
            break
        noise = rng.normal(0, 1, N)
        # Threshold at various levels
        for thresh in [0, 0.25, 0.5, 0.75]:
            if time.time() > deadline:
                break
            x = np.where(noise > thresh, 1.0, -1.0)
            try_candidate(x)

    # Phase 5: Try structured low-discrepancy-like patterns
    # Van der Corput / Halton sequence based patterns
    for base in [2, 3, 4, 5]:
        if time.time() > deadline:
            break
        x = np.zeros(N)
        for i in range(N):
            # Compute van der Corput number
            n = i + 1
            result = 0.0
            f = 1.0 / base
            while n > 0:
                result += f * (n % base)
                n //= base
                f /= base
            x[i] = 1.0 if result < 0.5 else -1.0
        try_candidate(x)

    # Phase 6: Simulated Annealing (from inspiration 2)
    if time.time() < deadline - 30.0:
        x_sa = best_x.copy()
        current_score = best_score
        temp = 0.1
        cooling = 0.995
        
        # Pre-compute FFT size
        size = 2 * N - 1
        size2 = 1
        while size2 < size:
            size2 *= 2

        # Helper to quickly compute score from FFT
        def compute_score_from_fft(F):
            A = ifft(F * np.conj(F)).real[:size]
            s = np.sum(np.real(ifft(F, size2))[:N])  # This is wrong, need to keep x
            # Actually, let's just recompute from x for correctness in SA
            return None

        # Standard SA loop
        for step in range(50000):
            if time.time() > deadline - 5.0:
                break
            if temp < 1e-4:
                break
            
            # Random move: flip 1-3 bits
            n_flips = rng.integers(1, 4)
            flip_idx = rng.choice(N, size=n_flips, replace=False)
            x_sa[flip_idx] *= -1.0
            
            new_score = _score(x_sa)
            
            # Acceptance criterion
            if new_score < current_score or (np.random.random() < np.exp(-(new_score - current_score) / temp)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_x = x_sa.copy()
            else:
                # Revert
                x_sa[flip_idx] *= -1.0
            
            temp *= cooling

    # Final local optimization pass on best
    if time.time() < deadline - 5.0:
        x = best_x.copy()
        sc = best_score
        for iteration in range(30):
            if time.time() > deadline - 2.0:
                break
            improved = False
            # Systematic scan: try flipping each bit
            for i in range(0, N, 1):
                if time.time() > deadline - 2.0:
                    break
                x[i] *= -1.0
                sc_new = _score(x)
                if sc_new < sc - 1e-10:
                    sc = sc_new
                    best_x = x.copy()
                    best_score = sc
                    improved = True
                else:
                    x[i] *= -1.0  # revert
            if not improved:
                break

    return best_x
# EVOLVE-BLOCK-END
