"""Rewritten program for signed autocorrelation maximum absolute-value variant."""
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
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_x = np.ones(N)
    best_score = _score(best_x)

    def try_candidate(x):
        nonlocal best_x, best_score
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()

    rng = np.random.default_rng(12345)

    # Phase 1: Deterministic structured candidates
    # Alternating
    alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    try_candidate(alt)

    # Periodic patterns with various structures
    for period in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 20, 21, 24, 25, 28, 30, 32, 35, 40, 42, 48, 50, 56, 60, 64, 70, 72, 80, 84, 90, 96, 100, 105, 112, 120, 128, 140, 144, 160, 168, 180, 192, 200, 210, 224, 240, 252, 280, 288, 320, 336, 360, 384, 400]:
        if time.time() > deadline:
            break
        for num_neg in range(0, period):
            if time.time() > deadline:
                break
            pattern = np.ones(period)
            if num_neg > 0:
                # Try different placements of negatives
                # Evenly spaced
                neg_pos = np.linspace(0, period - 1, num_neg, dtype=int)
                neg_pos = np.array(sorted(set(neg_pos.tolist())))[:num_neg]
                pattern[neg_pos] = -1.0
            x = np.tile(pattern, N // period + 1)[:N]
            try_candidate(x)

            # Also try clustered negatives
            if num_neg > 0 and num_neg < period:
                pattern2 = np.ones(period)
                pattern2[:num_neg] = -1.0
                x2 = np.tile(pattern2, N // period + 1)[:N]
                try_candidate(x2)

    # Phase 2: Binary sequences from known good constructions
    # Thue-Morse sequence
    def thue_morse(n):
        tm = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            tm[i] = tm[i >> 1] ^ (i & 1)
        return 1.0 - 2.0 * tm  # Map 0->1, 1->-1

    try_candidate(thue_morse(N))

    # Rudin-Shapiro sequence
    def rudin_shapiro(n):
        rs = np.zeros(n, dtype=np.float64)
        rs[0] = 1.0
        if n > 1:
            rs[1] = 1.0
        for i in range(2, n):
            # Rudin-Shapiro: a(n) = a(floor(n/2)) if n mod 4 is 0 or 1, -a(floor(n/2)) if n mod 4 is 2 or 3
            # Actually: a(4k)=a(2k), a(4k+1)=a(2k), a(4k+2)=-a(2k+1), a(4k+3)=-a(2k+1)
            # Let me use the standard definition
            # a(n) where a(0)=a(1)=1, a(2n)=a(n), a(2n+1)=-a(n)
            # Wait, that's not right either. Let me use:
            # Rudin-Shapiro: a(0)=1, a(1)=1, a(2n)=a(n), a(2n+1)=-a(n)
            half = i // 2
            if i % 2 == 0:
                rs[i] = rs[half]
            else:
                rs[i] = -rs[half]
        return rs

    try_candidate(rudin_shapiro(N))

    # Phase 3: Simulated Annealing
    # Start from best candidate
    x = best_x.copy()
    sc = best_score

    # Annealing schedule
    T_start = 0.05
    T_end = 0.001
    n_iterations = 50000

    # We'll do multiple passes
    for pass_num in range(3):
        if time.time() > deadline - 10:
            break

        T = T_start
        cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)

        for i in range(n_iterations):
            if time.time() > deadline - 5:
                break

            # Flip a random subset of bits (1-3 bits)
            n_flips = rng.integers(1, 4)
            flip_idx = rng.choice(N, size=n_flips, replace=False)
            x_old = x[flip_idx].copy()
            x[flip_idx] *= -1.0

            sc_new = _score(x)
            delta = sc_new - sc

            if delta < 0:
                # Accept
                sc = sc_new
                if sc < best_score:
                    best_score = sc
                    best_x = x.copy()
            else:
                # Maybe accept
                if T > 1e-10:
                    accept_prob = np.exp(-delta / T)
                    if rng.random() < accept_prob:
                        sc = sc_new
                    else:
                        x[flip_idx] = x_old

            T *= cooling_rate

    # Phase 4: Intensive local search from best
    if time.time() < deadline - 15:
        x = best_x.copy()
        sc = best_score

        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            improved = False

            # Try flipping each bit
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                x[i] *= -1.0
                sc_new = _score(x)
                if sc_new < sc - 1e-12:
                    sc = sc_new
                    best_x = x.copy()
                    best_score = sc
                    improved = True
                else:
                    x[i] *= -1.0  # revert

            if not improved:
                break

    # Phase 5: Try random restarts with local search
    if time.time() < deadline - 10:
        for restart in range(20):
            if time.time() > deadline - 5:
                break

            # Start from a random point
            x = rng.choice([-1.0, 1.0], size=N)
            sc = _score(x)
            if sc < best_score:
                best_score = sc
                best_x = x.copy()
                sc = best_score

            # Quick local search
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                improved = False
                for _ in range(50):
                    if time.time() > deadline - 5:
                        break
                    i = rng.integers(0, N)
                    x[i] *= -1.0
                    sc_new = _score(x)
                    if sc_new < sc - 1e-12:
                        sc = sc_new
                        if sc < best_score:
                            best_score = sc
                            best_x = x.copy()
                        improved = True
                    else:
                        x[i] *= -1.0
                if not improved:
                    break

    # Phase 6: Final intensive local search
    if time.time() < deadline - 3:
        x = best_x.copy()
        sc = best_score

        for iteration in range(30):
            if time.time() > deadline - 1:
                break
            improved = False

            # Try flipping pairs of bits
            for i in range(0, N, 2):
                if time.time() > deadline - 1:
                    break
                if i + 1 >= N:
                    break
                x[i] *= -1.0
                x[i+1] *= -1.0
                sc_new = _score(x)
                if sc_new < sc - 1e-12:
                    sc = sc_new
                    best_x = x.copy()
                    best_score = sc
                    improved = True
                else:
                    x[i] *= -1.0
                    x[i+1] *= -1.0

            if not improved:
                break

    return best_x
# EVOLVE-BLOCK-END
