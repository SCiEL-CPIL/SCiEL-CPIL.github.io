"""Program for third autocorrelation inequality – signed step function on 400 intervals."""
import numpy as np
import math
import random
import time

N = 400

# EVOLVE-BLOCK-START

def _compute_c3(s):
    """Return (max_abs_C3, C3_array) for a sign sequence s of length N."""
    # Convolution of three copies has length 3N-2; pad to next power of 2
    total_len = 3 * N - 2
    n_pad = 1 << (total_len - 1).bit_length()
    s_pad = np.zeros(n_pad, dtype=np.float64)
    s_pad[:N] = s
    S = np.fft.fft(s_pad)
    C3 = np.fft.ifft(S ** 3).real
    # The meaningful part is indices 0..3N-3
    C3 = C3[:3 * N - 2]
    return np.max(np.abs(C3)), C3


def _logistic_signs(seed, n=N):
    """Generate n signs from logistic map x -> 4x(1-x), sign = +1 if x>0.5 else -1."""
    x = seed % 1.0
    if x <= 0.0 or x >= 1.0:
        x = 0.3
    signs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        x = 4.0 * x * (1.0 - x)
        signs[i] = 1.0 if x > 0.5 else -1.0
    return signs


def _tent_signs(seed, n=N):
    """Generate n signs from tent map x -> 1 - |2x - 1|, sign = +1 if x>0.5 else -1."""
    x = seed % 1.0
    if x <= 0.0 or x >= 1.0:
        x = 0.3
    signs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        x = 1.0 - abs(2.0 * x - 1.0)
        signs[i] = 1.0 if x > 0.5 else -1.0
    return signs


def _circle_map_signs(seed, n=N, K=0.31):
    """Generate n signs from circle map x -> x + K + (1/2pi)*sin(2pi*x), sign by fractional part."""
    x = seed % 1.0
    signs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        x = (x + K + (1.0 / (2.0 * math.pi)) * math.sin(2.0 * math.pi * x)) % 1.0
        signs[i] = 1.0 if x > 0.5 else -1.0
    return signs


def _sa_optimize(signs, max_time=60.0, seed=42):
    """Simulated annealing with multi-point flips to minimize max|C3|."""
    rng = random.Random(seed)
    current = signs.copy()
    current_bound, _ = _compute_c3(current)

    T_start = 0.5
    T_end = 0.0005
    alpha = 0.9995
    T = T_start
    max_flip = 3  # flip up to 3 bits at once

    t0 = time.time()
    while T > T_end and (time.time() - t0) < max_time:
        # Choose number of bits to flip: 1, 2, or 3
        k = rng.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
        indices = rng.sample(range(N), k)

        # Flip
        for idx in indices:
            current[idx] *= -1.0

        new_bound, _ = _compute_c3(current)
        delta = new_bound - current_bound

        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_bound = new_bound
        else:
            for idx in indices:
                current[idx] *= -1.0

        T *= alpha

    return current, current_bound


def _peak_attack(signs, current_bound, max_time=15.0):
    """Targeted search: find peak lag, try flips near contributing positions."""
    t0 = time.time()
    rng = random.Random(123)
    improved = True
    while improved and (time.time() - t0) < max_time:
        improved = False
        _, C3 = _compute_c3(signs)
        # Find the lag with maximum |C3|
        peak_idx = int(np.argmax(np.abs(C3)))
        peak_val = abs(C3[peak_idx])
        if peak_val <= current_bound:
            break

        # The C3 at lag L = sum over i,j,k: s[i]*s[j]*s[k] where i+j+k = L (mod N) approximately
        # For a step function on a circle, the autocorrelation at lag L involves products
        # Try flipping positions in a window around positions that contribute to this lag
        # Simple heuristic: try flipping each position and see if it helps
        for i in range(N):
            signs[i] *= -1.0
            new_bound, _ = _compute_c3(signs)
            if new_bound < current_bound:
                current_bound = new_bound
                improved = True
            else:
                signs[i] *= -1.0

    return signs, current_bound


def _greedy_polish(signs, current_bound, max_passes=10):
    """Greedy single-flip polish."""
    for _ in range(max_passes):
        improved = False
        for i in range(N):
            signs[i] *= -1.0
            new_bound, _ = _compute_c3(signs)
            if new_bound < current_bound:
                current_bound = new_bound
                improved = True
            else:
                signs[i] *= -1.0
        if not improved:
            break
    return signs, current_bound


def construct():
    """
    Construct a signed step function with N=400 equal intervals on [-1/4, 1/4].
    Returns a sequence of 400 signed step heights (+1 or -1).
    Score is the upper bound on C3 (lower is better).
    """
    t_start = time.time()
    time_budget = 110.0  # leave 10s margin

    # ---- Generate diverse initial candidates ----
    candidates = []

    # Irrational seeds for chaotic maps
    seeds = [
        math.sqrt(2) % 1.0,
        math.sqrt(3) % 1.0,
        math.pi % 1.0,
        (1 + math.sqrt(5)) / 2 % 1.0,
        math.e % 1.0,
        math.sqrt(5) % 1.0,
        math.sqrt(7) % 1.0,
        math.sqrt(13) % 1.0,
        0.123456789,
        0.987654321,
        0.5 + (1 / math.sqrt(2)) % 1.0,
        math.log(2) % 1.0,
        math.log(3) % 1.0,
        math.atan(1) % 1.0,
    ]

    for s in seeds:
        candidates.append(_logistic_signs(s))
        candidates.append(_tent_signs(s))
        candidates.append(_circle_map_signs(s, K=0.31))
        candidates.append(_circle_map_signs(s, K=0.27))

    # A few random initializations
    rng_init = np.random.RandomState(42)
    for _ in range(4):
        candidates.append(rng_init.choice([-1.0, 1.0], size=N))

    # Evaluate all candidates, keep top 6
    scored = []
    for c in candidates:
        b, _ = _compute_c3(c)
        scored.append((b, c))
    scored.sort(key=lambda x: x[0])
    top_candidates = [c for _, c in scored[:6]]
    top_bounds = [b for b, _ in scored[:6]]

    # ---- Run SA from each top candidate, with time split ----
    n_sa = len(top_candidates)
    sa_time_each = min(30.0, (time_budget - 20.0) / n_sa)  # reserve 20s for peak attack + polish

    best_signs = None
    best_bound = float('inf')

    for idx, (cand, cand_bound) in enumerate(zip(top_candidates, top_bounds)):
        if time.time() - t_start > time_budget - 25.0:
            break
        sa_time = sa_time_each
        signs, bound = _sa_optimize(cand, max_time=sa_time, seed=100 + idx)
        if bound < best_bound:
            best_bound = bound
            best_signs = signs.copy()

    # ---- Peak attack on the best ----
    if time.time() - t_start < time_budget - 10.0:
        best_signs, best_bound = _peak_attack(best_signs, best_bound,
                                               max_time=min(15.0, time_budget - (time.time() - t_start) - 10.0))

    # ---- Greedy polish ----
    if time.time() - t_start < time_budget - 5.0:
        best_signs, best_bound = _greedy_polish(best_signs, best_bound, max_passes=5)

    return best_signs

# EVOLVE-BLOCK-END
