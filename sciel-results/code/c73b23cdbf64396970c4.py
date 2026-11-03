import time
import random
import math
import numpy as np

def compute_score(U):
    U = tuple(sorted(set(U)))
    if 0 not in U or len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def _fast_score(U):
    """FFT-based approximate score for large sets; falls back to exact for small."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    n = len(U)
    if n < 4 or max_u > 200000:
        return compute_score(U)
    # Sumset via autocorrelation of indicator vector (linear, non-cyclic)
    m = 2 * max_u + 1
    N = 1
    while N < 2 * m - 1:
        N <<= 1
    f = np.zeros(N, dtype=np.float64)
    f[U] = 1.0
    F = np.fft.rfft(f)
    auto = np.fft.irfft(F * np.conj(F))
    sumset_size = int(np.count_nonzero(auto[:m] > 0.5))
    # Difference set via cyclic autocorrelation (size 2*max_u+1)
    N2 = 1
    while N2 < 2 * m:
        N2 <<= 1
    F2 = np.fft.rfft(f, n=N2)
    diff_auto = np.fft.irfft(F2 * np.conj(F2))
    diffset_size = int(np.count_nonzero(diff_auto[:m] > 0.5))
    ratio = diffset_size / sumset_size
    if ratio <= 1.0:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def construct():
    start = time.time()
    deadline = start + 110  # 40s safety margin under 150s kill

    def time_ok():
        return time.time() < deadline

    best_U = [0, 1, 3]
    best_score = compute_score(best_U)

    def consider(U):
        nonlocal best_U, best_score
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
        return s

    # --- Structured families: large-base powers and b-1 families ---
    # For {0, b, b^2, ...}: ratio -> 2, exponent -> 1 + ln2/ln(2b^k)
    # For {0, b-1, (b-1)^2, ...}: similar
    for b in range(2, 2000):
        if not time_ok():
            break
        # powers of b
        for k in range(2, 12):
            if not time_ok():
                break
            U = [b**i for i in range(k)]
            s = _fast_score(U)
            if s > best_score:
                # Verify with exact score
                exact_s = compute_score(U)
                if exact_s > best_score:
                    best_score = exact_s
                    best_U = U[:]
        # (b-1) family
        for k in range(2, 12):
            if not time_ok():
                break
            U = [(b - 1) ** i for i in range(k)]
            s = _fast_score(U)
            if s > best_score:
                exact_s = compute_score(U)
                if exact_s > best_score:
                    best_score = exact_s
                    best_U = U[:]

    # Small structured seeds
    seeds = [[0,1,3], [0,1,3,7], [0,1,3,9], [0,1,3,7,15], [0,1,3,9,27]]
    for s in seeds:
        if not time_ok():
            break
        consider(s)

    # --- Greedy local search with rich candidate pool ---
    def greedy_grow(U):
        U = sorted(set(U))
        cur = compute_score(U)
        improved = True
        while improved and time_ok():
            improved = False
            mx = max(U)
            # Candidate pool: all integers up to a bound + structured large jumps
            bound = min(8000, mx * 3 + 200)
            cands = list(range(1, bound + 1))
            # Structured large jumps (to exploit large-base regime)
            for mult in (2, 3, 4, 5, 7, 10, 16, 32):
                cands.append(mx * mult)
            cands += [mx * mx, mx * mx * 2, mx * 3 + 1]
            cands = sorted(set(c for c in cands if c > 0))
            best_cand = None
            best_cand_score = cur
            for c in cands:
                if not time_ok():
                    break
                if c in U:
                    continue
                new_U = U + [c]
                # Use fast score for screening, exact for verification
                ns = _fast_score(new_U)
                if ns > best_cand_score:
                    exact_ns = compute_score(new_U)
                    if exact_ns > best_cand_score:
                        best_cand_score = exact_ns
                        best_cand = c
            if best_cand is not None:
                U = U + [best_cand]
                U = sorted(set(U))
                cur = best_cand_score
                improved = True
        return U, cur

    for seed in seeds:
        if not time_ok():
            break
        U, s = greedy_grow(seed)
        if s > best_score:
            best_score = s
            best_U = list(U)

    # Final: try greedy grow from the current best as well
    if time_ok():
        U, s = greedy_grow(best_U)
        if s > best_score:
            best_score = s
            best_U = list(U)

    # --- Local search on best set: modify one element ---
    if time_ok():
        U = best_U[:]
        for iteration in range(5000):
            if not time_ok():
                break
            improved = False
            for i in range(len(U)):
                old_val = U[i]
                # Try a range of replacements
                search_range = max(10, max(U) // 4)
                for new_val in range(0, min(max(U) + search_range, 10000)):
                    if new_val == old_val:
                        continue
                    new_U = U[:i] + [new_val] + U[i+1:]
                    if len(set(new_U)) != len(new_U):
                        continue
                    ns = _fast_score(new_U)
                    if ns > best_score:
                        exact_ns = compute_score(new_U)
                        if exact_ns > best_score:
                            best_score = exact_ns
                            best_U = new_U[:]
                            U = new_U[:]
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break

    return best_U
