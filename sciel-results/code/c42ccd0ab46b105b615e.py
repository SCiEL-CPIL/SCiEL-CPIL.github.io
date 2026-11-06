"""Optimized construction for low-discrepancy signs on arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

N = 256
DEADLINE = 80.0  # safety margin before 90s kill

def _max_abs_aps_sum(signs):
    """Compute max |signed sum| over all APs in [1..N]. Vectorized per step."""
    s = np.asarray(signs, dtype=np.int64)
    n = len(s)
    best = 1  # singletons always have |sum| = 1
    # prefix sums for O(1) range sum
    ps = np.concatenate(([0], np.cumsum(s)))
    for d in range(1, n):
        # for each start a (1-indexed), length L, term a + (L-1)*d <= n
        # sum = ps[a+(L-1)d] - ps[a-1]  (1-indexed -> 0-indexed: ps[a-1+(L-1)d] - ps[a-2]... careful)
        # use 1-indexed: positions a, a+d, ..., a+(L-1)d
        # in 0-indexed array s: indices a-1, a-1+d, ...
        # prefix: ps[k] = sum of s[0..k-1]
        # sum of s[i0..i1] = ps[i1+1]-ps[i0]
        # For AP starting at a (1-idx), 0-idx start = a-1, end 0-idx = a-1+(L-1)*d
        # sum = ps[a-1+(L-1)*d + 1] - ps[a-1] = ps[a+(L-1)*d] - ps[a-1]
        for a in range(1, n + 1):
            # max L such that a + (L-1)*d <= n  => L <= (n-a)/d + 1
            maxL = (n - a) // d + 1
            if maxL < 2:
                continue
            # for all L in [2..maxL], compute sums
            # sums for L=2..maxL: ps[a+(L-1)*d] - ps[a-1]
            # L-1 ranges 1..maxL-1
            Lm1 = np.arange(1, maxL)
            idx = a + Lm1 * d
            sums = ps[idx] - ps[a - 1]
            m = np.max(np.abs(sums))
            if m > best:
                best = int(m)
    return best

def _max_abs_aps_sum_fast(signs):
    """Faster: precompute per-step using vectorized loops over starts, but skip tiny gains.
       We'll instead use a smarter approach: for each step d, compute all AP sums via
       strided slicing and cumsum over each residue class mod d."""
    s = np.asarray(signs, dtype=np.int64)
    n = len(s)
    best = 1
    for d in range(1, n):
        # For each residue r in [0..d-1], the subsequence s[r], s[r+d], ... 
        # Any AP with step d is a contiguous subsequence of one such residue class.
        # So for each residue class, compute prefix sums and max subarray abs.
        for r in range(d):
            if r >= n:
                break
            sub = s[r::d]
            if len(sub) < 2:
                continue
            ps = np.concatenate(([0], np.cumsum(sub)))
            # all contiguous subarray sums: ps[j]-ps[i] for 0<=i<j<=len
            # We need max |ps[j]-ps[i]| over all pairs = max(ps) - min(ps)
            m = int(np.max(ps) - np.min(ps))
            if m > best:
                best = m
    return best

def _initial_construction():
    """Start from a structured low-discrepancy-like sequence.
       Use a Thue-Morse-like or balanced construction, then randomize slightly."""
    # Thue-Morse: t(n) = parity of popcount(n); map to signs
    def thue_morse(n):
        # generate first n thue-morse bits
        tm = [0]
        for i in range(n - 1):
            tm.append(tm[-1] ^ 1)
            # actually thue-morse is: t(2k)=t(k), t(2k+1)=1-t(k)
        # regenerate properly
        tm = [0] * n
        for i in range(1, n):
            # t(i) = popcount(i) mod 2
            tm[i] = bin(i).count('1') % 2
        return tm

    tm = thue_morse(N)
    signs = [1 if b == 0 else -1 for b in tm]
    return signs

def _local_search(signs, budget_sec):
    """Simulated annealing with fast evaluation. Flip one sign at a time."""
    signs = list(signs)
    current = _max_abs_aps_sum_fast(signs)
    best = current
    best_signs = list(signs)
    start = time.time()
    rng = np.random.default_rng(42)
    T = 1.0
    # We'll do flips; each eval is O(N^2 / something) ~ for N=256, per d up to 255, residue classes...
    # _max_abs_aps_sum_fast is O(N^2) ~ 65536 ops per eval, fast enough.
    # Do many iterations.
    for it in range(200000):
        if time.time() - start > budget_sec:
            break
        # pick random index
        i = rng.integers(0, N)
        signs[i] = -signs[i]
        new = _max_abs_aps_sum_fast(signs)
        delta = new - current
        if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            current = new
            if new < best:
                best = new
                best_signs = list(signs)
        else:
            signs[i] = -signs[i]  # revert
        T *= 0.9999
    return best_signs, best

def construct():
    t0 = time.time()
    # Start with Thue-Morse based
    signs = _initial_construction()
    best = _max_abs_aps_sum_fast(signs)
    best_signs = list(signs)
    # Local search with a time budget
    remaining = DEADLINE - (time.time() - t0) - 5.0  # leave 5s margin
    if remaining > 2.0:
        new_signs, new_best = _local_search(signs, remaining)
        if new_best < best:
            best = new_best
            best_signs = new_signs
    # Return as Python ints
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
