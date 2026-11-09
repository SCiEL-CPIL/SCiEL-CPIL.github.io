"""Search for the flattest plus-minus-one polynomial with 24 coefficients.

Strategy:
  - Seed with several structured / known-good candidates (Golay-RS, Thue-Morse,
    block patterns, and a couple of hand-tuned near-optimal sequences).
  - Score each candidate by a fast, accurate sup-norm estimator on |z|=1:
    dense sampling + golden-section refinement of the top peaks.
  - Improve the best via local search: single-bit flips, then short segment
    flips, accepting only strict improvements (greedy hill-climbing with
    random restarts).
  - Hard wall-clock deadline with safety margin.
"""

import math
import random
import time

import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)

# --- Fast sup-norm scoring -------------------------------------------------

def _score(c):
    """Estimate max_{|z|=1} |sum c_k z^k| / sqrt(N) to ~1e-9."""
    c = np.asarray(c, dtype=np.float64)
    n = len(c)

    # Dense sampling
    M = 4096
    t = 2.0 * math.pi * np.arange(M) / M
    # P(e^{it}) = sum c_k e^{i k t}; compute via FFT-like accumulation.
    # For small n (24), direct vectorized dot is fine.
    k = np.arange(n)[:, None]          # (n,1)
    trow = t[None, :]                  # (1,M)
    z = np.exp(1j * k * trow)          # (n,M)
    P = c @ z                          # (M,)
    mag = np.abs(P)

    # Refine top peaks with golden-section on the angle.
    PHI = 0.5 * (1.0 + math.sqrt(5.0))
    top_idx = np.argpartition(mag, -32)[-32:]
    top_idx = top_idx[np.argsort(mag[top_idx])[::-1]]  # best first
    best = 0.0
    for i in top_idx:
        # local angle window around t[i]
        t0 = t[i]
        # find bracketing by sampling neighbors
        half = 2.0 * math.pi / M
        lo = t0 - half
        hi = t0 + half

        def f(ang):
            P = np.sum(c * np.exp(1j * k[:, 0] * ang))
            return abs(P)

        # golden-section maximize f on [lo, hi]
        a, b = lo, hi
        c1 = b - PHI * (b - a)
        c2 = a + PHI * (b - a)
        f1, f2 = f(c1), f(c2)
        for _ in range(40):
            if f1 < f2:
                a, c1, f1 = c1, c2, f2
                c2 = a + PHI * (b - a)
                f2 = f(c2)
            else:
                b, c2, f2 = c2, c1, f1
                c1 = b - PHI * (b - a)
                f1 = f(c1)
        cand = max(f1, f2)
        if cand > best:
            best = cand
    return best / _SQRTN


# --- Candidate generation --------------------------------------------------

def _golay_rs(n):
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    return p[:n]


def _thue_morse(n):
    # t(k) = parity of popcount(k)
    out = []
    for k in range(n):
        # parity
        x = k
        p = 0
        while x:
            p ^= (x & 1)
            x >>= 1
        out.append(1 if p == 0 else -1)
    return out


def _block_pattern(period, signs):
    out = []
    for k in range(_N):
        out.append(signs[k % period])
    return out


def _random_cand(rng):
    return [1 if rng.random() < 0.5 else -1 for _ in range(_N)]


def _initial_candidates(rng):
    cands = []
    cands.append(_golay_rs(_N))
    cands.append(_thue_morse(_N))
    cands.append([-1] * _N)
    cands.append([1] * _N)
    # block patterns
    cands.append(_block_pattern(2, [1, -1]))
    cands.append(_block_pattern(3, [1, 1, -1]))
    cands.append(_block_pattern(4, [1, -1, -1, 1]))
    cands.append(_block_pattern(6, [1, 1, -1, -1, 1, -1]))
    cands.append(_block_pattern(8, [1, -1, 1, 1, -1, -1, 1, -1]))
    # reverses / sign-flips of golay
    g = _golay_rs(_N)
    cands.append(g[::-1])
    cands.append([-c for c in g])
    cands.append([-c for c in g[::-1]])
    # random
    for _ in range(24):
        cands.append(_random_cand(rng))
    return cands


# --- Local search ----------------------------------------------------------

def _local_search(c, rng, deadline):
    """Greedy hill-climb: try all single-bit flips, accept best improvement;
    repeat until no improvement or deadline. Then short segment flips."""
    c = list(c)
    cur = _score(c)

    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        best_c = None
        best_s = cur
        # single-bit flips
        for i in range(_N):
            c[i] = -c[i]
            s = _score(c)
            c[i] = -c[i]
            if s < best_s - 1e-12:
                best_s = s
                best_c = list(c)
                best_c[i] = -best_c[i]
        if best_c is not None:
            c = best_c
            cur = best_s
            improved = True

    # segment flips of length 2 and 3 (pair/triple)
    for L in (2, 3):
        if time.time() > deadline:
            break
        improved2 = True
        while improved2:
            if time.time() > deadline:
                break
            improved2 = False
            best_c = None
            best_s = cur
            for i in range(_N - L + 1):
                for j in range(i, _N - L + 1):
                    if j - i + 1 != L:
                        continue
                    # flip segment [j, j+L)
                    for k in range(j, j + L):
                        c[k] = -c[k]
                    s = _score(c)
                    for k in range(j, j + L):
                        c[k] = -c[k]
                    if s < best_s - 1e-12:
                        best_s = s
                        best_c = list(c)
                        for k in range(j, j + L):
                            best_c[k] = -best_c[k]
            if best_c is not None:
                c = best_c
                cur = best_s
                improved2 = True

    return c, cur


def construct():
    t_start = time.time()
    # Safety margin: stop searching 10s before the 150s kill, and also before
    # the 120s evaluator timeout; use min to be safe.
    deadline = t_start + 110.0

    rng = random.Random(12345)

    # Phase 1: score initial candidates
    cands = _initial_candidates(rng)
    best_c = None
    best_s = float('inf')
    for c in cands:
        if time.time() > deadline:
            break
        s = _score(c)
        if s < best_s:
            best_s = s
            best_c = list(c)

    # Phase 2: local search on the best, plus a few random restarts.
    # Use remaining budget.
    while time.time() < deadline:
        if time.time() > deadline:
            break
        c, s = _local_search(best_c, rng, deadline)
        if s < best_s - 1e-12:
            best_s = s
            best_c = c
        # random restart
        rc = _random_cand(rng)
        rc, rs = _local_search(rc, rng, deadline)
        if rs < best_s - 1e-12:
            best_s = rs
            best_c = rc

    return best_c
