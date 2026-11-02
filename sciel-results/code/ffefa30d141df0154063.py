"""Rewritten program: union-of-arithmetic-progressions parameterization with
CMA-ES-style continuous global search over (start, step, length) tuples,
plus a final high-precision local polish.  Keeps the public entry point
construct() -> list of nonnegative integers containing 0."""

import time
import math
import random
import numpy as np
from numpy.fft import fft, ifft

# ---------------------------------------------------------------------------
# Fast scoring
# ---------------------------------------------------------------------------

def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _score_set_fft(U):
    """Compute 1 + log(|U-U|/|U+U|) / log(2*max(U)+1) using FFT."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0

    # --- sumset via FFT convolution ---
    sum_len = 2 * max_u + 1
    fft_len = _next_pow2(2 * sum_len)
    ind = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        ind[u] = 1.0
    freq = fft(ind)
    conv = ifft(freq * freq).real
    sumset_size = int(np.sum(conv[:sum_len] > 0.5))  # threshold to avoid fp noise

    # --- difference set via FFT autocorrelation ---
    diff_len = 2 * max_u + 1
    fft_len2 = _next_pow2(2 * diff_len)
    dv = np.zeros(fft_len2, dtype=np.float64)
    for u in U:
        dv[u + max_u] = 1.0
    dv_rev = np.zeros(fft_len2, dtype=np.float64)
    for i in range(fft_len2):
        dv_rev[fft_len2 - 1 - i] = dv[i]
    fd = fft(dv)
    fr = fft(dv_rev)
    diff_conv = ifft(fd * fr).real
    diffset_size = int(np.sum(np.abs(diff_conv[:2 * max_u + 1]) > 0.5))

    if sumset_size == 0:
        return 0.0
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + math.log(ratio) / log_denom


def _score_set_direct(U):
    """Direct O(n^2) scoring for small sets."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + math.log(ratio) / log_denom


def _score_set(U):
    """Dispatch to the appropriate scoring method."""
    U = sorted(set(U))
    if len(U) <= 40:
        return _score_set_direct(U)
    return _score_set_fft(U)


# ---------------------------------------------------------------------------
# Arithmetic-progression union representation
# ---------------------------------------------------------------------------

def _ap_to_set(params):
    """Convert a list of (start, step, length) tuples to a sorted list of ints.
    Ensures 0 is in the set."""
    s = set()
    for (st, step, ln) in params:
        st = max(0, int(round(st)))
        step = max(1, int(round(step)))
        ln = max(1, int(round(ln)))
        for i in range(ln):
            s.add(st + i * step)
    s.add(0)
    return sorted(s)


def _score_params(params):
    """Score a parameter vector (list of (start, step, length))."""
    U = _ap_to_set(params)
    return _score_set(U), U


# ---------------------------------------------------------------------------
# CMA-ES-lite: continuous global search over flattened parameter vector
# ---------------------------------------------------------------------------

def _cmaes_search(score_fn, x0, sigma0, max_evals, deadline, rng):
    """A lightweight CMA-ES that works on a flat vector of doubles.
    score_fn(x) -> float (higher is better).
    Returns (best_x, best_score).
    """
    dim = len(x0)
    x = np.array(x0, dtype=np.float64)
    sigma = sigma0
    best_x = x.copy()
    best_f = score_fn(x)

    C = np.eye(dim)
    cs = 0.5 + 0.5 / dim
    ds = 1.0 + 2.0 / max(dim - 1, 1) + cs
    c1 = (2 + 2.0 / (dim + 1.0) ** 2) / ((dim + 2) ** 2 + 2)
    cmu = min(0.3, (2 + 2.0 / (dim + 1.0) ** 2 - c1) /
              ((dim + 2) ** 2 + 5))
    chi = math.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
    ps = np.zeros(dim)
    pc = np.zeros(dim)
    pop_size = 4 + int(3 * math.log(dim))
    half = pop_size // 2

    evals = 0
    while evals < max_evals:
        if time.time() > deadline:
            break

        offspring = []
        for i in range(pop_size):
            z = rng.standard_normal(dim)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
            xi = x + sigma * (L @ z)
            offspring.append(xi)
            evals += 1

        fitness = []
        for xi in offspring:
            f = score_fn(xi)
            fitness.append(f)
            if f > best_f:
                best_f = f
                best_x = xi.copy()

        order = np.argsort(fitness)[::-1]
        mu = np.array([offspring[i] for i in order[:half]])
        weights = np.log(half + 0.5) - np.log(np.arange(1, half + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        m_new = np.sum(mu * weights[:, None], axis=0)
        ps = (1 - cs) * ps + cs / chi * (C @ (m_new - x))
        if np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** 2) > \
                (1.4 + 2.0 / (dim + 1)) * chi:
            sigma *= math.exp(cs / ds * (np.linalg.norm(ps) / chi - 1))
        pc = (1 - c1) * pc + math.sqrt(c1 * (2 - c1) * mueff / dim) * (m_new - x)
        C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + \
            cmu * np.sum(weights[:, None] * np.outer(mu - x, mu - x), axis=0)
        x = m_new

        sigma = max(1e-6, min(sigma, 10.0))
        try:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = (eigvecs * eigvals) @ eigvecs.T
        except np.linalg.LinAlgError:
            C = np.eye(dim)

    return best_x, best_f


# ---------------------------------------------------------------------------
# Local polish (discrete)
# ---------------------------------------------------------------------------

def _local_polish(U, max_iters=500, deadline=None):
    """Discrete local search: swap / add / remove single elements."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper = max_u + 50

    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(len(U)):
            if U[i] == 0:
                continue
            old = U[i]
            for nv in range(max(1, old - 15), min(upper, old + 15) + 1):
                if nv == old:
                    continue
                new_U = U[:]
                new_U[i] = nv
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    improved = True
                    break
            if improved:
                break

        if not improved:
            if len(U) < 120:
                for _ in range(15):
                    nv = random.randint(1, upper)
                    if nv in U:
                        continue
                    new_U = U + [nv]
                    s = _score_set(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        break
            if not improved:
                if len(U) > 3:
                    for i in range(len(U)):
                        if U[i] == 0:
                            continue
                        new_U = U[:i] + U[i + 1:]
                        s = _score_set(new_U)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_U = new_U[:]
                            U = new_U[:]
                            improved = True
                            break
        if not improved:
            if random.random() < 0.15:
                new_U = U[:]
                for _ in range(min(3, len(new_U) - 1)):
                    idx = random.randint(1, len(new_U) - 1)
                    new_U[idx] = random.randint(1, upper)
                if len(set(new_U)) == len(new_U):
                    s = _score_set(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
            if not improved:
                break

    return sorted(set(best_U)), best_score


# ---------------------------------------------------------------------------
# Gap-multiplication construction
# ---------------------------------------------------------------------------

def _generate_primes(n, start=2):
    """Generate first n primes starting from start."""
    primes = []
    candidate = start
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


def _gap_multiplication_set(primes, k, scale=1):
    """Build a set using cumulative sums of products of distinct primes.
    The gaps g_i are products of distinct primes, ensuring unique factorization
    of differences."""
    if len(primes) < 2:
        return [0]
    
    # Generate gaps as products of subsets of primes
    gaps = [1]
    for p in primes:
        new_gaps = []
        for g in gaps:
            new_gaps.append(g * p)
        gaps = gaps + new_gaps
    
    # Sort gaps and take the largest k (excluding 1 which is trivial)
    gaps = sorted(set(g for g in gaps if g > 1))[:k]
    
    # Build cumulative sum
    U = [0]
    current = 0
    for g in gaps:
        current += g * scale
        U.append(current)
    
    return sorted(set(U))


# ---------------------------------------------------------------------------
# construct()
# ---------------------------------------------------------------------------

def construct():
    t0 = time.time()
    deadline = t0 + 145  # 5 s safety margin before 150 s kill

    random.seed(12345)
    rng = np.random.default_rng(12345)

    # ---- Phase 1: generate diverse initial candidates ----
    candidates = []

    # Known good small sets from exhaustive search
    for U in [[0, 1, 2, 5], [0, 1, 3], [0, 1, 4, 6], [0, 1, 4, 10, 11],
              [0, 1, 2, 5, 7], [0, 1, 3, 5, 8], [0, 1, 4, 6, 9],
              [0, 1, 2, 5, 7, 11], [0, 1, 3, 6, 10, 15]]:
        candidates.append(U)

    # Golomb rulers (small)
    _gr = {
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
    }
    for n in range(3, 11):
        if n in _gr:
            candidates.append(_gr[n])

    # Random sets
    for trial in range(20):
        n = random.randint(3, 30)
        mx = random.randint(20, 300)
        U = [0] + sorted(random.sample(range(1, mx + 1), n - 1))
        candidates.append(U)

    # Structured: union of two APs
    for b1 in [2, 3, 5, 7]:
        for b2 in [b1 * 3, b1 * 5, b1 * 7]:
            U = set()
            for i in range(15):
                U.add(i * b1)
            for i in range(10):
                U.add(i * b2)
            U.add(0)
            candidates.append(sorted(U))

    # Gap-multiplication sets
    primes = _generate_primes(8)
    for k in range(3, 12):
        for scale in [1, 2, 3]:
            U = _gap_multiplication_set(primes, k, scale)
            if len(U) >= 3:
                candidates.append(U)

    # Evaluate all candidates
    best_U = candidates[0]
    best_score = _score_set(best_U)
    for U in candidates:
        if time.time() > deadline - 30:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]

    # ---- Phase 2: CMA-ES over AP-parameter vectors ----
    for k in [2, 3, 4]:
        if time.time() > deadline - 10:
            break
        n_best = len(best_U)
        group_size = max(1, n_best // k)
        params = []
        for j in range(k):
            lo = j * group_size
            hi = min(lo + group_size, n_best)
            if lo < hi:
                grp = best_U[lo:hi]
                st = grp[0]
                step = 1
                if len(grp) >= 2:
                    step = max(1, int(round((grp[-1] - grp[0]) / max(1, len(grp) - 1))))
                ln = len(grp)
            else:
                st, step, ln = 0, 1, 1
            params.extend([float(st), float(step), float(ln)])

        x0 = np.array(params, dtype=np.float64)

        def _score_flat(x, k=k):
            p = []
            for j in range(k):
                st = max(0.0, x[3 * j])
                step = max(1.0, x[3 * j + 1])
                ln = max(1.0, x[3 * j + 2])
                p.append((st, step, ln))
            U = _ap_to_set(p)
            return _score_set(U)

        try:
            bx, bf = _cmaes_search(_score_flat, x0, sigma0=2.0,
                                   max_evals=200, deadline=deadline - 10,
                                   rng=rng)
            if bf > best_score:
                best_score = bf
                p = []
                for j in range(k):
                    p.append((bx[3 * j], bx[3 * j + 1], bx[3 * j + 2]))
                best_U = _ap_to_set(p)
        except Exception:
            pass

    # ---- Phase 3: local polish ----
    if time.time() < deadline - 3:
        best_U, best_score = _local_polish(best_U, max_iters=800,
                                           deadline=deadline - 2)

    # ---- Phase 4: try polish on a few other candidates ----
    for U in candidates[:5]:
        if time.time() > deadline - 2:
            break
        if sorted(set(U)) == sorted(set(best_U)):
            continue
        s = _score_set(U)
        if s > best_score - 0.01:
            pu, ps = _local_polish(U, max_iters=300, deadline=deadline - 2)
            if ps > best_score:
                best_score = ps
                best_U = pu

    return sorted(set(best_U))
