"""Sparse squares of twelve-term integer polynomials — fresh design with local search."""
import time
import random
from itertools import product

_DEADLINE = 90.0
_START = None


def _nonzero_count(poly):
    return sum(1 for c in poly if c != 0)


def _square_support(coeffs):
    """Return number of nonzero terms in the exact integer square of coeffs."""
    n = len(coeffs)
    # convolution
    sq = [0] * (2 * n - 1)
    for i in range(n):
        a = coeffs[i]
        if a == 0:
            continue
        for j in range(n):
            b = coeffs[j]
            if b == 0:
                continue
            sq[i + j] += a * b
    return sum(1 for c in sq if c != 0)


def _valid(candidate):
    """Check all task constraints."""
    if not isinstance(candidate, list):
        return False
    if len(candidate) == 0 or len(candidate) > 41:
        return False
    if candidate[-1] == 0:
        return False
    for c in candidate:
        if not isinstance(c, int):
            return False
        if c < -64 or c > 64:
            return False
    if sum(1 for c in candidate if c != 0) != 12:
        return False
    return True


def _random_candidate(rng, degree):
    """Generate a random valid 12-term polynomial of given degree."""
    positions = sorted(rng.sample(range(degree + 1), 12))
    # ensure highest position is used (it is, since we pick from range)
    coeffs = [0] * (degree + 1)
    for p in positions:
        coeffs[p] = rng.choice([1, -1])
    # ensure nonzero final coefficient
    if coeffs[-1] == 0:
        coeffs[-1] = 1
    # ensure exactly 12 nonzero
    if sum(1 for c in coeffs if c != 0) != 12:
        # fix by flipping a zero to 1 if needed
        nz = sum(1 for c in coeffs if c != 0)
        if nz < 12:
            for i in range(len(coeffs)):
                if coeffs[i] == 0:
                    coeffs[i] = 1
                    nz += 1
                    if nz == 12:
                        break
        elif nz > 12:
            for i in range(len(coeffs) - 1, -1, -1):
                if coeffs[i] != 0 and i != len(coeffs) - 1:
                    coeffs[i] = 0
                    nz -= 1
                    if nz == 12:
                        break
    if coeffs[-1] == 0:
        coeffs[-1] = 1
    return coeffs


def _local_search(coeffs, rng, deadline):
    """Local search: flip signs and shift terms to reduce square support."""
    best = _square_support(coeffs)
    best_c = list(coeffs)
    n = len(coeffs)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try flipping signs of individual nonzero terms
        for i in range(n):
            if coeffs[i] == 0:
                continue
            trial = list(coeffs)
            trial[i] = -trial[i]
            s = _square_support(trial)
            if s < best:
                best = s
                best_c = trial
                improved = True
            elif s == best:
                # accept with some probability to escape plateaus
                if rng.random() < 0.1:
                    best_c = trial
                    improved = True
        # Try moving a term to a nearby position
        for i in range(n - 1):
            if coeffs[i] == 0:
                continue
            for delta in (-1, 1):
                j = i + delta
                if 0 <= j < n and coeffs[j] == 0:
                    trial = list(coeffs)
                    trial[j] = trial[i]
                    trial[i] = 0
                    if trial[-1] != 0:
                        s = _square_support(trial)
                        if s < best:
                            best = s
                            best_c = trial
                            improved = True
                        elif s == best and rng.random() < 0.05:
                            best_c = trial
                            improved = True
        coeffs = best_c
    return best_c, best


def construct():
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + _DEADLINE - 2.0  # safety margin

    rng = random.Random(42)

    # Start with a good structured candidate: alternating signs on even positions
    # e.g., positions 0,2,4,...,22 with alternating signs
    base_degree = 22
    base = [0] * (base_degree + 1)
    for k in range(12):
        pos = 2 * k
        base[pos] = 1 if k % 2 == 0 else -1
    base[-1] = 1 if base[-1] == 0 else base[-1]
    # ensure exactly 12 nonzero
    if sum(1 for c in base if c != 0) != 12:
        base = _random_candidate(rng, base_degree)

    best_c = base
    best_score = _square_support(best_c)

    # Local search on the base
    best_c, best_score = _local_search(best_c, rng, deadline)

    # Try multiple random restarts if time allows
    for restart in range(200):
        if time.time() > deadline:
            break
        degree = rng.randint(11, 40)
        cand = _random_candidate(rng, degree)
        s = _square_support(cand)
        if s < best_score:
            best_c = cand
            best_score = s
        # short local search on promising candidates
        if s <= best_score + 2:
            cand2, s2 = _local_search(cand, rng, deadline)
            if s2 < best_score:
                best_c = cand2
                best_score = s2
        if time.time() > deadline:
            break

    # Final local polish
    if time.time() < deadline:
        best_c, best_score = _local_search(best_c, rng, deadline)

    # Ensure validity
    if not _valid(best_c):
        # fallback to a known valid candidate
        best_c = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        # pad to length 23, 12 ones at even positions
        while len(best_c) < 41:
            best_c.append(0)
        # trim trailing zeros
        while len(best_c) > 1 and best_c[-1] == 0:
            best_c.pop()
        # ensure 12 nonzero
        if sum(1 for c in best_c if c != 0) != 12:
            best_c = [1] * 12

    return best_c
