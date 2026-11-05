"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = time.time() + 82.0
_RNG = random.Random(12345)

def _square(coeffs):
    n = len(coeffs)
    out = [0] * (2 * n - 1)
    for i in range(n):
        a = coeffs[i]
        if a == 0:
            continue
        for j in range(n):
            b = coeffs[j]
            if b == 0:
                continue
            out[i + j] += a * b
    return out

def _nnz_square(coeffs):
    return sum(1 for v in _square(coeffs) if v != 0)

def _validate(coeffs):
    if len(coeffs) > 41:
        return False
    if not coeffs or coeffs[-1] == 0:
        return False
    if any(c < -64 or c > 64 for c in coeffs):
        return False
    if sum(1 for c in coeffs if c != 0) != 12:
        return False
    return True

def _random_candidate():
    pos = _RNG.sample(range(41), 12)
    pos.sort()
    coeffs = [0] * 41
    for p in pos:
        coeffs[p] = _RNG.choice([1, 1, 1, -1, -1, 1, -1, 2, -2, 3, -3])
    while coeffs and coeffs[-1] == 0:
        coeffs.pop()
    if not coeffs:
        return None
    return coeffs

def _local_improve(coeffs, max_rounds=100):
    n = len(coeffs)
    vals = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 8, -8, 10, -10]
    for _ in range(max_rounds):
        if time.time() > _DEADLINE:
            break
        before = _nnz_square(coeffs)
        improved = False
        for i in range(n):
            if time.time() > _DEADLINE:
                break
            orig = coeffs[i]
            for v in vals:
                if v == orig:
                    continue
                coeffs[i] = v
                s = _nnz_square(coeffs)
                if s < before:
                    before = s
                    improved = True
                    break
            coeffs[i] = orig
        if not improved:
            break
    return coeffs

def construct():
    start = time.time()
    best_coeffs = None
    best_score = 10**9

    attempts = 0
    while time.time() < _DEADLINE:
        attempts += 1
        coeffs = _random_candidate()
        if coeffs is None:
            continue
        if not _validate(coeffs):
            continue
        coeffs = _local_improve(coeffs, max_rounds=80)
        s = _nnz_square(coeffs)
        if s < best_score:
            best_score = s
            best_coeffs = coeffs[:]
        if best_score <= 14 and attempts > 500:
            break

    if best_coeffs is None or not _validate(best_coeffs):
        return [1] * 12
    return best_coeffs
# EVOLVE-BLOCK-END
