"""Rewritten program: lattice-of-5-smooth design with local polish."""
import time
import random
from itertools import combinations

# deadline guard
_DEADLINE = 80.0  # seconds wall-clock budget
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _sumset(A):
    return {a + b for a in A for b in A}


def _prodset(A):
    return {a * b for a in A for b in A}


def _objective(A):
    """Return max(|A+A|, |A*A|)."""
    sa = _sumset(A)
    pa = _prodset(A)
    return max(len(sa), len(pa))


def _is_valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x <= 0 or x > 10000 for x in A):
        return False
    return True


def _generate_5smooth(maxv=10000):
    """Generate all 5-smooth numbers <= maxv."""
    nums = set()
    a = 1
    while a <= maxv:
        b = a
        while b <= maxv:
            c = b
            while c <= maxv:
                nums.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(nums)


def _init_random(n=20, maxv=10000, seed=0):
    rng = random.Random(seed)
    # Start with a mix: some structured, some random
    base = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36]
    # Ensure distinct and <= maxv
    base = sorted(set(x for x in base if x <= maxv))
    while len(base) < n:
        base.append(rng.randint(1, maxv))
    base = sorted(base[:n])
    # Deduplicate
    seen = set()
    out = []
    for x in base:
        if x not in seen:
            seen.add(x)
            out.append(x)
    while len(out) < n:
        x = rng.randint(1, maxv)
        if x not in seen:
            seen.add(x)
            out.append(x)
    out = sorted(out[:n])
    return out


def _local_polish(A, max_iter=200, seed=0):
    """Hill-climbing: swap one element for a better one."""
    rng = random.Random(seed)
    A = sorted(list(A))
    best = _objective(A)
    for _ in range(max_iter):
        if _elapsed() > _DEADLINE - 5:
            break
        improved = False
        i = rng.randrange(20)
        old = A[i]
        # Try a few candidate replacements
        for _ in range(30):
            if _elapsed() > _DEADLINE - 5:
                break
            new_val = rng.randint(1, 10000)
            if new_val == old or new_val in A:
                continue
            trial = A[:]
            trial[i] = new_val
            trial.sort()
            obj = _objective(trial)
            if obj < best:
                best = obj
                A = trial
                improved = True
                break
        if not improved and _elapsed() > _DEADLINE - 10:
            break
    return A, best


def construct():
    t0 = time.monotonic()
    deadline = t0 + _DEADLINE

    # Phase 1: structured initial candidates
    smooth = _generate_5smooth(10000)
    # Pick 20 from 5-smooth that minimize product set: use a grid in log-space
    # Simple heuristic: take geometric progression within 5-smooth
    # Also try small clusters
    candidates = []

    # Candidate 1: powers of 2 up to 2^19 = 524288 (too big), so powers of 2 up to 2^13=8192
    powers2 = [2**i for i in range(0, 14)]  # 1..8192, 14 elements
    # Add some 3-multiples to fill 20
    extra3 = [3, 6, 9, 12, 15, 18]
    c1 = sorted(set(powers2 + extra3)[:20])
    while len(c1) < 20:
        c1.append(len(c1) + 1)
    c1 = sorted(set(x for x in c1 if x <= 10000)[:20])
    candidates.append(c1)

    # Candidate 2: 5-smooth grid: 2^a * 3^b for a,b in range
    grid = []
    for a in range(0, 8):
        for b in range(0, 4):
            v = (2**a) * (3**b)
            if v <= 10000 and v not in grid:
                grid.append(v)
    c2 = sorted(grid[:20])
    while len(c2) < 20:
        c2.append(c2[-1] + 1 if c2 else 1)
    c2 = sorted(set(x for x in c2 if x <= 10000)[:20])
    candidates.append(c2)

    # Candidate 3: small integers 1..20 (baseline)
    candidates.append(list(range(1, 21)))

    # Candidate 4: random 5-smooth selection
    rng = random.Random(42)
    smooth_sample = rng.sample(smooth, min(20, len(smooth)))
    candidates.append(sorted(smooth_sample))

    # Candidate 5: geometric-ish spread
    import math
    c5 = []
    for i in range(20):
        v = int(1 + (9999 - 1) * (i / 19.0))
        c5.append(v)
    c5 = sorted(set(c5))
    while len(c5) < 20:
        c5.append(c5[-1] + 1)
    candidates.append(c5)

    # Evaluate all candidates
    best_A = None
    best_obj = float('inf')
    for cand in candidates:
        if not _is_valid(cand):
            continue
        obj = _objective(cand)
        if obj < best_obj:
            best_obj = obj
            best_A = cand

    # Phase 2: local polish from best candidate
    if best_A is not None:
        if time.monotonic() < deadline - 10:
            polished, obj = _local_polish(best_A, max_iter=300, seed=0)
            if obj < best_obj:
                best_obj = obj
                best_A = polished

    # Phase 3: additional random restarts if time allows
    restart_count = 0
    while time.monotonic() < deadline - 8 and restart_count < 50:
        restart_count += 1
        A = _init_random(n=20, maxv=10000, seed=restart_count * 137)
        obj = _objective(A)
        if obj < best_obj:
            best_obj = obj
            best_A = A
        if time.monotonic() < deadline - 15:
            polished, obj2 = _local_polish(A, max_iter=50, seed=restart_count * 137 + 1)
            if obj2 < best_obj:
                best_obj = obj2
                best_A = polished

    # Final: return best
    if best_A is None or not _is_valid(best_A):
        return list(range(1, 21))
    return sorted(best_A)
