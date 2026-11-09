"""Rewritten program: ambient 12-set minimizing the largest Sidon subset size."""
import time
import random
import itertools

DEADLINE = 80.0  # safety margin under the 90s wall-clock limit


def _is_sidon(sub):
    """True iff all unordered pair sums (i<=j, incl. diagonals) are unique."""
    k = len(sub)
    seen = set()
    for i in range(k):
        a = sub[i]
        for j in range(i, k):
            s = a + sub[j]
            if s in seen:
                return False
            seen.add(s)
    return True


def _largest_sidon_subset(arr):
    """Exact largest Sidon subset size for a 12-element set (brute force)."""
    n = len(arr)
    best = 1
    # iterate subsets by descending size to allow early exit
    for r in range(n, best, -1):
        for comb in itertools.combinations(range(n), r):
            if _is_sidon([arr[i] for i in comb]):
                return r
    return best


def construct():
    start = time.time()
    best_val = 4          # seed value (range(12) has largest Sidon subset 4)
    best_set = list(range(12))

    rng = random.Random(20240517)
    evens = list(range(0, 128, 2))   # even numbers: 64 of them, spread out
    odds = list(range(1, 128, 2))    # odd numbers: 64 of them
    pool = list(range(128))

    # A few structured initial candidates to seed the search
    candidates = []
    candidates.append(list(range(12)))                       # the seed
    candidates.append(list(evens[:12]))                       # first 12 evens
    candidates.append(list(odds[:12]))                        # first 12 odds
    candidates.append(list(range(1, 128, 11))[:12])           # arithmetic-ish spread
    candidates.append(sorted(rng.sample(pool, 12)))           # random
    candidates.append(sorted(rng.sample(evens, 12)))          # random evens
    candidates.append(sorted(rng.sample(odds, 12)))           # random odds

    for cand in candidates:
        if time.time() - start > DEADLINE:
            break
        v = _largest_sidon_subset(cand)
        if v < best_val:
            best_val = v
            best_set = cand
            if best_val == 3:
                return best_set   # 3 is the practical minimum; stop early

    # Main randomized search
    while time.time() - start < DEADLINE:
        # bias toward even or odd numbers to increase spacing, sometimes full pool
        roll = rng.random()
        if roll < 0.4:
            cand = sorted(rng.sample(evens, 12))
        elif roll < 0.8:
            cand = sorted(rng.sample(odds, 12))
        else:
            cand = sorted(rng.sample(pool, 12))

        v = _largest_sidon_subset(cand)
        if v < best_val:
            best_val = v
            best_set = cand
            if best_val == 3:
                return best_set

    return best_set
