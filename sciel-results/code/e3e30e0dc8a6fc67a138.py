import time
import random
from itertools import combinations


def _is_sidon_with_diagonals(s):
    """Check if all unordered pair sums (including diagonals) are unique."""
    n = len(s)
    seen = set()
    for i in range(n):
        si = s[i]
        for j in range(i, n):
            v = si + s[j]
            if v in seen:
                return False
            seen.add(v)
    return True


def _max_sidon_size(arr):
    """Find largest subset of arr with all pair sums (incl. diagonals) unique."""
    for k in range(6, 0, -1):
        for comb in combinations(arr, k):
            if _is_sidon_with_diagonals(comb):
                return k
    return 1


def construct():
    deadline = time.time() + 70
    best_set = list(range(12))
    best_val = _max_sidon_size(best_set)

    random.seed(42)

    # Try structured candidates
    structured = [
        list(range(0, 12)),
        list(range(0, 11)) + [13],
        list(range(0, 11)) + [14],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13],
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13],
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13],
    ]
    for s in structured:
        if time.time() > deadline:
            break
        val = _max_sidon_size(s)
        if val < best_val:
            best_val = val
            best_set = s
            if best_val <= 5:
                return best_set

    # Random search with compressed ranges
    for _ in range(30000):
        if time.time() > deadline:
            break
        base = random.randint(0, 100)
        R = random.randint(11, 16)
        pool = list(range(base, min(base + R + 1, 129)))
        if len(pool) < 12:
            continue
        current = random.sample(pool, 12)
        val = _max_sidon_size(current)
        if val < best_val:
            best_val = val
            best_set = current
            if best_val <= 5:
                return best_set

    return best_set
