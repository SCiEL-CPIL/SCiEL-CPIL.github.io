"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _max_sidon_with_diagonals(subset):
    """Return True if subset has all pair sums (incl. diagonals) unique."""
    subset = sorted(subset)
    k = len(subset)
    sums = set()
    for i in range(k):
        for j in range(i, k):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _largest_sidon_with_diagonals(ambient):
    """Find the largest subset of ambient with all pair sums (incl. diagonals) unique."""
    ambient = sorted(ambient)
    n = len(ambient)
    best_size = 0
    # Try all subsets from largest to smallest
    for size in range(n, 0, -1):
        found = False
        for combo in itertools.combinations(ambient, size):
            if _max_sidon_with_diagonals(combo):
                best_size = size
                found = True
                break
        if found:
            break
    return best_size

def _evaluate(ambient):
    """Evaluate a candidate ambient set. Returns the max Sidon-with-diagonals subset size."""
    if len(ambient) != 12:
        return 999
    if len(set(ambient)) != 12:
        return 999
    if any(x < 0 or x > 128 for x in ambient):
        return 999
    return _largest_sidon_with_diagonals(ambient)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s
    
    # Try some heuristic constructions first
    candidates = []
    
    # Construction 1: Two clusters
    for small_start in range(0, 10):
        for large_start in range(100, 120):
            small = list(range(small_start, small_start + 6))
            large = list(range(large_start, large_start + 6))
            if max(large) <= 128:
                candidates.append(small + large)
    
    # Construction 2: Arithmetic progression
    for start in range(0, 10):
        for step in range(2, 12):
            ap = [start + i * step for i in range(12)]
            if max(ap) <= 128:
                candidates.append(ap)
    
    # Construction 3: Symmetric pairs
    for center in range(30, 100):
        for spread in range(1, 20):
            s = []
            for i in range(6):
                a = center - spread * (6 - i)
                b = center + spread * (6 - i)
                if 0 <= a <= 128 and 0 <= b <= 128:
                    s.extend([a, b])
            if len(s) == 12 and len(set(s)) == 12:
                candidates.append(s)
    
    # Evaluate candidates
    best_set = None
    best_score = 999
    
    for c in candidates:
        if time.time() > deadline:
            break
        score = _evaluate(c)
        if score < best_score:
            best_score = score
            best_set = c
    
    # If we have time, do some random search
    if time.time() < deadline - 10:
        rng = random.Random(42)
        for _ in range(2000):
            if time.time() > deadline:
                break
            # Random set with some structure
            base = rng.randint(0, 10)
            step = rng.randint(2, 8)
            s = []
            for i in range(12):
                val = base + i * step + rng.randint(-2, 2)
                if 0 <= val <= 128:
                    s.append(val)
            if len(s) == 12 and len(set(s)) == 12:
                score = _evaluate(s)
                if score < best_score:
                    best_score = score
                    best_set = s
    
    if best_set is None:
        return list(range(12))
    
    return best_set
# EVOLVE-BLOCK-END
