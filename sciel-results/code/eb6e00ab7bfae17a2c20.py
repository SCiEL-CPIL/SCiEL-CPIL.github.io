"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _is_sidon(subset):
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

def _has_sidon_subset(ambient, target_size):
    """Check if ambient has ANY Sidon subset of size target_size."""
    for combo in itertools.combinations(ambient, target_size):
        if _is_sidon(combo):
            return True
    return False

def _largest_sidon_with_diagonals(ambient):
    """Find the largest subset of ambient with all pair sums (incl. diagonals) unique."""
    ambient = sorted(ambient)
    n = len(ambient)
    # We only care about small sizes for minimization
    # Check from 6 downwards, if 6 is not found, check 5, etc.
    # If we find a set with max size 5, it's better than 6.
    for size in range(6, 0, -1):
        if _has_sidon_subset(ambient, size):
            return size
    return 0

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
    
    candidates = []
    
    # Construction 1: Two clusters (APs)
    for small_start in range(0, 10):
        for large_start in range(100, 120):
            small = list(range(small_start, small_start + 6))
            large = list(range(large_start, large_start + 6))
            if max(large) <= 128:
                candidates.append(small + large)
    
    # Construction 2: Arithmetic progression
    for start in range(0, 10):
        for step in range(1, 12):
            ap = [start + i * step for i in range(12)]
            if max(ap) <= 128:
                candidates.append(ap)
                
    # Construction 3: Mixed steps (Two APs with different steps)
    for s1 in range(1, 5):
        for s2 in range(1, 5):
            if s1 == s2: continue
            a1 = list(range(0, 6 * s1, s1))
            a2_start = 50
            a2 = [a2_start + i * s2 for i in range(6)]
            if max(a2) <= 128 and len(set(a1 + a2)) == 12:
                candidates.append(a1 + a2)

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
            if best_score <= 5:
                # 5 is very good, 4 is even better but rare. 
                # If we find 5, we can stop early if we want, but let's keep searching a bit for 4.
                # Actually, if we find 5, it's a significant improvement over 6.
                pass
    
    # If we have time, do some random search with structure
    if time.time() < deadline - 10:
        rng = random.Random(42)
        for _ in range(2000):
            if time.time() > deadline:
                break
            # Random set with some structure (AP-like)
            base = rng.randint(0, 10)
            step = rng.randint(1, 8)
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
