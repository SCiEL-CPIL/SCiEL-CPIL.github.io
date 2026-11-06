"""Search for ambient 12-set minimizing largest Sidon subset size."""
import random
import time
from itertools import combinations

def _is_sidon(S):
    n = len(S)
    S2 = {x * x for x in S}
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in S2:
                return False
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size(A, timeout=2.0):
    n = len(A)
    # Check sizes from n down to 1
    # For n=12, this is fast enough to be exact.
    for k in range(n, 0, -1):
        if time.time() > timeout:
            return None
        for subset in combinations(A, k):
            if _is_sidon(subset):
                return k
    return 0

def construct():
    deadline = time.time() + 80.0

    def _validate(A):
        if len(A) != 12:
            return False
        if len(set(A)) != 12:
            return False
        if any(x < 0 or x > 128 for x in A):
            return False
        return True

    candidates = []

    # Strategy 1: Dense sets from small ranges
    # Sets with small diameter tend to have smaller max Sidon subsets.
    for max_val in [12, 15, 20, 25, 30, 40, 50]:
        if time.time() > deadline - 10.0:
            break
        # Sample random subsets of [0, max_val]
        # Number of such subsets is C(max_val+1, 12). For max_val=20, ~290k.
        # We can iterate all if max_val is small, or sample.
        if max_val <= 20:
            # Iterate all subsets for very small ranges
            for subset in combinations(range(max_val + 1), 12):
                A = list(subset)
                ms = _max_sidon_size(A, timeout=0.1)
                if ms is not None and ms < 12:
                    candidates.append((A, ms))
        else:
            # Sample
            rng = random.Random(max_val)
            for _ in range(5000):
                if time.time() > deadline - 10.0:
                    break
                A = sorted(rng.sample(range(max_val + 1), 12))
                ms = _max_sidon_size(A, timeout=0.1)
                if ms is not None and ms < 12:
                    candidates.append((A, ms))

    # Strategy 2: Clusters of 3 close elements
    for gap in [10, 15, 20, 25, 30, 40, 50]:
        for offset in range(0, 129 - 3*gap, 2):
            if time.time() > deadline - 10.0:
                break
            A = []
            for k in range(4):
                base = offset + k * gap
                if base + 2 > 128:
                    break
                A.extend([base, base+1, base+2])
            if len(A) == 12 and len(set(A)) == 12:
                ms = _max_sidon_size(A, timeout=0.5)
                if ms is not None and ms < 12:
                    candidates.append((A, ms))
    
    # Strategy 3: Pairs of close elements (6 pairs)
    for gap in [5, 8, 10, 12, 15, 20]:
        for offset in range(0, 129 - 6*gap, 2):
            if time.time() > deadline - 10.0:
                break
            A = []
            for k in range(6):
                base = offset + k * gap
                if base + 1 > 128:
                    break
                A.extend([base, base+1])
            if len(A) == 12 and len(set(A)) == 12:
                ms = _max_sidon_size(A, timeout=0.5)
                if ms is not None and ms < 12:
                    candidates.append((A, ms))
    
    # Strategy 4: Random global search
    rng = random.Random(42)
    for _ in range(10000):
        if time.time() > deadline - 5.0:
            break
        A = sorted(rng.sample(range(0, 129), 12))
        ms = _max_sidon_size(A, timeout=0.3)
        if ms is not None and ms < 12:
            candidates.append((A, ms))
    
    if not candidates:
        # Fallback: try a known bad set
        # {0,1,2, 10,11,12, 20,21,22, 30,31,32}
        A = [0,1,2, 10,11,12, 20,21,22, 30,31,32]
        if _validate(A):
            return A
        return [3*i for i in range(12)]
    
    candidates.sort(key=lambda t: t[1])
    best = candidates[0][0]
    
    if not _validate(best):
        return [3*i for i in range(12)]
    
    return best
