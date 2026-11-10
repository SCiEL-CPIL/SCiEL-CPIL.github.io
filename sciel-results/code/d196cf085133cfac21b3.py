import time
import random
import itertools

def _is_sidon(subset):
    """Check if subset is Sidon: all pair sums a[i]+a[j] for i<=j are unique."""
    n = len(subset)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size(cand):
    """Exhaustive: find largest Sidon subset of cand (12 elements)."""
    n = len(cand)
    # Precompute all pair sums as a matrix for quick checking
    # Use bitmask DP: for each subset, check if Sidon
    # 2^12 = 4096 subsets, check each in O(k^2)
    # But we can be smarter: iterate subsets by size descending
    
    # Precompute sum matrix
    sums = [[cand[i] + cand[j] for j in range(n)] for i in range(n)]
    
    # For a subset S (bitmask), check Sidon:
    # Collect all sums S[i]+S[j] for i<=j in S, check uniqueness
    best = 1  # single element is always Sidon
    for size in range(n, 1, -1):
        found = False
        for combo in itertools.combinations(range(n), size):
            mask = sum(1 << i for i in combo)
            # Check Sidon
            seen = set()
            ok = True
            for ii in range(len(combo)):
                i = combo[ii]
                for jj in range(ii, len(combo)):
                    j = combo[jj]
                    s = sums[i][j]
                    if s in seen:
                        ok = False
                        break
                    seen.add(s)
                if not ok:
                    break
            if ok:
                best = size
                found = True
                break
        if found:
            break
    return best

def construct():
    deadline = time.time() + 75  # safety margin before 90s kill
    
    # Strategy: Use structure to minimize max Sidon size.
    # Key: 3-term APs limit Sidon to 2 per AP.
    # If we can partition 12 elements into 4 APs of size 3, max Sidon <= 8.
    # If we can do better with overlapping structures, maybe 7.
    
    # Let's try: construct sets with many 3-term APs
    # Then verify with exact evaluator and search for improvements.
    
    best_cand = None
    best_size = 12
    
    # Initial candidates based on AP structure
    # 4 disjoint 3-term APs: {0,1,2}, {10,11,12}, {20,21,22}, {30,31,32}
    # But we need 12 distinct in [0,128]
    
    # Try various structured candidates
    candidates = []
    
    # 4 APs of length 3
    for base in range(0, 120, 10):
        for gap in range(1, 5):
            c = []
            for k in range(4):
                start = base + k * 10
                c.extend([start, start + gap, start + 2 * gap])
            if len(set(c)) == 12 and all(0 <= x <= 128 for x in c):
                candidates.append(sorted(c))
    
    # 3 APs of length 4: {a, a+d, a+2d, a+3d}
    # Sidon can take at most 3 from a 4-AP (any 3 from 4-AP: check)
    # Actually {a, a+d, a+2d, a+3d}: a+a+3d = a+d+a+2d, so NOT Sidon as 4-set
    # Max Sidon from 4-AP is 3
    for base in range(0, 110, 10):
        for gap in range(1, 4):
            c = []
            for k in range(3):
                start = base + k * 10
                c.extend([start, start + gap, start + 2 * gap, start + 3 * gap])
            if len(set(c)) == 12 and all(0 <= x <= 128 for x in c):
                candidates.append(sorted(c))
    
    # Random search with local improvement
    rng = random.Random(42)
    
    # Evaluate initial candidates
    for c in candidates:
        sz = _max_sidon_size(c)
        if sz < best_size:
            best_size = sz
            best_cand = c
        if time.time() > deadline:
            break
    
    # Local search: mutate one element at a time
    if best_cand is None:
        best_cand = list(range(12))
        best_size = _max_sidon_size(best_cand)
    
    current = list(best_cand)
    current_size = best_size
    no_improve = 0
    
    while time.time() < deadline and no_improve < 500:
        # Try random mutations
        improved = False
        for _ in range(100):
            if time.time() > deadline:
                break
            trial = list(current)
            idx = rng.randint(0, 11)
            old_val = trial[idx]
            # Try replacing with a new value
            for _ in range(20):
                new_val = rng.randint(0, 128)
                if new_val in trial:
                    continue
                trial[idx] = new_val
                if len(set(trial)) != 12:
                    continue
                sz = _max_sidon_size(trial)
                if sz < current_size:
                    current = trial
                    current_size = sz
                    if sz < best_size:
                        best_size = sz
                        best_cand = list(trial)
                    improved = True
                    break
                elif sz == current_size and rng.random() < 0.3:
                    # Accept equal to explore
                    current = trial
                    break
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
    
    return best_cand if best_cand else list(range(12))
