import time
import random
from itertools import combinations

def _is_sidon_fast(arr):
    n = len(arr)
    seen = set()
    for i in range(n):
        ai = arr[i]
        for j in range(i, n):
            s = ai + arr[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size(arr):
    n = len(arr)
    for k in range(n, 0, -1):
        for subset in combinations(arr, k):
            if _is_sidon_fast(subset):
                return k
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    best_set = list(range(12))
    best_size = _max_sidon_size(best_set)

    random.seed(42)

    candidates = []

    # Key insight: To minimize the max Sidon subset, we want an ambient set
    # where many pairs of elements have the same sum.
    # If we pick elements that are all congruent to the same value mod 2,
    # then all pair sums are even, creating many collisions.
    # More aggressively: pick elements from a small set of residues.
    
    # Strategy 1: All even numbers in a small range
    # {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}
    # Sums: 0+2=2, 4+0=4... wait, 0+2=2, 1+1=2 (but 1 not in set)
    # Actually for Sidon WITH diagonals: a+a and a+b must be unique.
    # For {0,2,4,...,22}: 0+2=2, 2+0=2 (same pair), but 0+4=4, 2+2=4 -> collision!
    # So AP with step 2 is NOT Sidon.
    
    # Let's think about what makes a set have small max Sidon subset.
    # We want many additive collisions.
    
    # Strategy: Use a set where many pairs sum to the same value.
    # E.g., pairs that sum to a constant: {a, b, c, d} where a+d = b+c
    # For 12 elements, we can have 6 pairs all summing to the same value S.
    # Then any two pairs from these 6 give a collision.
    # But we need the elements to be distinct.
    
    # Let's try: 6 pairs summing to 128: (0,128), (1,127), (2,126), (3,125), (4,124), (5,123)
    # This gives 12 distinct elements. Any two pairs have the same sum 128.
    # But we also need to check other sums.
    
    candidates.append([0, 128, 1, 127, 2, 126, 3, 125, 4, 124, 5, 123])
    
    # Try different target sums
    for S in range(130, 250):
        # Find 6 distinct pairs (a,b) with a+b=S, a<b, all 12 values distinct and in [0,128]
        pairs = []
        used = set()
        for a in range(0, 129):
            b = S - a
            if b > a and 0 <= b <= 128 and a not in used and b not in used:
                pairs.append((a, b))
                used.add(a)
                used.add(b)
                if len(pairs) == 6:
                    break
        if len(pairs) == 6:
            arr = sorted([x for p in pairs for x in p])
            candidates.append(arr)
    
    # Strategy: Mix of high-collision structures
    # Two groups of 6, each with internal collisions
    for _ in range(200):
        S1 = random.randint(130, 250)
        S2 = random.randint(130, 250)
        if S1 == S2:
            continue
        pairs1 = []
        used1 = set()
        for a in range(0, 129):
            b = S1 - a
            if b > a and 0 <= b <= 128 and a not in used1 and b not in used1:
                pairs1.append((a, b))
                used1.add(a)
                used1.add(b)
                if len(pairs1) == 6:
                    break
        pairs2 = []
        used2 = set()
        for a in range(0, 129):
            b = S2 - a
            if b > a and 0 <= b <= 128 and a not in used2 and b not in used2:
                pairs2.append((a, b))
                used2.add(a)
                used2.add(b)
                if len(pairs2) == 6:
                    break
        if len(pairs1) == 6 and len(pairs2) == 6:
            all_vals = [x for p in pairs1 for x in p] + [x for p in pairs2 for x in p]
            if len(set(all_vals)) == 12:
                candidates.append(sorted(all_vals))
    
    # Random clustered sets
    for _ in range(500):
        # Pick a small range and cluster
        span = random.randint(12, 40)
        lo = random.randint(0, max(0, 128 - span))
        hi = min(lo + span, 128)
        if hi - lo + 1 >= 12:
            candidates.append(sorted(random.sample(range(lo, hi + 1), 12)))
    
    # APs with small steps
    for step in range(2, 20):
        max_start = 128 - 11 * step
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, max(1, step // 2)):
            arr = [start + i * step for i in range(12)]
            if max(arr) <= 128:
                candidates.append(arr)
    
    # Deduplicate
    seen_sets = set()
    unique_candidates = []
    for c in candidates:
        t = tuple(c)
        if t not in seen_sets:
            seen_sets.add(t)
            unique_candidates.append(c)
    
    # Evaluate candidates
    for arr in unique_candidates:
        if time.time() > deadline - 15.0:
            break
        size = _max_sidon_size(arr)
        if size < best_size:
            best_size = size
            best_set = arr[:]
            if best_size <= 3:
                break
    
    # Local search
    if best_set and time.time() < deadline - 10.0:
        current = best_set[:]
        current_size = best_size
        stall = 0
        max_stall = 50
        
        while stall < max_stall and time.time() < deadline - 5.0:
            stall += 1
            improved = False
            n = len(current)
            
            # Single replacement
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                old = current[i]
                for _ in range(20):
                    new_val = random.randint(0, 128)
                    if new_val == old or new_val in current:
                        continue
                    current[i] = new_val
                    current.sort()
                    ns = _max_sidon_size(current)
                    if ns < current_size:
                        current_size = ns
                        improved = True
                        if current_size < best_size:
                            best_size = current_size
                            best_set = current[:]
                            stall = 0
                            if best_size <= 2:
                                return best_set
                    else:
                        current[i] = old
                        current.sort()
            
            # Pairwise swap
            if not improved:
                for i in range(n):
                    if time.time() > deadline - 5.0:
                        break
                    for j in range(i + 1, n):
                        current[i], current[j] = current[j], current[i]
                        current.sort()
                        ns = _max_sidon_size(current)
                        if ns < current_size:
                            current_size = ns
                            improved = True
                            if current_size < best_size:
                                best_size = current_size
                                best_set = current[:]
                                stall = 0
                                if best_size <= 2:
                                    return best_set
                        else:
                            current[i], current[j] = current[j], current[i]
                            current.sort()
        
        if current_size < best_size:
            best_size = current_size
            best_set = current[:]
    
    return best_set
