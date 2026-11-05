"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations
from collections import Counter

# EVOLVE-BLOCK-START
def _is_sidon(subset):
    """Check if a subset has all unique pair sums (including diagonals)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_subset_size(arr):
    """Compute the largest subset of arr with all unordered pair sums unique (including diagonals)."""
    n = len(arr)
    for k in range(n, 0, -1):
        for subset in combinations(arr, k):
            if _is_sidon(subset):
                return k
    return 0

def _count_duplicate_sums(arr):
    """Count total number of duplicate pair sum occurrences (higher = more collisions)."""
    sums = {}
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            sums[s] = sums.get(s, 0) + 1
    total_pairs = n * (n + 1) // 2
    unique_sums = len(sums)
    return total_pairs - unique_sums

def _evaluate_exact(arr):
    """Exact evaluation: return max_sidon_size (lower is better)."""
    return _max_sidon_subset_size(arr)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit

    random.seed(42)

    # Phase 1: Fast search using collision count as surrogate
    # We want to MAXIMIZE collisions to minimize the largest Sidon subset

    best_arr = list(range(12))
    best_collisions = _count_duplicate_sums(best_arr)
    # Also track top candidates for exact verification
    candidates = []  # list of (collisions, arr)

    def try_update_best(arr, collisions):
        nonlocal best_arr, best_collisions, candidates
        if collisions > best_collisions:
            best_arr = arr[:]
            best_collisions = collisions
        # Keep top candidates
        candidates.append((collisions, arr[:]))
        if len(candidates) > 50:
            candidates.sort(key=lambda x: -x[0])
            candidates = candidates[:50]

    for restart in range(3000):
        if time.time() > deadline - 10:  # Reserve 10s for final phase
            break

        # Multiple initialization strategies
        strategy = restart % 5
        if strategy == 0:
            arr = sorted(random.sample(range(129), 12))
        elif strategy == 1:
            # 4 clusters of 3
            base = random.randint(0, 85)
            gap = random.randint(10, 14)
            arr = []
            for i in range(4):
                offset = base + i * gap
                arr.extend([offset, offset+1, offset+2])
            if max(arr) > 128 or len(set(arr)) < 12:
                continue
            arr = sorted(arr)
        elif strategy == 2:
            # 6 pairs
            base = random.randint(0, 95)
            gap = random.randint(12, 18)
            arr = []
            for i in range(6):
                offset = base + i * gap
                arr.extend([offset, offset + random.randint(1, 4)])
            if len(set(arr)) < 12 or max(arr) > 128:
                continue
            arr = sorted(arr)
        elif strategy == 3:
            # 3 groups of 4
            base = random.randint(0, 75)
            gap = random.randint(14, 20)
            arr = []
            for i in range(3):
                offset = base + i * gap
                arr.extend([offset, offset+1, offset+2, offset+3])
            if max(arr) > 128 or len(set(arr)) < 12:
                continue
            arr = sorted(arr)
        else:
            # Dense cluster: 12 elements in a small range
            base = random.randint(0, 100)
            spread = random.randint(8, 15)
            arr = sorted(random.sample(range(base, min(base + spread + 1, 129)), 12))
            if len(arr) < 12:
                continue

        collisions = _count_duplicate_sums(arr)
        try_update_best(arr, collisions)

        # Local search with fast collision evaluation
        no_improve = 0
        for step in range(5000):
            if time.time() > deadline - 10:
                break

            move_type = random.random()
            if move_type < 0.4:
                # Single replacement
                i = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val in arr:
                    continue
                new_arr = arr.copy()
                new_arr[i] = new_val
            elif move_type < 0.65:
                # Double replacement
                i, j = random.sample(range(12), 2)
                nv1 = random.randint(0, 128)
                nv2 = random.randint(0, 128)
                if nv1 in arr or nv2 in arr or nv1 == nv2:
                    continue
                new_arr = arr.copy()
                new_arr[i] = nv1
                new_arr[j] = nv2
            elif move_type < 0.85:
                # Small perturbation
                i = random.randint(0, 11)
                delta = random.choice([-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8])
                new_val = arr[i] + delta
                if new_val < 0 or new_val > 128 or new_val in arr:
                    continue
                new_arr = arr.copy()
                new_arr[i] = new_val
            else:
                # Swap two elements with new values
                i, j = random.sample(range(12), 2)
                nv1 = random.randint(0, 128)
                nv2 = random.randint(0, 128)
                if nv1 in arr or nv2 in arr or nv1 == nv2:
                    continue
                new_arr = arr.copy()
                new_arr[i] = nv1
                new_arr[j] = nv2

            new_arr = sorted(new_arr)
            if len(set(new_arr)) != 12:
                continue

            new_collisions = _count_duplicate_sums(new_arr)

            if new_collisions > collisions:
                arr = new_arr
                collisions = new_collisions
                no_improve = 0
                try_update_best(arr, collisions)
            else:
                no_improve += 1
                if no_improve > 200:
                    break

    # Phase 2: Exact verification of top candidates
    # Sort candidates by collision count (descending) and evaluate exactly
    candidates.sort(key=lambda x: -x[0])
    
    best_exact_size = 999
    best_exact_arr = best_arr

    for coll, cand in candidates:
        if time.time() > deadline - 1:
            break
        exact_size = _evaluate_exact(cand)
        if exact_size < best_exact_size:
            best_exact_size = exact_size
            best_exact_arr = cand
            if best_exact_size <= 4:
                break

    # If the initial best_arr wasn't in candidates, evaluate it too
    if best_exact_arr is best_arr:
        # Already evaluated
        pass
    else:
        # Evaluate best_arr from fast search if not already done
        init_size = _evaluate_exact(best_arr)
        if init_size < best_exact_size:
            best_exact_size = init_size
            best_exact_arr = best_arr

    return best_exact_arr
# EVOLVE-BLOCK-END
