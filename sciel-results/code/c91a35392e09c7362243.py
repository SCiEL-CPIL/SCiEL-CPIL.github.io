"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def _collision_count(arr):
    """Count duplicate pair sums using NumPy vectorization. Higher = more collisions."""
    a = np.array(arr, dtype=np.int32)
    n = len(a)
    i_idx, j_idx = np.triu_indices(n)
    sums = a[i_idx] + a[j_idx]
    unique_count = len(np.unique(sums))
    total_pairs = n * (n + 1) // 2
    return total_pairs - unique_count

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
    # Use bitmask DP or greedy with verification
    # For n=12, check from top down with pruning
    for k in range(n, 1, -1):
        for subset in combinations(arr, k):
            if _is_sidon(subset):
                return k
    return 2  # Any 2 elements form a Sidon set (only 3 sums, always unique for distinct elements)

def _evaluate_exact(arr):
    """Exact evaluation: return max Sidon subset size."""
    return _max_sidon_subset_size(arr)

def _max_sidon_subset_size_fast(arr):
    """Faster exact evaluation using greedy + verification."""
    n = len(arr)
    arr_sorted = sorted(arr)
    
    # Try to build largest Sidon set greedily, then verify
    # But we need exact, so use combinations with early exit
    for k in range(n, 1, -1):
        found = False
        for subset in combinations(arr_sorted, k):
            if _is_sidon(subset):
                return k
    return 2

def construct():
    start_time = time.time()
    deadline = start_time + 55

    best_arr = list(range(12))
    best_exact_size = _evaluate_exact(best_arr)
    
    random.seed(42)
    np.random.seed(42)

    # Key insight: arithmetic progressions create many collisions
    # {a, a+d, a+2d, ..., a+11d} has many duplicate sums
    # Also: union of multiple APs with same difference
    
    # Generate structured candidates
    structured_candidates = []
    
    # Pure arithmetic progressions
    for a in range(0, 129 - 11 * 1):
        for d in range(1, 12):
            arr = [a + i * d for i in range(12)]
            if max(arr) <= 128 and min(arr) >= 0 and len(set(arr)) == 12:
                structured_candidates.append(arr)
    
    # Two interleaved APs
    for a in range(0, 100):
        for d in range(2, 15):
            arr1 = [a + i * 2 * d for i in range(6)]
            arr2 = [a + d + i * 2 * d for i in range(6)]
            arr = sorted(arr1 + arr2)
            if max(arr) <= 128 and min(arr) >= 0 and len(set(arr)) == 12:
                structured_candidates.append(arr)
    
    # Three interleaved APs
    for a in range(0, 100):
        for d in range(3, 12):
            arr = []
            for r in range(3):
                arr.extend([a + r + i * 3 * d for i in range(4)])
            arr = sorted(arr)
            if max(arr) <= 128 and min(arr) >= 0 and len(set(arr)) == 12:
                structured_candidates.append(arr)
    
    # Four interleaved APs
    for a in range(0, 100):
        for d in range(4, 10):
            arr = []
            for r in range(4):
                arr.extend([a + r + i * 4 * d for i in range(3)])
            arr = sorted(arr)
            if max(arr) <= 128 and min(arr) >= 0 and len(set(arr)) == 12:
                structured_candidates.append(arr)
    
    # Six pairs (two elements each, close together)
    for a in range(0, 100):
        for d in range(5, 15):
            for gap in range(1, 4):
                arr = []
                for i in range(6):
                    arr.extend([a + i * d, a + i * d + gap])
                arr = sorted(arr)
                if max(arr) <= 128 and min(arr) >= 0 and len(set(arr)) == 12:
                    structured_candidates.append(arr)
    
    # Evaluate structured candidates
    for arr in structured_candidates:
        if time.time() > deadline:
            break
        exact_size = _evaluate_exact(arr)
        if exact_size < best_exact_size:
            best_exact_size = exact_size
            best_arr = arr[:]
            if best_exact_size <= 4:
                return best_arr

    # Random + local search phase
    for restart in range(2000):
        if time.time() > deadline:
            break

        strategy = restart % 4
        if strategy == 0:
            arr = sorted(random.sample(range(129), 12))
        elif strategy == 1:
            # Perturbed arithmetic progression
            a = random.randint(0, 50)
            d = random.randint(2, 10)
            arr = [a + i * d for i in range(12)]
            # Perturb some elements
            for _ in range(random.randint(1, 4)):
                idx = random.randint(0, 11)
                arr[idx] += random.randint(-2, 2)
            arr = sorted(set(arr))
            if len(arr) < 12:
                # Fill in missing values
                while len(arr) < 12:
                    v = random.randint(0, 128)
                    if v not in arr:
                        arr.append(v)
                arr = sorted(arr[:12])
        elif strategy == 2:
            # Clusters
            base = random.randint(0, 60)
            gaps = sorted(random.sample(range(5, 25), 3))
            arr = []
            for i, g in enumerate(gaps):
                offset = base + sum(gaps[:i])
                arr.extend([offset, offset + random.randint(1, 3), offset + random.randint(4, 6)])
            arr = sorted(set(arr))
            while len(arr) < 12:
                v = random.randint(0, 128)
                if v not in arr:
                    arr.append(v)
            arr = sorted(arr[:12])
        else:
            # Random with clustering bias
            arr = sorted(random.sample(range(129), 12))
            for _ in range(5):
                i = random.randint(0, 11)
                j = random.randint(0, 11)
                if i != j:
                    diff = arr[j] - arr[i]
                    if 0 < abs(diff) <= 8:
                        new_val = arr[i] + diff
                        if 0 <= new_val <= 128 and new_val not in arr:
                            arr[i] = new_val
                            arr = sorted(arr)

        if len(set(arr)) != 12 or min(arr) < 0 or max(arr) > 128:
            continue

        # Local search to maximize collisions
        current_collisions = _collision_count(arr)
        no_improve = 0
        for step in range(300):
            if time.time() > deadline:
                break

            i = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val in arr:
                continue
            new_arr = arr.copy()
            new_arr[i] = new_val
            new_arr = sorted(new_arr)
            
            if len(set(new_arr)) != 12:
                continue

            new_collisions = _collision_count(new_arr)
            if new_collisions > current_collisions:
                arr = new_arr
                current_collisions = new_collisions
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 80:
                    break

        # Evaluate exact
        exact_size = _evaluate_exact(arr)
        if exact_size < best_exact_size:
            best_exact_size = exact_size
            best_arr = arr
            if best_exact_size <= 4:
                return best_arr

    # Final polish: targeted local search on best solution
    if best_exact_size > 4 and time.time() < deadline - 5:
        arr = best_arr.copy()
        current_exact = best_exact_size
        no_improve = 0
        for step in range(1000):
            if time.time() > deadline - 3:
                break
            
            i = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val in arr:
                continue
            new_arr = arr.copy()
            new_arr[i] = new_val
            new_arr = sorted(new_arr)
            
            if len(set(new_arr)) != 12:
                continue
            
            new_exact = _evaluate_exact(new_arr)
            if new_exact < current_exact:
                arr = new_arr
                current_exact = new_exact
                no_improve = 0
                if current_exact < best_exact_size:
                    best_exact_size = current_exact
                    best_arr = arr
                    if best_exact_size <= 4:
                        return best_arr
            else:
                no_improve += 1
                if no_improve > 150:
                    break

    return best_arr
# EVOLVE-BLOCK-END
