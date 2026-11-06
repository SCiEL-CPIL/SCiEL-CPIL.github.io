"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

_DEADLINE = time.time() + 80.0  # 80s safety margin under 90s kill

def _is_sidon(subset):
    """Check if subset has all pair sums (including diagonals) unique."""
    n = len(subset)
    sums = []
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            sums.append(s)
    return len(sums) == len(set(sums))

def _max_sidon_size_brute(arr):
    """Find largest Sidon subset size by checking all subsets from largest down."""
    n = len(arr)
    # Try sizes from n down to 1
    for size in range(n, 0, -1):
        for combo in itertools.combinations(arr, size):
            if _is_sidon(combo):
                return size
    return 0

def _max_sidon_size_fast(arr):
    """Faster check: greedy + verification approach for small sets."""
    n = len(arr)
    # First check if the whole set is Sidon
    if _is_sidon(arr):
        return n
    # Check all subsets of size n-1
    for i in range(n):
        sub = [arr[j] for j in range(n) if j != i]
        if _is_sidon(sub):
            return n - 1
    # Check all subsets of size n-2
    for i in range(n):
        for j in range(i+1, n):
            sub = [arr[k] for k in range(n) if k != i and k != j]
            if _is_sidon(sub):
                return n - 2
    # Check all subsets of size n-3
    for combo in itertools.combinations(range(n), n-3):
        sub = [arr[i] for i in combo]
        if _is_sidon(sub):
            return n - 3
    # Check all subsets of size n-4
    for combo in itertools.combinations(range(n), n-4):
        sub = [arr[i] for i in combo]
        if _is_sidon(sub):
            return n - 4
    # Fallback: brute force from size 5 down
    for size in range(5, 0, -1):
        for combo in itertools.combinations(arr, size):
            if _is_sidon(combo):
                return size
    return 1

def construct():
    best_set = None
    best_size = 12  # worst case
    
    # Strategy: search small ranges where sum collisions are forced
    # For 12 elements in [0, R], the number of possible pair sums is (R+1)*12/2
    # but we need C(12,2)+12 = 78 unique sums. So we need enough room.
    # In [0, 25], sums range from 0 to 50, giving 51 possible values.
    # 78 sums into 51 values guarantees collisions -> smaller Sidon subsets.
    
    # First, try some hand-crafted dense sets in small ranges
    candidate_sets = []
    
    # Generate random sets in small ranges
    random.seed(42)
    
    # Try ranges [0, 20] through [0, 30]
    for max_val in range(20, 31):
        # Generate many random 12-element subsets
        for _ in range(500):
            if time.time() > _DEADLINE:
                break
            s = random.sample(range(max_val + 1), 12)
            s.sort()
            candidate_sets.append(s)
        if time.time() > _DEADLINE:
            break
    
    # Also try structured sets: consecutive blocks with perturbations
    for base in range(0, 15):
        if time.time() > _DEADLINE:
            break
        # Try base + [0..11] with some swaps
        s = list(range(base, base + 12))
        if s[-1] <= 128:
            candidate_sets.append(s)
        # Try with some elements shifted
        for shift_pattern in [[0,0,0,0,0,0,1,1,1,1,1,1],
                              [0,0,1,1,1,1,1,1,1,2,2,2],
                              [0,1,1,1,2,2,2,2,3,3,3,3],
                              [0,0,0,1,1,1,1,2,2,2,3,3]]:
            s = [base + p for p in shift_pattern]
            if len(set(s)) == 12 and max(s) <= 128:
                candidate_sets.append(s)
    
    # Evaluate candidates, keeping track of best
    evaluated = set()
    for s in candidate_sets:
        if time.time() > _DEADLINE:
            break
        key = tuple(s)
        if key in evaluated:
            continue
        evaluated.add(key)
        
        size = _max_sidon_size_fast(s)
        if size < best_size:
            best_size = size
            best_set = s
            if best_size <= 4:
                # Can't do better than 4 easily; try to find 3
                pass
    
    # If we found size 5, try harder to find size 4
    if best_size == 5:
        # More intensive search
        for max_val in range(18, 26):
            for _ in range(2000):
                if time.time() > _DEADLINE:
                    break
                s = random.sample(range(max_val + 1), 12)
                s.sort()
                key = tuple(s)
                if key in evaluated:
                    continue
                evaluated.add(key)
                size = _max_sidon_size_fast(s)
                if size < best_size:
                    best_size = size
                    best_set = s
    
    if best_set is None:
        # Fallback: dense set in [0, 25]
        best_set = list(range(12))
    
    return best_set
# EVOLVE-BLOCK-END
