import time
import random
from itertools import combinations

def _is_sidon(subset):
    """Check if a subset has all unordered pair sums unique (including diagonals)."""
    n = len(subset)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _count_duplicate_sums(S):
    """Count the number of duplicate pair sums (higher = more collisions)."""
    n = len(S)
    sums = {}
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            sums[s] = sums.get(s, 0) + 1
    # Count excess: for each sum value with count c, it contributes c-1 to "collisions"
    # Total pairs - distinct sums = number of "wasted" pairs
    total_pairs = n * (n + 1) // 2
    distinct = len(sums)
    return total_pairs - distinct

def _max_sidon_size(S, target_size=None):
    """Find the largest Sidon subset size in S. If target_size given, check if size target_size exists."""
    n = len(S)
    if target_size is not None:
        # Check if there exists a Sidon subset of size target_size
        for combo in combinations(range(n), target_size):
            if _is_sidon([S[i] for i in combo]):
                return True
        return False
    else:
        # Find max
        for k in range(n, 0, -1):
            for combo in combinations(range(n), k):
                if _is_sidon([S[i] for i in combo]):
                    return k
        return 1

def _has_sidon_subset_of_size(S, k):
    """Check if S has a Sidon subset of size k."""
    n = len(S)
    for combo in combinations(range(n), k):
        if _is_sidon([S[i] for i in combo]):
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # First, verify seed
    seed_set = list(range(12))
    
    best_set = seed_set
    best_max_sidon = 4  # seed's value
    
    random.seed(42)
    
    # Strategy: Greedy swap to maximize duplicate sums
    # Start with random sets, then try swaps that increase collision count
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        # Generate random starting set
        S = sorted(random.sample(range(129), 12))
        current_collisions = _count_duplicate_sums(S)
        
        # Try greedy swaps to increase collisions
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try swapping each element with each possible value
            for i in range(12):
                if time.time() > deadline:
                    break
                old_val = S[i]
                for new_val in range(129):
                    if new_val == old_val:
                        continue
                    if new_val in S:
                        continue
                    # Try the swap
                    S[i] = new_val
                    S.sort()
                    new_collisions = _count_duplicate_sums(S)
                    if new_collisions > current_collisions:
                        current_collisions = new_collisions
                        improved = True
                    else:
                        # Revert
                        S[i] = old_val
                        S.sort()
        
        # Check max Sidon size for this set
        # First check if size 4 exists (if not, we found max=3)
        if not _has_sidon_subset_of_size(S, 4):
            best_set = S
            best_max_sidon = 3
            break
        else:
            # Check if size 5 exists
            if not _has_sidon_subset_of_size(S, 5):
                if best_max_sidon > 4:
                    best_set = S
                    best_max_sidon = 4
    
    if best_max_sidon == 3:
        return best_set
    
    # Try structured families more aggressively
    # Arithmetic progressions with large step
    for d in range(1, 12):
        if 11 * d > 128:
            break
        S = [i * d for i in range(12)]
        if not _has_sidon_subset_of_size(S, 4):
            best_set = S
            best_max_sidon = 3
            break
        elif not _has_sidon_subset_of_size(S, 5):
            if best_max_sidon > 4:
                best_set = S
                best_max_sidon = 4
    
    if best_max_sidon == 3:
        return best_set
    
    # Try: two interleaved APs
    for d1 in range(1, 6):
        for d2 in range(1, 6):
            for offset in range(0, 128, 3):
                if time.time() > deadline:
                    break
                S1 = [offset + i * d1 for i in range(6)]
                S2 = [offset + 1 + i * d2 for i in range(6)]
                S = S1 + S2
                if len(set(S)) != 12 or max(S) > 128:
                    continue
                if not _has_sidon_subset_of_size(S, 4):
                    best_set = S
                    best_max_sidon = 3
                    break
                elif not _has_sidon_subset_of_size(S, 5):
                    if best_max_sidon > 4:
                        best_set = S
                        best_max_sidon = 4
            if best_max_sidon == 3:
                break
        if best_max_sidon == 3:
            break
    
    if best_max_sidon == 3:
        return best_set
    
    return best_set
