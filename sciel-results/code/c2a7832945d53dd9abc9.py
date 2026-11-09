import random
import time
import sys
from functools import lru_cache

L = 5354228880  # lcm(1..24)

def subset_sum_exists(weights, target):
    """
    Check if target can be formed by a subset of weights using bitset DP.
    """
    if target == 0:
        return True
    if not weights:
        return False
    
    # Filter weights > target
    weights = [w for w in weights if w <= target]
    if not weights:
        return False
    
    # Use bitset: bit i is set if sum i is achievable
    # Start with bit 0 set
    bits = 1
    for w in weights:
        bits |= (bits << w)
        # Early exit if target bit is set
        if (bits >> target) & 1:
            return True
        # Mask to keep only bits up to target
        bits &= (1 << (target + 1)) - 1
    
    return ((bits >> target) & 1) == 1

def is_valid(S):
    """
    Check if S is a valid set.
    S is a list of ints.
    """
    if not S:
        return True
    S_set = set(S)
    for x in S:
        target = L // x
        weights = [L // y for y in S_set if y != x]
        if subset_sum_exists(weights, target):
            return False
    return True

def can_add(S, x):
    """
    Check if x can be added to set S while maintaining validity.
    Only need to check:
    1. For each y in S: L/y is not a subset sum of {L/z for z in S, z != y} ∪ {L/x}
       This means: is L/y a subset sum of {L/z for z in S, z != y} already? (checked before)
       OR is L/y = L/x + subset of {L/z for z in S, z != y, z != x}?
       Actually simpler: for each y in S, check if L/y can be formed by subset of {L/z : z in S∪{x}, z≠y}
    2. For x itself: L/x is not a subset sum of {L/z for z in S}
    """
    S_set = set(S)
    if x in S_set:
        return False
    
    # Check x's constraint: L/x not in subset sums of {L/z for z in S}
    target_x = L // x
    weights_x = [L // y for y in S_set]
    if subset_sum_exists(weights_x, target_x):
        return False
    
    # Check each existing element y's constraint with x added
    for y in S_set:
        target_y = L // y
        # Weights available for y: all L/z for z in S∪{x}, z != y
        weights_y = [L // z for z in S_set if z != y]
        weights_y.append(L // x)
        if subset_sum_exists(weights_y, target_y):
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    best_set = None
    best_size = 0
    
    # Initial guess: Seed solution
    seed_set = list(range(13, 25))
    if is_valid(seed_set):
        best_set = seed_set
        best_size = len(seed_set)
    
    # Deterministic greedy construction with multiple insertion orders
    def greedy_construct(order):
        """Build a valid set by greedily adding elements in the given order."""
        S = []
        S_set = set()
        for x in order:
            if x in S_set:
                continue
            if can_add(S, x):
                S.append(x)
                S_set.add(x)
        return S
    
    # Try several deterministic orders
    orders = [
        list(range(24, 0, -1)),  # Descending
        list(range(1, 25)),      # Ascending
        [12, 13, 11, 14, 10, 15, 9, 16, 8, 17, 7, 18, 6, 19, 5, 20, 4, 21, 3, 22, 2, 23, 1, 24],
        # Focus on larger numbers first (reciprocals are smaller, harder to sum)
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        # Try starting with mid-range
        [18, 19, 17, 20, 16, 21, 15, 22, 14, 23, 13, 24, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        S = greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Local search: try to improve by adding or swapping
    def local_search(S, max_iterations=50):
        """Try to improve set S by adding or swapping elements."""
        S = list(S)
        S_set = set(S)
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Try adding each missing element
            missing = [i for i in range(1, 25) if i not in S_set]
            random.shuffle(missing)
            for m in missing:
                if time.time() > deadline:
                    return S
                if can_add(S, m):
                    S.append(m)
                    S_set.add(m)
                    improved = True
            
            # Try 1-for-1 swaps: remove one, add one different
            if not improved and len(S) >= 1:
                for i in range(len(S)):
                    if time.time() > deadline:
                        return S
                    S_without = S[:i] + S[i+1:]
                    S_without_set = set(S_without)
                    candidates = [k for k in range(1, 25) if k not in S_without_set and k != S[i]]
                    random.shuffle(candidates)
                    for c in candidates[:15]:
                        if time.time() > deadline:
                            return S
                        if can_add(S_without, c):
                            S = S_without + [c]
                            S_set = set(S)
                            improved = True
                            break
                    if improved:
                        break
            
            # Try 2-for-2 swaps
            if not improved and len(S) >= 2:
                n = len(S)
                for i in range(n):
                    if time.time() > deadline:
                        return S
                    for j in range(i + 1, n):
                        if time.time() > deadline:
                            return S
                        S_without = [x for x in S if x != S[i] and x != S[j]]
                        S_without_set = set(S_without)
                        candidates = [k for k in range(1, 25) if k not in S_without_set and k != S[i] and k != S[j]]
                        random.shuffle(candidates)
                        found = False
                        for c1 in candidates[:12]:
                            if time.time() > deadline:
                                return S
                            for c2 in candidates[:12]:
                                if c1 == c2:
                                    continue
                                if can_add(S_without, c1) and can_add(S_without + [c1], c2):
                                    S = S_without + [c1, c2]
                                    S_set = set(S)
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
                        if improved:
                            break
                    if improved:
                        break
        return S
    
    # Apply local search to best set found so far
    if best_set is not None and time.time() < deadline:
        improved_set = local_search(best_set)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    # Random restarts with greedy + local search
    random.seed(42)
    attempts = 0
    while time.time() < deadline and best_size < 24:
        attempts += 1
        
        # Random order for greedy construction
        random_order = list(range(1, 25))
        random.shuffle(random_order)
        S = greedy_construct(random_order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
        
        # Local search on this set
        if time.time() < deadline:
            improved_set = local_search(S, max_iterations=30)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set
        
        if best_size == 24:
            break
    
    if best_set is None:
        return [1]
    
    # Final validation to ensure correctness
    if not is_valid(best_set):
        # Fallback to seed if somehow invalid
        return seed_set
    
    return best_set
