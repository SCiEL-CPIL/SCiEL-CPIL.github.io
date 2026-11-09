import random
import time
import sys

L = 5354228880  # lcm(1..24)

def subset_sum_exists(weights, target):
    """
    Check if target can be formed by a subset of weights.
    Uses meet-in-the-middle.
    """
    if target == 0:
        return True
    if not weights:
        return False
    
    # Filter weights > target (positive weights only)
    weights = [w for w in weights if w <= target]
    if not weights:
        return False
        
    n = len(weights)
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]
    
    left_sums = {0}
    for w in left:
        new_sums = set()
        for s in left_sums:
            if s + w <= target:
                new_sums.add(s + w)
        left_sums.update(new_sums)
        
    right_sums = {0}
    for w in right:
        new_sums = set()
        for s in right_sums:
            if s + w <= target:
                new_sums.add(s + w)
        right_sums.update(new_sums)
        
    for s in left_sums:
        if (target - s) in right_sums:
            return True
    return False

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
            new_S = S + [x]
            if is_valid(new_S):
                S = new_S
                S_set.add(x)
        return S
    
    # Try several deterministic orders
    orders = [
        list(range(24, 0, -1)),  # Descending
        list(range(1, 25)),      # Ascending
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        # Interleaved large/small
        [12, 13, 11, 14, 10, 15, 9, 16, 8, 17, 7, 18, 6, 19, 5, 20, 4, 21, 3, 22, 2, 23, 1, 24],
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        S = greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Local search: try to improve by swapping or adding
    def local_search(S, max_iterations=100):
        """Try to improve set S by adding/removing/swapping elements."""
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
                new_S = S + [m]
                if is_valid(new_S):
                    S = new_S
                    S_set.add(m)
                    improved = True
            
            # Try removing one element and adding two others (if possible)
            if not improved and len(S) >= 3:
                # Try to find a swap that increases size
                for i in range(len(S)):
                    if time.time() > deadline:
                        return S
                    for j in range(i + 1, len(S)):
                        if time.time() > deadline:
                            return S
                        # Remove S[i] and S[j], try to add two missing elements
                        S_without = [x for x in S if x != S[i] and x != S[j]]
                        S_without_set = set(S_without)
                        candidates = [k for k in range(1, 25) if k not in S_without_set and k != S[i] and k != S[j]]
                        random.shuffle(candidates)
                        for c1 in candidates[:10]:
                            if time.time() > deadline:
                                return S
                            for c2 in candidates[:10]:
                                if c1 == c2:
                                    continue
                                new_S = S_without + [c1, c2]
                                if is_valid(new_S):
                                    S = new_S
                                    S_set = set(S)
                                    improved = True
                                    break
                            if improved:
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
            improved_set = local_search(S, max_iterations=50)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set
        
        if best_size == 24:
            break
    
    if best_set is None:
        return [1]
        
    return best_set
