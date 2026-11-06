import time
import random
from math import gcd

_START_TIME = None
_TIME_LIMIT = 85.0

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) > _TIME_LIMIT

# Precompute LCM of 1..24
def _lcm(a, b):
    return a * b // gcd(a, b)

_L = 1
for i in range(1, 25):
    _L = _lcm(_L, i)

# v(i) = _L // i
_V = {i: _L // i for i in range(1, 25)}

def _subset_sums(items):
    """
    Returns a set of all possible subset sums of the given list of integers.
    """
    sums = {0}
    for item in items:
        # Create a list of new sums to avoid modifying set while iterating
        # Optimization: if item is 0, skip (not possible here)
        new_sums = [s + item for s in sums]
        sums.update(new_sums)
    return sums

def _can_add(x, S):
    """
    Check if x can be added to set S (list of ints) without violating constraints.
    """
    if x in S:
        return False
        
    vx = _V[x]
    
    # Get values of elements in S
    vals_S = [_V[y] for y in S]
    
    # Check 1: 1/x is not a subset sum of S
    # i.e., vx is not a subset sum of vals_S
    if S:
        sums_S = _subset_sums(vals_S)
        if vx in sums_S:
            return False
        
        # Check 2: For each y in S, 1/y is not a subset sum of (S \ {y}) U {x}
        # 1/y = 1/x + sum(1/z for z in S' subset of S\{y,x})
        # 1/y - 1/x = sum(1/z ...)
        # vy - vx = sum(v(z) for z in S' subset of S\{y,x})
        # Note: We only need to check if vy > vx (y < x), else LHS is negative and RHS is positive (or 0).
        for y in S:
            if y < x:
                target = _V[y] - vx
                if target <= 0:
                    continue
                # Items are values of S excluding y
                # We can reuse sums_S? No, sums_S includes y.
                # We need sums of S \ {y}.
                # Since |S| is small, we recompute.
                # Optimization: If we had sums of S\{y} precomputed, it'd be faster.
                # But for now, recompute.
                vals_S_minus_y = [_V[z] for z in S if z != y]
                sums_S_minus_y = _subset_sums(vals_S_minus_y)
                if target in sums_S_minus_y:
                    return False
    return True

def _greedy_initial():
    """
    Greedy strategy: Try multiple orders.
    """
    best_S = []
    # Orders: Large to small, Small to Large, Random
    orders = [
        list(range(24, 0, -1)),
        list(range(1, 25)),
    ]
    # Add a few random orders
    for _ in range(5):
        order = list(range(1, 25))
        random.shuffle(order)
        orders.append(order)
    
    for order in orders:
        S = []
        for x in order:
            if _time_exceeded():
                break
            if _can_add(x, S):
                S.append(x)
        if len(S) > len(best_S):
            best_S = S[:]
    return best_S

def _local_search(initial_S):
    """
    Improve the set using local search (add/swap).
    """
    S = list(initial_S)
    S_set = set(S)
    best_S = S[:]
    best_size = len(S)
    
    # Precompute all possible candidates not in S
    # We will iterate until no improvement or time limit
    
    while not _time_exceeded():
        improved = False
        
        # 1. Try to add any element
        # Shuffle candidates to avoid bias
        candidates = [x for x in range(1, 25) if x not in S_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if _time_exceeded():
                break
            if _can_add(x, S):
                S.append(x)
                S_set.add(x)
                improved = True
                if len(S) > best_size:
                    best_S = S[:]
                    best_size = len(S)
                break # Restart loop to re-evaluate with new S
        
        if improved:
            continue
            
        # 2. Try swaps: remove y, add x
        # For each y in S, for each x not in S, try swap
        # This is O(N^2) checks.
        # We can be smarter: 
        # If we remove y, S' = S \ {y}.
        # Can we add x to S'?
        
        # Shuffle S and candidates
        S_list = list(S)
        random.shuffle(S_list)
        
        for y in S_list:
            if _time_exceeded():
                break
            
            S_minus_y = [z for z in S if z != y]
            S_minus_y_set = set(S_minus_y)
            
            # Try to add any x not in S_minus_y
            candidates = [x for x in range(1, 25) if x not in S_minus_y_set]
            random.shuffle(candidates)
            
            for x in candidates:
                if x == y:
                    continue
                if _time_exceeded():
                    break
                # Check if x can be added to S_minus_y
                # Note: _can_add(x, S_minus_y) checks if x conflicts with S_minus_y.
                # But we also need to ensure that removing y doesn't create new conflicts?
                # No, removing an element cannot create new conflicts for the remaining elements.
                # The only new conflict is if x itself creates a conflict with S_minus_y.
                # Wait, the condition is symmetric.
                # If S was valid, S\{y} is valid.
                # We just need to check if S\{y} U {x} is valid.
                # This is exactly what _can_add(x, S_minus_y) does.
                
                if _can_add(x, S_minus_y):
                    # Perform swap
                    S.remove(y)
                    S_set.remove(y)
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    if len(S) > best_size:
                        best_S = S[:]
                        best_size = len(S)
                    break # Restart loop
            
            if improved:
                break
        
        if not improved:
            # No single add or swap improves the size.
            # Try to find a different local minimum by random restart or perturbation?
            # Or just break if we've tried enough.
            # Let's try a random restart with a subset of the current best.
            if random.random() < 0.1:
                # Perturb: remove 2 random elements, then try to fill
                if len(S) > 2:
                    to_remove = random.sample(S, 2)
                    for y in to_remove:
                        S.remove(y)
                        S_set.remove(y)
                    # Continue loop to try adding
            else:
                break # Stuck in local minimum, break to try different strategy or exit

    return best_S

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # 1. Greedy Initial
    best_set = _greedy_initial()
    best_size = len(best_set)
    
    # 2. Local Search
    if not _time_exceeded():
        improved_set = _local_search(best_set)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
    
    # 3. If still time, try more random restarts with greedy + local search
    while not _time_exceeded() and best_size < 20: # 20 is a reasonable upper bound to try for
        # Random initial set
        S = []
        # Pick random subset size ~ best_size
        target_size = max(5, best_size - 2)
        S = random.sample(range(1, 25), target_size)
        # Validate and repair
        valid_S = []
        for x in S:
            if _can_add(x, valid_S):
                valid_S.append(x)
        
        # Local search on this
        if not _time_exceeded():
            improved = _local_search(valid_S)
            if len(improved) > best_size:
                best_set = improved
                best_size = len(best_set)

    if not best_set:
        return [24]
        
    return best_set
