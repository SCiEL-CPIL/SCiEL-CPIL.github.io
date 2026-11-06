"""Sidon set construction: maximizing size in [1,284] containing {1,2,5,11}."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_valid_sidon(arr):
    """Check if sorted list arr forms a valid Sidon set (all pairwise sums distinct)."""
    n = len(arr)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_construct(starting_set, rng):
    """
    Greedy construction: start with starting_set, then greedily add elements
    from 1..284 in a randomized order, keeping only those that don't create
    sum conflicts.
    """
    arr = list(starting_set)
    sums = set()
    for i in range(len(arr)):
        for j in range(i, len(arr)):
            sums.add(arr[i] + arr[j])
    
    # Candidates: all numbers 1..284 not already in arr
    candidates = [x for x in range(1, 285) if x not in arr]
    rng.shuffle(candidates)
    
    for x in candidates:
        # Check if adding x creates any conflict
        new_sums = {x + y for y in arr} | {2 * x}
        if len(new_sums) == len(arr) + 1 and not new_sums & sums:
            arr.append(x)
            sums |= new_sums
    
    return sorted(arr)

def _local_search(arr, rng, max_iter=500):
    """
    Local search: try to swap one element out and another in to improve or maintain size.
    Actually, since we want to maximize size, we try to add elements by removing
    conflicting ones. This is a destroy-and-repair approach.
    """
    current = sorted(arr)
    best = sorted(current)
    best_len = len(best)
    
    def get_sums(a):
        s = set()
        for i in range(len(a)):
            for j in range(i, len(a)):
                s.add(a[i] + a[j])
        return s
    
    for _ in range(max_iter):
        # Try to add a random element
        current_set = set(current)
        candidates = [x for x in range(1, 285) if x not in current_set]
        if not candidates:
            break
        
        x = rng.choice(candidates)
        
        # Find all elements that conflict with x
        # A conflict occurs if x + y1 == x + y2 (impossible since y1 != y2) 
        # or x + y == z + w for existing z,w in current
        # More precisely: adding x creates conflict if:
        #   - 2*x is already in sums (i.e., there exist a,b in current with a+b=2x)
        #   - x + y is already in sums for some y in current (i.e., exist a,b in current with a+b = x+y)
        
        cur_sums = get_sums(current)
        
        # Check conflicts
        conflicts = set()
        if 2 * x in cur_sums:
            # Find pairs that sum to 2x
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    if current[i] + current[j] == 2 * x:
                        conflicts.add(current[i])
                        conflicts.add(current[j])
        
        for y in current:
            s = x + y
            if s in cur_sums:
                # Find pairs that sum to s
                for i in range(len(current)):
                    for j in range(i, len(current)):
                        if current[i] + current[j] == s:
                            conflicts.add(current[i])
                            conflicts.add(current[j])
        
        if not conflicts:
            # Can add x directly
            new_arr = sorted(current + [x])
            if len(new_arr) > best_len:
                best = new_arr
                best_len = len(best)
            current = new_arr
        else:
            # Try to remove the smallest conflict set
            # Simple heuristic: remove one random conflicting element, then try to add x
            conflict_list = list(conflicts)
            to_remove = rng.choice(conflict_list)
            new_arr = [v for v in current if v != to_remove]
            new_arr.append(x)
            new_arr = sorted(new_arr)
            
            if _is_valid_sidon(new_arr) and len(new_arr) >= len(current):
                current = new_arr
                if len(current) > best_len:
                    best = current
                    best_len = len(best)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    rng = random.Random(42)
    
    # Start with prescribed elements
    base = [1, 2, 5, 11]
    
    # First, get a good greedy solution
    best = _greedy_construct(base, rng)
    best_len = len(best)
    
    # Now do multiple rounds of randomized greedy + local search
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Randomized greedy with different random seeds/orders
        # Sometimes start fresh, sometimes perturb
        if iteration % 3 == 0:
            # Perturb best: remove a random element, then re-greedy
            if len(best) > len(base):
                arr = list(best)
                # Remove a non-prescribed element
                removable = [x for x in arr if x not in base]
                if removable:
                    to_remove = rng.choice(removable)
                    arr.remove(to_remove)
                    # Re-greedy from here
                    candidate_set = set(arr)
                    candidates = [x for x in range(1, 285) if x not in candidate_set]
                    rng.shuffle(candidates)
                    sums = set()
                    for i in range(len(arr)):
                        for j in range(i, len(arr)):
                            sums.add(arr[i] + arr[j])
                    for x in candidates:
                        new_sums = {x + y for y in arr} | {2 * x}
                        if len(new_sums) == len(arr) + 1 and not new_sums & sums:
                            arr.append(x)
                            sums |= new_sums
                    new_solution = sorted(arr)
                else:
                    new_solution = best
            else:
                new_solution = best
        else:
            new_solution = _greedy_construct(base, rng)
        
        # Local search on the result
        if time.time() < deadline - 5:
            improved = _local_search(new_solution, rng, max_iter=200)
            if len(improved) > best_len:
                best = improved
                best_len = len(best)
        
        # Also try local search on best itself occasionally
        if iteration % 5 == 0 and time.time() < deadline - 5:
            improved = _local_search(best, rng, max_iter=100)
            if len(improved) > best_len:
                best = improved
                best_len = len(best)
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return sorted(best)

# EVOLVE-BLOCK-END
