import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit, leaving 35s for evaluator overhead
    
    best_set = []
    best_size = 0
    
    # Helper to check if adding c to current_set is valid
    def try_add(current_set, current_sums, c):
        added = Fraction(1, c)
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + added)
        if len(new_sums) == 2 * len(current_sums):
            return True, new_sums
        return False, None

    # Backtracking search with time limit
    def backtrack_search(remaining, current_set, current_sums, depth, max_depth):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        
        if depth >= max_depth or not remaining:
            return
        
        # Try adding each remaining element
        for i in range(len(remaining)):
            c = remaining[i]
            ok, new_sums = try_add(current_set, current_sums, c)
            if ok:
                new_remaining = remaining[:i] + remaining[i+1:]
                backtrack_search(new_remaining, current_set + [c], new_sums, depth + 1, max_depth)
                if time.time() > deadline:
                    return
            
            # If we've tried enough branches at this level, stop
            if depth > 8 and i > 5:
                break
    
    # 1. Initial Greedy (Sorted)
    current_set = []
    current_sums = {Fraction(0, 1)}
    for c in range(1, 23):
        ok, new_sums = try_add(current_set, current_sums, c)
        if ok:
            current_set.append(c)
            current_sums = new_sums
            
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set[:]

    # 2. Backtracking from greedy result - try to extend
    if best_set and time.time() < deadline - 10:
        remaining = [c for c in range(1, 23) if c not in best_set]
        backtrack_search(remaining, best_set[:], {Fraction(0, 1)}, 0, 12)
        # Recompute sums for best_set
        bs = {Fraction(0, 1)}
        for x in best_set:
            added = Fraction(1, x)
            new_bs = set(bs)
            for s in bs:
                new_bs.add(s + added)
            bs = new_bs
        # Try backtracking from this state
        if time.time() < deadline - 5:
            backtrack_search(remaining, best_set[:], bs, len(best_set), 14)

    # 3. Randomized Restarts with occasional backtracking
    while time.time() < deadline:
        candidates = list(range(1, 23))
        random.shuffle(candidates)
        
        current_set = []
        current_sums = {Fraction(0, 1)}
        
        for c in candidates:
            if time.time() > deadline:
                break
            
            ok, new_sums = try_add(current_set, current_sums, c)
            if ok:
                current_set.append(c)
                current_sums = new_sums
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
        
        # If we found a good set, try backtracking to extend it
        if len(current_set) >= best_size - 1 and time.time() < deadline - 3:
            remaining = [c for c in range(1, 23) if c not in current_set]
            # Use a smaller max_depth for speed
            backtrack_search(remaining, current_set[:], current_sums, len(current_set), min(len(current_set) + 3, 12))

    if best_set:
        return best_set
    else:
        return [1]
