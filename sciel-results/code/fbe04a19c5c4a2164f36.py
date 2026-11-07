import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 seconds margin before 90s kill
    
    # Initial best set: a known valid set of size 3
    best_set = [11, 13, 17]
    best_size = len(best_set)
    
    # Helper to compute all non-empty subset sums
    def get_sums(S):
        sums = set()
        n = len(S)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += S[i]
            sums.add(s)
        return sums
    
    # Check if we can add x to S with current sums
    def can_add(S, x, current_sums):
        # Generate new sums involving x
        new_sums = set()
        new_sums.add(x)
        for s in current_sums:
            new_sums.add(x + s)
            
        # Check for overlap (duplicate sums)
        if current_sums & new_sums:
            return False, None
            
        # Check divisibility
        # We only need to check pairs where at least one sum is in new_sums
        # 1. new_sums vs current_sums
        for s_new in new_sums:
            for s_old in current_sums:
                if s_new > s_old:
                    if s_new % s_old == 0:
                        return False, None
                else:
                    if s_old % s_new == 0:
                        return False, None
                        
        # 2. new_sums vs new_sums
        new_sums_list = list(new_sums)
        for i in range(len(new_sums_list)):
            for j in range(i + 1, len(new_sums_list)):
                a, b = new_sums_list[i], new_sums_list[j]
                if a > b:
                    if a % b == 0:
                        return False, None
                else:
                    if b % a == 0:
                        return False, None
                            
        return True, new_sums

    # Improved: use a priority queue / greedy with multiple strategies
    
    # Strategy 1: Greedy from small to large
    def greedy_search(candidates, order_func):
        current_set = []
        current_sums = set()
        for x in order_func(candidates):
            valid, new_sums = can_add(current_set, x, current_sums)
            if valid:
                current_set.append(x)
                current_sums = current_sums.union(new_sums)
        return current_set

    # Strategy 2: Randomized greedy with restarts
    candidates = list(range(1, 257))
    
    max_restarts = 5000
    restarts_done = 0
    
    for _ in range(max_restarts):
        if time.time() > deadline:
            break
        restarts_done += 1
            
        random.shuffle(candidates)
        current_set = []
        current_sums = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
                
            valid, new_sums = can_add(current_set, x, current_sums)
            if valid:
                current_set.append(x)
                current_sums = current_sums.union(new_sums)
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
                    
        # If we found a good set, try to extend it with remaining candidates
        if len(current_set) >= best_size and len(current_set) > 0:
            # Try adding more elements to the current best
            used = set(current_set)
            remaining = [x for x in range(1, 257) if x not in used]
            random.shuffle(remaining)
            
            extended = current_set.copy()
            ext_sums = current_sums.copy()
            
            for x in remaining:
                if time.time() > deadline:
                    break
                valid, new_sums = can_add(extended, x, ext_sums)
                if valid:
                    extended.append(x)
                    ext_sums = ext_sums.union(new_sums)
                    if len(extended) > best_size:
                        best_size = len(extended)
                        best_set = extended.copy()
                    
    # Also try a systematic approach: start with small numbers
    # Sort by some heuristic and greedily add
    for strategy in range(10):
        if time.time() > deadline:
            break
        
        if strategy == 0:
            # Ascending
            ordered = sorted(candidates)
        elif strategy == 1:
            # Descending
            ordered = sorted(candidates, reverse=True)
        elif strategy == 2:
            # Primes first
            def is_prime(n):
                if n < 2: return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0: return False
                return True
            ordered = sorted(candidates, key=lambda x: (not is_prime(x), x))
        elif strategy == 3:
            # Odd first
            ordered = sorted(candidates, key=lambda x: (x % 2 == 0, x))
        else:
            # Random
            ordered = candidates.copy()
            random.shuffle(ordered)
            
        current_set = []
        current_sums = set()
        for x in ordered:
            if time.time() > deadline:
                break
            valid, new_sums = can_add(current_set, x, current_sums)
            if valid:
                current_set.append(x)
                current_sums = current_sums.union(new_sums)
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
    
    # Try local search on the best set: try swapping elements
    if time.time() < deadline:
        best_set_copy = best_set.copy()
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Try removing one element and adding another
            for i in range(len(best_set_copy)):
                if time.time() > deadline:
                    break
                trial = best_set_copy[:i] + best_set_copy[i+1:]
                trial_sums = get_sums(trial)
                used = set(trial)
                
                for x in range(1, 257):
                    if x in used:
                        continue
                    if time.time() > deadline:
                        break
                    valid, new_sums = can_add(trial, x, trial_sums)
                    if valid:
                        new_set = trial + [x]
                        # Verify this is actually better or same size
                        if len(new_set) > len(best_set_copy):
                            best_set_copy = new_set
                            if len(best_set_copy) > best_size:
                                best_size = len(best_set_copy)
                                best_set = best_set_copy.copy()
                            break
    
    # Ensure return type is list of ints
    return [int(x) for x in best_set]
