"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s buffer for safety within 120s limit
    
    prescribed = [1, 2, 5, 11]
    fixed_set = set(prescribed)
    
    # Helper to check if a list is a valid Sidon set
    # Returns (is_valid, conflict_count)
    def check_sidon(candidates):
        n = len(candidates)
        if n != len(set(candidates)):
            return False, 100000
        sums = set()
        conflict = 0
        for i in range(n):
            for j in range(i, n):
                s = candidates[i] + candidates[j]
                if s in sums:
                    conflict += 1
                else:
                    sums.add(s)
        return conflict == 0, conflict

    # 1. Greedy Construction to get a strong baseline
    a = list(prescribed)
    sums = set()
    for i, x in enumerate(a):
        for y in a[i:]:
            sums.add(x + y)
    
    for x in range(12, 285):
        new = {x + y for y in a} | {2 * x}
        if len(new) == len(a) + 1 and not new & sums:
            a.append(x)
            sums |= new
            
    best_set = a
    best_size = len(best_set)
    
    # If we already hit the theoretical max (20), return immediately
    if best_size == 20:
        return best_set

    # 2. CSP with Forward Checking to find larger sets
    # This approach uses constraint propagation to prune incompatible candidates early.
    
    def csp_find_solution(target_size, time_limit):
        nonlocal start_time, deadline
        end_time = start_time + time_limit
        if end_time > deadline:
            end_time = deadline - 0.5
            
        if time.time() >= end_time:
            return None
            
        # Start with the prescribed elements
        current = list(prescribed)
        current_sums = set()
        for i, x in enumerate(current):
            for y in current[i:]:
                current_sums.add(x + y)
                
        # Candidate pool: all numbers 1..284 not in current
        candidates = [x for x in range(1, 285) if x not in fixed_set]
        
        # We want to add (target_size - len(current)) more elements
        needed = target_size - len(current)
        
        # Backtracking search with forward checking
        # State: (current_set, current_sums, candidates_list, depth)
        # We'll use a simple recursive backtracking with pruning
        
        best_found = None
        
        def backtrack(current_set, current_sums, candidates, depth, max_depth):
            nonlocal best_found, end_time
            
            if time.time() > end_time:
                return None
                
            if depth == max_depth:
                # Check if valid (should be by construction)
                return list(current_set)
                
            # Try each candidate
            for i in range(len(candidates)):
                x = candidates[i]
                
                # Check if x can be added without conflicts
                new_sums = set()
                valid = True
                for y in current_set:
                    s = x + y
                    if s in current_sums:
                        valid = False
                        break
                    new_sums.add(s)
                s = 2 * x
                if s in current_sums:
                    valid = False
                else:
                    new_sums.add(s)
                    
                if not valid:
                    continue
                    
                # Forward checking: prune candidates that would conflict with x
                new_candidates = []
                for j in range(len(candidates)):
                    if j == i:
                        continue
                    y = candidates[j]
                    # Check if y conflicts with x
                    conflict = False
                    for z in current_set:
                        if (x + y) in current_sums or (x + z) in current_sums or (y + z) in current_sums:
                            # More precise check: does adding y create a duplicate sum?
                            pass
                    # Simpler: just check if y + x or 2*y creates a conflict with existing sums or x's new sums
                    # Actually, forward checking should check if y is still compatible with the new set
                    # For efficiency, we'll just filter out y if it directly conflicts with x
                    if (x + y) in current_sums or (2 * y) in current_sums or (x + y) in new_sums or (2 * y) in new_sums:
                        conflict = True
                    if not conflict:
                        new_candidates.append(y)
                        
                # Recursive call
                new_set = list(current_set) + [x]
                new_sums_set = current_sums | new_sums
                result = backtrack(new_set, new_sums_set, new_candidates, depth + 1, max_depth)
                if result is not None:
                    return result
                    
            return None
            
        # To make this more efficient, we can use a heuristic ordering of candidates
        # and limit the branching factor
        
        # Simple approach: try adding elements one by one with backtracking
        # But this might be too slow for large target sizes.
        
        # Alternative: use a randomized restart strategy with CSP
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > end_time:
                break
                
            # Shuffle candidates for diversity
            shuffled_candidates = candidates[:]
            random.shuffle(shuffled_candidates)
            
            # Try to build a solution of size target_size
            current_set = list(prescribed)
            current_sums = set()
            for i, x in enumerate(current_set):
                for y in current_set[i:]:
                    current_sums.add(x + y)
                    
            success = True
            for depth in range(needed):
                found = False
                for x in shuffled_candidates:
                    if x in current_set:
                        continue
                        
                    # Check if x can be added
                    new_sums = set()
                    valid = True
                    for y in current_set:
                        s = x + y
                        if s in current_sums:
                            valid = False
                            break
                        new_sums.add(s)
                    s = 2 * x
                    if s in current_sums:
                        valid = False
                    else:
                        new_sums.add(s)
                        
                    if valid:
                        current_set.append(x)
                        current_sums |= new_sums
                        found = True
                        break
                        
                if not found:
                    success = False
                    break
                    
            if success and len(current_set) == target_size:
                # Verify
                is_valid, _ = check_sidon(current_set)
                if is_valid:
                    return current_set
                    
        return None

    # Try to reach 20 using CSP
    if best_size < 20:
        res = csp_find_solution(20, 40)
        if res:
            return res

    # If CSP fails, fall back to simulated annealing
    def try_find_solution_seeded(target_size, time_limit, seed_set):
        nonlocal start_time, deadline
        end_time = start_time + time_limit
        if end_time > deadline:
            end_time = deadline - 0.5
            
        if time.time() >= end_time:
            return None
            
        fixed_elements = set(prescribed)
        
        max_restarts = 20
        restarts = 0
        
        while time.time() < end_time and restarts < max_restarts:
            restarts += 1
            
            if len(seed_set) > target_size:
                return None
                
            current_set = list(seed_set)
            needed = target_size - len(current_set)
            
            if needed < 0:
                return None
                
            pool = [x for x in range(1, 285) if x not in set(current_set)]
            random.shuffle(pool)
            current_set.extend(pool[:needed])
            random.shuffle(current_set)
            
            is_fixed = [x in fixed_elements for x in current_set]
            
            is_valid, current_conflict = check_sidon(current_set)
            
            if is_valid:
                return current_set
            
            temp = 5.0
            min_temp = 0.01
            cooling_rate = 0.9995
            max_steps = 200000
            
            for step in range(max_steps):
                if time.time() > end_time:
                    break
                    
                if temp < min_temp:
                    break
                    
                idx = random.randint(0, len(current_set) - 1)
                
                if is_fixed[idx]:
                    continue
                    
                old_val = current_set[idx]
                new_val = random.randint(1, 284)
                
                if new_val == old_val:
                    continue
                    
                current_set[idx] = new_val
                
                is_valid, new_conflict = check_sidon(current_set)
                
                delta = new_conflict - current_conflict
                
                if delta <= 0:
                    current_conflict = new_conflict
                    if is_valid:
                        return current_set
                else:
                    try:
                        prob = math.exp(-delta / temp)
                    except OverflowError:
                        prob = 0.0
                    
                    if random.random() < prob:
                        current_conflict = new_conflict
                    else:
                        current_set[idx] = old_val
                        
                temp *= cooling_rate

        return None

    # Try to reach 20 with SA if CSP failed
    if best_size < 20:
        res = try_find_solution_seeded(20, 50, best_set)
        if res:
            return res

    # If that failed, try to reach 19
    if best_size < 19:
        res = try_find_solution_seeded(19, 15, best_set)
        if res:
            return res

    # If that failed, try to reach 18
    if best_size < 18:
        res = try_find_solution_seeded(18, 5, best_set)
        if res:
            return res

    return best_set

# EVOLVE-BLOCK-END
