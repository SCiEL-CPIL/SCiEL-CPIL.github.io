import time
import random
import sys

# Increase recursion limit for deep backtracking
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin for 120s limit? 
    # The prompt says 150s wall-clock kill, evaluator 120s. 
    # Let's use 115s to be safe.
    deadline = start_time + 115.0
    
    initial_set = [1, 2, 5, 11]
    best_size = len(initial_set)
    best_set = list(initial_set)
    
    # Pre-calculate sums for initial set
    initial_sums = set()
    for i in range(len(initial_set)):
        for j in range(i, len(initial_set)):
            initial_sums.add(initial_set[i] + initial_set[j])
            
    # We will try to extend the set.
    # Strategy: Randomized Backtracking
    # We try to add elements from 12 to 284.
    
    def check_time():
        return time.time() < deadline

    def can_add(x, current_set, current_sums):
        # Check if adding x creates duplicate sums
        # New sums: x + y for y in current_set, and 2*x
        # Check if any of these are in current_sums
        # Also check if 2*x is in current_sums
        # Note: x + y where y < x is already checked when y was added? 
        # No, sums are symmetric. We store all sums a+b with a<=b.
        # If we add x, we generate sums x+y for y in current_set (y < x) and 2x.
        # We must ensure none of these are in current_sums.
        
        # Optimization: 
        # If x + min(current_set) > max(current_sums) is not a simple check.
        
        for y in current_set:
            s = x + y
            if s in current_sums:
                return False, None
        if 2 * x in current_sums:
            return False, None
        return True, None

    def backtrack(current_set, current_sums, start_val):
        nonlocal best_size, best_set
        
        if not check_time():
            return
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        # Heuristic: If we can't beat best_size even with max possible additions, stop
        # Max possible additions is roughly sqrt(284) - len(current_set)
        # This is weak.
        
        # Try to add the next element
        # We iterate through candidates from start_val to 284
        # To avoid deep recursion on failing branches, we can limit the depth or use randomization
        
        # Randomization: Shuffle candidates? 
        # No, we want to try small numbers first usually, but random restarts help.
        
        # Let's try a simple greedy-with-backtracking for the first few, 
        # then random restarts for the rest of the time.
        
        # For this specific function call, we do a depth-first search.
        # To prevent explosion, we only go deep if we are promising.
        
        # Actually, a better approach for short time:
        # 1. Greedy fill.
        # 2. Local search: try to swap elements to allow more additions.
        
        pass # Placeholder, logic below

    # Better Strategy:
    # 1. Run a greedy algorithm to get a baseline.
    # 2. Use the remaining time for randomized local search or backtracking.
    
    def greedy_fill(start_set, start_sums):
        current_set = list(start_set)
        current_sums = set(start_sums)
        for x in range(max(current_set) + 1, 285):
            if not check_time():
                break
            # Check if x can be added
            valid = True
            new_sums = []
            for y in current_set:
                s = x + y
                if s in current_sums:
                    valid = False
                    break
                new_sums.append(s)
            if valid:
                if 2 * x in current_sums:
                    valid = False
                else:
                    new_sums.append(2 * x)
            
            if valid:
                current_set.append(x)
                current_sums.update(new_sums)
        return current_set, current_sums

    # 1. Initial Greedy
    greedy_set, greedy_sums = greedy_fill(initial_set, initial_sums)
    if len(greedy_set) > best_size:
        best_size = len(greedy_set)
        best_set = list(greedy_set)
        
    # 2. Randomized Backtracking / Local Search
    # We will try to improve the greedy set by restarting with random perturbations
    # or by doing a backtracking search with random ordering.
    
    # Let's implement a randomized backtracking search.
    # It tries to build a set from scratch, but uses random choices for candidates.
    
    def randomized_search():
        nonlocal best_size, best_set
        current_set = list(initial_set)
        current_sums = set(initial_sums)
        
        # We will try to add elements one by one.
        # At each step, we pick a random valid candidate from a window of the smallest valid ones.
        # This helps escape local optima of the pure greedy.
        
        while len(current_set) < 30 and check_time(): # 30 is a loose upper bound
            candidates = []
            # Look at the next 50 numbers
            for x in range(max(current_set) + 1, min(max(current_set) + 51, 285)):
                valid = True
                for y in current_set:
                    if x + y in current_sums:
                        valid = False
                        break
                if valid and 2 * x not in current_sums:
                    candidates.append(x)
                    
            if not candidates:
                break
            
            # Pick one randomly
            x = random.choice(candidates)
            
            # Add it
            new_sums = [x + y for y in current_set] + [2 * x]
            current_set.append(x)
            current_sums.update(new_sums)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
                
            # Occasionally, backtrack? 
            # Simple random walk: if we get stuck, we restart the whole thing.
            
    # Run multiple randomized searches
    while check_time():
        randomized_search()
        
    return best_set
