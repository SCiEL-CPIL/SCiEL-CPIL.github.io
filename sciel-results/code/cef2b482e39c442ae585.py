"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Backtracking search: build Sidon set element by element
    # State: (tuple of current elements, set of used differences, set of required diffs still needed)
    # We want to minimize max element
    
    best_solution = None
    best_max_val = 10**6
    
    def get_diffs_set(elements):
        """Get set of all positive differences."""
        diffs = set()
        for i in range(len(elements)):
            for j in range(i+1, len(elements)):
                diffs.add(elements[j] - elements[i])
        return diffs
    
    def is_valid_sidon(elements):
        """Check if elements form a Sidon set (all positive differences distinct)."""
        n = len(elements)
        if n < 2:
            return True
        diffs = []
        for i in range(n):
            for j in range(i+1, n):
                d = elements[j] - elements[i]
                if d in diffs:
                    return False
                diffs.append(d)
        return True
    
    def covers_all_diffs(elements):
        """Check if all differences 1..30 are present."""
        diffs = get_diffs_set(elements)
        return all(d in diffs for d in range(1, 31))
    
    def backtrack(elements, used_diffs, next_idx, max_val_bound):
        """
        Backtracking search.
        elements: tuple of elements so far (sorted)
        used_diffs: frozenset of differences used so far
        next_idx: index of next element to choose
        max_val_bound: current best max value to beat
        Returns: (success, solution_tuple) or (False, None)
        """
        nonlocal best_solution, best_max_val
        
        if time.time() > deadline:
            return (False, None)
        
        # Check if we've covered all required differences
        if all(d in used_diffs for d in range(1, 31)):
            # We have a valid solution
            current_max = elements[-1]
            if current_max < best_max_val:
                best_max_val = current_max
                best_solution = elements
            return (True, elements)
        
        # If we've used too many elements, prune
        if len(elements) > 32:
            return (False, None)
        
        # Lower bound: we need at least (30 - len(used_diffs & set(range(1,31)))) more differences
        # Each new element can add at most (len(elements)) new differences
        remaining_needed = 30 - len(used_diffs & set(range(1, 31)))
        if remaining_needed > 0:
            max_possible_new_diffs = (32 - len(elements)) * len(elements)
            if remaining_needed > max_possible_new_diffs:
                return (False, None)
        
        # Try adding the next element
        last = elements[-1]
        
        # Try candidate values from last+1 upward
        # But we can prune: if last+1 > max_val_bound, no point
        if last + 1 > max_val_bound:
            return (False, None)
        
        # Try candidates in increasing order
        for candidate in range(last + 1, max_val_bound + 1):
            if time.time() > deadline:
                break
            
            # Check if adding candidate maintains Sidon property
            new_diffs = []
            valid = True
            for i in range(len(elements)):
                d = candidate - elements[i]
                if d <= 0:
                    continue
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Check if this candidate helps cover required differences
            new_used_diffs = frozenset(used_diffs | set(new_diffs))
            
            # Prune: if even with this element we can't cover all remaining diffs
            remaining_after = 30 - len(new_used_diffs & set(range(1, 31)))
            if remaining_after > 0:
                max_future_diffs = (32 - len(elements) - 1) * (len(elements) + 1)
                if remaining_after > max_future_diffs:
                    continue
            
            # Recurse
            success, sol = backtrack(elements + (candidate,), new_used_diffs, next_idx + 1, max_val_bound)
            if success and sol is not None:
                return (True, sol)
        
        return (False, None)
    
    # Start with just [0]
    initial_elements = (0,)
    initial_diffs = frozenset()
    
    # Try increasing bounds
    # A Sidon set with n elements has max value at least ~n^2 for optimal
    # For 30 differences, we likely need around 8-10 elements
    # Try bounds from small to large
    
    for bound in range(10, 500, 5):
        if time.time() > deadline:
            break
        success, sol = backtrack(initial_elements, initial_diffs, 0, bound)
        if success and sol is not None:
            if sol[-1] < best_max_val:
                best_max_val = sol[-1]
                best_solution = sol
            break
    
    # If backtracking didn't find a good solution, try a greedy approach as fallback
    if best_solution is None or best_max_val > 200:
        # Greedy construction
        a = [0]
        for d in range(1, 31):
            if d in get_diffs_set(a):
                continue
            found = False
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if is_valid_sidon(trial):
                    a = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if is_valid_sidon(trial):
                        a = trial
                        break
                    y += 1
        
        if covers_all_diffs(a) and len(a) <= 32 and max(a) < best_max_val:
            best_solution = tuple(a)
            best_max_val = max(a)
    
    # Final verification
    if best_solution is not None:
        sol = list(best_solution)
        diffs = get_diffs_set(sol)
        if (len(diffs) == len(sol) * (len(sol) - 1) // 2 and 
            all(d in diffs for d in range(1, 31)) and
            0 in sol and
            len(sol) <= 32 and
            max(sol) <= 10**6):
            return sol
    
    # Emergency fallback: return a known valid small set if we have one
    # Try a simple construction
    a = [0]
    for d in range(1, 31):
        if d in get_diffs_set(a):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if is_valid_sidon(trial):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if is_valid_sidon(trial):
                    a = trial
                    break
                y += 1
    
    return a

# EVOLVE-BLOCK-END
