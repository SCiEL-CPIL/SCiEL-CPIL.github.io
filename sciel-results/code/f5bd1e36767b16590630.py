"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations_with_replacement

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    n = 4096
    max_size = 80
    
    # We'll use a backtracking search with heuristic ordering.
    # The key insight: for B_3 sequences (all unordered triples with repetition
    # have unique sums), we want to maximize the number of elements in [1, n].
    
    # Greedy with backtracking:
    # - Maintain a set of all triple sums seen so far
    # - When adding a new element x, check that no triple sum involving x collides
    # - Use MRV (Minimum Remaining Values) heuristic: try candidates that leave
    #   the most options for future elements
    
    def check_and_add(selected, sums_set, x):
        """Check if x can be added to selected without creating duplicate triple sums."""
        # Triples involving x:
        # (x, x, x), (x, x, y) for y in selected, (x, y, z) for y <= z in selected
        new_sums = set()
        
        # (x, x, x)
        s = 3 * x
        if s in sums_set:
            return False
        new_sums.add(s)
        
        # (x, x, y) for each y in selected
        for y in selected:
            s = 2 * x + y
            if s in sums_set or s in new_sums:
                return False
            new_sums.add(s)
        
        # (x, y, z) for y <= z in selected
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                y = selected[i]
                z = selected[j]
                s = x + y + z
                if s in sums_set or s in new_sums:
                    return False
                new_sums.add(s)
        
        return True
    
    def get_new_sums(selected, sums_set, x):
        """Get the set of new triple sums if x is added."""
        new_sums = set()
        new_sums.add(3 * x)
        for y in selected:
            new_sums.add(2 * x + y)
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                new_sums.add(x + selected[i] + selected[j])
        return new_sums
    
    def backtrack(selected, sums_set, start_val, target_size, time_limit):
        """
        Backtracking search to find a valid set of size target_size.
        selected: list of chosen elements (sorted)
        sums_set: set of all triple sums
        start_val: minimum value to try next
        target_size: desired total size
        """
        if time.time() > time_limit:
            return selected
        
        if len(selected) == target_size:
            return selected
        
        # Try candidate values starting from start_val
        # Heuristic: try smaller values first (they're more constrained)
        # but also consider that larger gaps might be better
        
        best_result = selected[:]
        
        for x in range(start_val, n + 1):
            if time.time() > time_limit:
                break
            
            if not check_and_add(selected, sums_set, x):
                continue
            
            new_sums = get_new_sums(selected, sums_set, x)
            new_selected = selected + [x]
            new_sums_set = sums_set | new_sums
            
            result = backtrack(new_selected, new_sums_set, x + 1, target_size, time_limit)
            if len(result) > len(best_result):
                best_result = result
            
            # If we found a full solution, return it
            if len(best_result) == target_size:
                return best_result
            
            # Pruning: if even trying all remaining values can't beat best_result, stop
            # (simple bound: remaining candidates = n - x)
            if len(selected) + (n - x) <= len(best_result):
                break
        
        return best_result
    
    # Strategy: try to build up to target_size using backtracking
    # Start with a greedy solution, then try to extend it
    
    # First, get a quick greedy solution
    greedy_selected = []
    greedy_sums = set()
    for x in range(1, n + 1):
        if check_and_add(greedy_selected, greedy_sums, x):
            greedy_selected.append(x)
            greedy_sums = greedy_sums | get_new_sums(greedy_selected[:-1], greedy_sums, x)
        if time.time() > deadline:
            break
    
    best_solution = greedy_selected[:]
    best_size = len(greedy_selected)
    
    # Try to extend the greedy solution with backtracking
    if best_size < max_size:
        # Try backtracking from the greedy solution
        result = backtrack(greedy_selected, greedy_sums, greedy_selected[-1] + 1, max_size, deadline)
        if len(result) > best_size:
            best_solution = result
            best_size = len(result)
    
    # Try a different starting point: skip some early numbers
    # This can help when early small numbers block many future options
    if best_size < max_size and time.time() < deadline:
        # Try starting from different offsets
        for offset in [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]:
            if time.time() > deadline - 5:
                break
            
            test_selected = []
            test_sums = set()
            
            # Skip the first 'offset' values
            start = offset + 1
            for x in range(start, n + 1):
                if time.time() > deadline - 5:
                    break
                if check_and_add(test_selected, test_sums, x):
                    test_selected.append(x)
                    test_sums = test_sums | get_new_sums(test_selected[:-1], test_sums, x)
            
            if len(test_selected) > best_size:
                best_solution = test_selected
                best_size = len(test_selected)
            
            # Try to extend this with backtracking
            if best_size < max_size and time.time() < deadline - 10:
                result = backtrack(test_selected, test_sums, test_selected[-1] + 1, max_size, deadline - 5)
                if len(result) > best_size:
                    best_solution = result
                    best_size = len(result)
            
            if best_size == max_size:
                break
    
    # Verify the solution is valid
    if best_solution:
        all_sums = [sum(t) for t in combinations_with_replacement(best_solution, 3)]
        if len(all_sums) != len(set(all_sums)):
            # Fallback to greedy
            best_solution = greedy_selected
    
    return best_solution
# EVOLVE-BLOCK-END
