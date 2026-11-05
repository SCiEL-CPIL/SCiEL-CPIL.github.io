import time
from math import lcm
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 45.0
    
    # LCM of 1 to 24
    L = 1
    for i in range(1, 25):
        L = lcm(L, i)
        
    # V[i] = L / i
    V = [0] * 25
    for i in range(1, 25):
        V[i] = L // i
        
    def get_subset_sums(values):
        """
        Returns a set of all possible subset sums of the input list of integers.
        Uses meet-in-the-middle for efficiency.
        """
        n = len(values)
        if n == 0:
            return {0}
        
        # Split into two halves
        mid = n // 2
        left = values[:mid]
        right = values[mid:]
        
        def gen_sums(arr):
            sums = {0}
            for val in arr:
                # Create new sums by adding val to existing sums
                new_sums = set()
                for s in sums:
                    new_sums.add(s + val)
                sums.update(new_sums)
            return sums
            
        left_sums = gen_sums(left)
        right_sums = gen_sums(right)
        
        # Combine
        all_sums = set()
        for ls in left_sums:
            for rs in right_sums:
                all_sums.add(ls + rs)
                
        return all_sums

    def is_subset_sum(target, values):
        """
        Checks if target can be formed as a sum of a subset of values.
        """
        if target < 0:
            return False
        if target == 0:
            return True
        
        # Quick check: if target is larger than sum of all values, impossible
        total_sum = sum(values)
        if target > total_sum:
            return False
            
        # Use meet-in-the-middle
        n = len(values)
        if n == 0:
            return target == 0
            
        mid = n // 2
        left = values[:mid]
        right = values[mid:]
        
        def gen_sums(arr):
            sums = {0}
            for val in arr:
                new_sums = set()
                for s in sums:
                    new_sums.add(s + val)
                sums.update(new_sums)
            return sums
            
        left_sums = gen_sums(left)
        right_sums = gen_sums(right)
        
        # Convert one to a set for O(1) lookup
        right_sums_set = set(right_sums)
        
        for ls in left_sums:
            if (target - ls) in right_sums_set:
                return True
        return False

    def can_add(x, S):
        """
        Checks if x can be added to set S without violating the reciprocal-subsum-free property.
        S is a list of selected integers.
        """
        if x in S:
            return False
            
        v_x = V[x]
        S_vals = [V[y] for y in S]
        
        # Check 1: Is 1/x a sum of reciprocals of S?
        # i.e., is v_x a subset sum of S_vals?
        if is_subset_sum(v_x, S_vals):
            return False
            
        # Check 2: For each y in S, is 1/y a sum of reciprocals of (S \ {y}) U {x}?
        # This is equivalent to: is (1/y - 1/x) a sum of reciprocals of (S \ {y})?
        # i.e., is (V[y] - V[x]) a subset sum of S_vals \ {V[y]}?
        
        for y in S:
            v_y = V[y]
            # We need to check if (v_y - v_x) is a subset sum of S_vals excluding v_y
            # Note: v_y - v_x could be negative. If negative, it's not a subset sum (sums are positive).
            if v_y - v_x < 0:
                continue
                
            # Create list of values excluding v_y
            # Since values are distinct (L/i are distinct for i in 1..24), we can just filter
            # But for efficiency, we can pass the list and handle exclusion in is_subset_sum?
            # My is_subset_sum takes a list. Let's just build the list.
            S_vals_minus_y = [v for v in S_vals if v != v_y]
            
            if is_subset_sum(v_y - v_x, S_vals_minus_y):
                return False
                
        return True

    # Backtracking search to find the largest valid set
    # We try to find a set of size K. Start with K=24, decrease if not found.
    # To make search efficient, we order candidates.
    # Adding larger numbers (smaller reciprocals) first is often good because
    # their reciprocals are small, making it harder to form sums for larger reciprocals?
    # Actually, the constraint is symmetric in a way, but the "sum of others" condition
    # depends on the set.
    
    # Let's try a recursive search.
    # State: current set S, list of remaining candidates C.
    # We want to maximize |S|.
    
    best_set = []
    
    def backtrack(S, candidates, target_size):
        nonlocal best_set
        nonlocal deadline
        
        if time.time() > deadline:
            return False
            
        if len(S) >= target_size:
            if len(S) > len(best_set):
                best_set = S[:]
            return True
            
        # Pruning: if we can't reach target_size with remaining candidates
        if len(S) + len(candidates) < target_size:
            return False
            
        # Try to add each candidate
        for i in range(len(candidates)):
            x = candidates[i]
            
            # Heuristic: Try to add x
            if can_add(x, S):
                new_S = S + [x]
                # Remaining candidates are those after i in the original list?
                # No, we can pick any from the rest.
                # To avoid permutations, we should only pick candidates that are "greater" in some order?
                # If we just iterate through a fixed list of candidates, we should only consider
                # candidates that come after the current one in the list to avoid duplicates.
                # Let's assume candidates is sorted.
                remaining = candidates[i+1:]
                
                if backtrack(new_S, remaining, target_size):
                    return True
                    
            # If we didn't return True, we try excluding x (implicitly by moving to next i)
            
        return False

    # We want to find the maximum size.
    # Let's try sizes from 24 down to 1.
    # To speed up, we can use a greedy initial solution as a lower bound.
    
    # Greedy approach to get a good initial set
    greedy_S = []
    # Try adding elements in descending order
    for x in range(24, 0, -1):
        if can_add(x, greedy_S):
            greedy_S.append(x)
            
    current_best_size = len(greedy_S)
    if current_best_size > 0:
        best_set = greedy_S[:]
        
    # Now try to improve using backtracking
    # We can try to find a set of size current_best_size + 1, etc.
    # Or just run backtracking to find the max possible.
    
    # Let's try to find a set of size 24, then 23, etc.
    # But backtracking for size 24 might be slow if it fails late.
    # Better: iterate target_size from 24 down to current_best_size + 1.
    
    candidates_all = list(range(24, 0, -1)) # Descending order
    
    for target_size in range(24, current_best_size, -1):
        if time.time() > deadline:
            break
            
        # Reset best_set? No, we want the global best.
        # But backtrack modifies best_set if it finds a larger one.
        # We need to ensure backtrack explores enough.
        
        # Simple backtracking might be too slow for large target sizes if the solution is sparse.
        # Let's try a different strategy:
        # Use the greedy set as a starting point and try to swap/add?
        # Or just run the backtrack with a time limit.
        
        # Let's try to find ANY set of size target_size.
        found = backtrack([], candidates_all, target_size)
        if found:
            # We found a set of size at least target_size
            # best_set is updated inside backtrack
            break
            
    # If no larger set found, return the best we have
    if not best_set:
        return [24]
        
    return best_set

# Test locally? No, just provide the code.
