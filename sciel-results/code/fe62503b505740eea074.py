import random
import time
import sys
from itertools import combinations

L = 5354228880  # lcm(1..24)

def subset_sum_exists(weights, target):
    """
    Check if target can be formed by a subset of weights.
    Uses meet-in-the-middle.
    Returns True if found, False otherwise.
    """
    n = len(weights)
    if n == 0:
        return target == 0
    if target == 0:
        return True
    
    # Filter weights > target
    weights = [w for w in weights if w <= target]
    if not weights:
        return target == 0
        
    n = len(weights)
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]
    
    left_sums = set()
    left_sums.add(0)
    for w in left:
        new_sums = set()
        for s in left_sums:
            if s + w <= target:
                new_sums.add(s + w)
        left_sums.update(new_sums)
        
    right_sums = set()
    right_sums.add(0)
    for w in right:
        new_sums = set()
        for s in right_sums:
            if s + w <= target:
                new_sums.add(s + w)
        right_sums.update(new_sums)
        
    # Check intersection
    for s in left_sums:
        if (target - s) in right_sums:
            return True
    return False

def get_violation(S):
    """
    Returns (x, T) if x's reciprocal is sum of reciprocals of T, else None.
    S is a set of ints.
    """
    S_list = list(S)
    for x in S_list:
        target = L // x
        weights = []
        T = []
        for y in S_list:
            if y != x:
                w = L // y
                weights.append(w)
                T.append(y)
        
        # Check if target is subset sum of weights
        if subset_sum_exists(weights, target):
            # Find the actual subset T
            def find_subset(weights, indices, target):
                if target == 0:
                    return []
                if not weights:
                    return None
                # Try including weights[0]
                if weights[0] <= target:
                    res = find_subset(weights[1:], indices[1:], target - weights[0])
                    if res is not None:
                        return [indices[0]] + res
                # Try excluding weights[0]
                res = find_subset(weights[1:], indices[1:], target)
                if res is not None:
                    return res
                return None

            T_indices = find_subset(weights, T, target)
            if T_indices is not None:
                return x, T_indices
            else:
                # Should not happen
                return x, T
                
    return None

def is_valid(S):
    return get_violation(S) is None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin before 90s kill
    
    all_elements = list(range(1, 25))
    
    # Precompute weights for each element
    weights = {x: L // x for x in all_elements}
    
    best_set = None
    best_size = 0
    
    # Backtracking search with forward checking
    # We try to build a valid set by adding elements one at a time
    # and prune branches when a violation is detected
    
    def backtrack(current_set, remaining, depth):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        # Update best if current set is larger
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
        
        # If no more elements to consider, return
        if not remaining:
            return
        
        # Forward checking: for each element in current_set,
        # check if adding any remaining element would create a violation
        # We'll do this lazily - just try adding elements and backtrack if invalid
        
        # Try adding elements from remaining in a smart order
        # Prioritize elements that are "safe" - unlikely to create violations
        
        # Sort remaining by some heuristic: elements with larger values first
        # (larger elements have smaller reciprocals, less likely to be sums)
        sorted_remaining = sorted(remaining, key=lambda x: -x)
        
        for elem in sorted_remaining:
            if time.time() > deadline:
                return
            
            # Try adding elem
            new_set = current_set | {elem}
            
            # Check if new_set is valid
            if is_valid(new_set):
                # Recurse
                new_remaining = [x for x in remaining if x != elem]
                backtrack(new_set, new_remaining, depth + 1)
            
            # Also try not adding elem (skip it)
            # But we want to maximize size, so we should try adding first
            # However, sometimes skipping is necessary
            # Let's try both: add and skip
            
        # Try skipping elements (to find larger sets that don't include all)
        # This is the "subset" part of subset selection
        # We need to explore combinations, not just greedy
    
    # Better approach: systematic search over subsets using backtracking
    # with pruning based on validity
    
    def search_subsets(available, current_set, index):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        # Update best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
        
        if index >= len(available):
            return
        
        # Option 1: Skip available[index]
        search_subsets(available, current_set, index + 1)
        
        # Option 2: Include available[index] if valid
        elem = available[index]
        new_set = current_set | {elem}
        
        # Quick check: is new_set valid?
        # We can do a fast check by only checking violations involving elem
        # (violations not involving elem were already checked in previous steps)
        
        # Actually, we need to check all violations in new_set
        # But we can optimize: only check if elem is involved in a violation
        # OR if adding elem creates a new violation among existing elements
        # The latter can't happen - adding an element can only create violations
        # where the new element is the target or part of the sum
        
        # So we only need to check:
        # 1. Is elem's reciprocal a sum of reciprocals of a subset of current_set?
        # 2. For each x in current_set, is x's reciprocal now a sum that includes elem?
        
        # Check 1:
        target = weights[elem]
        if current_set:
            other_weights = [weights[x] for x in current_set]
            if subset_sum_exists(other_weights, target):
                # elem would be violated, don't add
                return  # Only skip, don't include
        
        # Check 2: For each x in current_set, check if x's reciprocal can be formed
        # using a subset of (current_set - {x}) U {elem}
        # This is more expensive. Let's just do full is_valid for simplicity
        # since N is small
        
        if is_valid(new_set):
            search_subsets(available, new_set, index + 1)
    
    # Start with empty set and all elements available
    # But this is 2^24 = 16M possibilities - too many without good pruning
    
    # Better: use the fact that we want LARGE sets
    # Start with full set and remove elements (repair approach) but smarter
    
    # Hybrid: try multiple strategies
    
    # Strategy 1: Greedy construction with multiple orders
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Random order
        order = all_elements[:]
        random.shuffle(order)
        
        current_set = set()
        for elem in order:
            if time.time() > deadline:
                break
            new_set = current_set | {elem}
            if is_valid(new_set):
                current_set = new_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    # Strategy 2: Start with full set, repair by removing elements
    for trial in range(50):
        if time.time() > deadline:
            break
        
        S = set(all_elements)
        
        # Repair loop
        max_iterations = 100
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            violation = get_violation(S)
            if violation is None:
                break
            
            x, T = violation
            candidates = {x} | set(T)
            
            # Try removing each candidate, pick the one that leads to valid set
            # or minimizes violations
            best_candidate = None
            best_score = float('inf')
            
            for c in candidates:
                S_test = S.copy()
                S_test.remove(c)
                
                # Score: 0 if valid, else count of violations
                if is_valid(S_test):
                    best_candidate = c
                    best_score = 0
                    break
                
                # Count violations (expensive, but N is small)
                # Just use a simple heuristic: prefer removing x
                # or the element with highest weight
                score = len(S_test)  # Prefer larger sets
                if best_candidate is None or score < best_score:
                    best_candidate = c
                    best_score = score
            
            if best_candidate is None:
                best_candidate = random.choice(list(candidates))
            
            S.remove(best_candidate)
            
            if not S:
                break
        
        # Expand: try to add back elements
        missing = list(set(all_elements) - S)
        random.shuffle(missing)
        for m in missing:
            if time.time() > deadline:
                break
            S_test = S.copy()
            S_test.add(m)
            if is_valid(S_test):
                S = S_test
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Strategy 3: Local search / hill climbing
    if best_set:
        for trial in range(20):
            if time.time() > deadline:
                break
            
            S = set(best_set)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                
                # Try removing one element and adding another
                for remove_elem in list(S):
                    if time.time() > deadline:
                        break
                    for add_elem in (set(all_elements) - S):
                        S_test = S - {remove_elem} | {add_elem}
                        if is_valid(S_test):
                            S = S_test
                            improved = True
                            break
                    if improved:
                        break
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
    
    if best_set is None:
        return [1]
        
    return list(best_set)

