import time
import random

def construct():
    n = 256
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    # Start with the proven safe tail
    # k=171 is the smallest k such that 3k > 2n (3*171=513 > 512)
    k = 171
    base_tail = set(range(k, n + 1))
    
    def can_add(x, S):
        """Check if x can be added to set S while maintaining the property."""
        S_sorted = sorted(S)
        
        # Case 1: x is the smallest element (a=x)
        # Need x < b <= c in S, and x does NOT divide (b+c)
        for i, b in enumerate(S_sorted):
            if x >= b:
                continue
            for c in S_sorted[i:]:
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is the middle element (b=x)
        # Need a < x <= c in S, and a does NOT divide (x+c)
        for a in S_sorted:
            if a >= x:
                break
            for c in S_sorted:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is the largest element (c=x)
        # Need a < b <= x in S∪{x}, and a does NOT divide (b+x)
        # Subcase 3a: b = x, so check a < x and a does NOT divide 2x
        for a in S_sorted:
            if a >= x:
                break
            if (2 * x) % a == 0:
                return False
        
        # Subcase 3b: b < x, so a < b < x, check a does NOT divide (b+x)
        for i, b in enumerate(S_sorted):
            if b >= x:
                break
            for a in S_sorted[:i]:
                if (b + x) % a == 0:
                    return False
        
        return True
    
    def is_valid_set(S):
        """Verify that set S satisfies the property for all triples."""
        S_sorted = sorted(S)
        for i, a in enumerate(S_sorted):
            for j in range(i + 1, len(S_sorted)):
                b = S_sorted[j]
                for k_idx in range(j, len(S_sorted)):
                    c = S_sorted[k_idx]
                    if (b + c) % a == 0:
                        return False
        return True
    
    # Try multiple greedy strategies
    best_set = None
    best_size = 0
    
    strategies = [
        'descending',  # Try largest candidates first
        'ascending',   # Try smallest candidates first
        'random1',     # Random order with seed 1
        'random2',     # Random order with seed 2
    ]
    
    for strategy in strategies:
        if time.time() > deadline:
            break
        
        current_set = set(base_tail)
        
        if strategy == 'descending':
            candidates = list(range(k - 1, 0, -1))
        elif strategy == 'ascending':
            candidates = list(range(1, k))
        elif strategy == 'random1':
            candidates = list(range(1, k))
            random.seed(1)
            random.shuffle(candidates)
        elif strategy == 'random2':
            candidates = list(range(1, k))
            random.seed(42)
            random.shuffle(candidates)
        
        # Greedy addition
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, current_set):
                current_set.add(x)
        
        # Local search: try to swap elements to improve
        if time.time() < deadline - 5.0:
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                current_list = sorted(current_set)
                
                # Try removing an element and adding a different one
                for i, remove_x in enumerate(current_list):
                    if time.time() > deadline - 2.0:
                        break
                    trial_set = set(current_set)
                    trial_set.discard(remove_x)
                    
                    # Try adding elements not in current_set
                    for add_x in range(1, n + 1):
                        if add_x in trial_set:
                            continue
                        if can_add(add_x, trial_set):
                            # Check if this improves or maintains size
                            new_set = trial_set | {add_x}
                            if len(new_set) > len(current_set):
                                current_set = new_set
                                improved = True
                                break
                            elif len(new_set) == len(current_set):
                                # Same size, but maybe better structure for future additions
                                # Keep it if it allows more additions later (heuristic: prefer larger elements)
                                if add_x > remove_x:
                                    current_set = new_set
                                    improved = True
                                    break
    
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    # If we have time, try a more aggressive local search on the best set
    if best_set is not None and time.time() < deadline - 10.0:
        current_set = set(best_set)
        improved = True
        iterations = 0
        max_iterations = 1000
        
        while improved and iterations < max_iterations and time.time() < deadline - 5.0:
            improved = False
            iterations += 1
            current_list = sorted(current_set)
            
            # Try all possible swaps
            for i, remove_x in enumerate(current_list):
                if time.time() > deadline - 5.0:
                    break
                trial_set = set(current_set)
                trial_set.discard(remove_x)
                
                for add_x in range(1, n + 1):
                    if add_x in trial_set:
                        continue
                    if can_add(add_x, trial_set):
                        new_set = trial_set | {add_x}
                        if len(new_set) > len(current_set):
                            current_set = new_set
                            improved = True
                            break
                        elif len(new_set) == len(current_set):
                            # Prefer configurations that might allow more additions
                            # Heuristic: check if this new set allows adding more elements
                            count_addable = 0
                            for test_x in range(1, n + 1):
                                if test_x not in new_set:
                                    if can_add(test_x, new_set):
                                        count_addable += 1
                                        if count_addable >= 3:
                                            break
                            
                            # Compare with current set's addable count
                            count_addable_current = 0
                            for test_x in range(1, n + 1):
                                if test_x not in current_set:
                                    if can_add(test_x, current_set):
                                        count_addable_current += 1
                                        if count_addable_current >= 3:
                                            break
                            
                            if count_addable > count_addable_current:
                                current_set = new_set
                                improved = True
                                break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    # Final validation
    if best_set is None or not is_valid_set(best_set):
        # Fallback to the base tail
        return sorted(list(base_tail))
    
    return sorted(list(best_set))
