import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    max_val = 4096
    max_sum = 3 * max_val + 1
    
    def try_add(x, selected, S, used_sums):
        """Try to add x to the current set. Returns (success, new_S, new_used_sums) if successful."""
        if len(selected) == 0:
            S_new = np.array([x], dtype=np.int32)
            used_sums_new = used_sums.copy()
            used_sums_new[3*x] = True
            return True, S_new, used_sums_new
        
        k = len(selected)
        
        # Generate new triple sums involving x
        s1 = np.array([3*x], dtype=np.int32)
        s2 = 2*x + S
        i_idx, j_idx = np.triu_indices(k)
        s3 = x + S[i_idx] + S[j_idx]
        
        all_new_sums = np.concatenate((s1, s2, s3))
        
        # Check for duplicates within new sums
        if len(np.unique(all_new_sums)) != len(all_new_sums):
            return False, None, None
            
        # Check against used_sums
        if np.any(used_sums[all_new_sums]):
            return False, None, None
            
        # If valid, update
        S_new = np.append(S, x)
        used_sums_new = used_sums.copy()
        used_sums_new[all_new_sums] = True
        return True, S_new, used_sums_new

    def greedy_build(order=None):
        """Build a set using greedy strategy with custom order."""
        selected = []
        S = np.array([], dtype=np.int32)
        used_sums = np.zeros(max_sum, dtype=bool)
        
        if order is None:
            order = range(1, max_val + 1)
        
        for x in order:
            if time.time() > deadline:
                break
            success, S_new, used_sums_new = try_add(x, selected, S, used_sums)
            if success:
                selected.append(x)
                S = S_new
                used_sums = used_sums_new
        
        return selected, S, used_sums

    # Primary strategy: Greedy from 1
    best_set, best_S, best_used = greedy_build(order=range(1, max_val + 1))
    best_len = len(best_set)
    
    # Try different orderings to find better sets
    orderings = [
        range(1, max_val + 1),  # ascending
        range(max_val, 0, -1),  # descending
        list(range(1, max_val + 1, 2)) + list(range(2, max_val + 1, 2)),  # odds then evens
        list(range(2, max_val + 1, 2)) + list(range(1, max_val + 1, 2)),  # evens then odds
    ]
    
    for order in orderings[1:]:
        if time.time() > deadline - 10:
            break
        alt_set, alt_S, alt_used = greedy_build(order=order)
        if len(alt_set) > best_len:
            best_set = alt_set
            best_S = alt_S
            best_used = alt_used
            best_len = len(best_set)
    
    # Local improvement: try to add skipped elements
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        
        current_set = list(best_set)
        current_S = best_S.copy()
        current_used = best_used.copy()
        current_set_set = set(current_set)
        
        improved = False
        # Try to insert elements that were skipped, in order
        for x in range(1, max_val + 1):
            if time.time() > deadline - 5:
                break
            if x in current_set_set:
                continue
            success, S_new, used_new = try_add(x, current_set, current_S, current_used)
            if success:
                current_set.append(x)
                current_set_set.add(x)
                current_S = S_new
                current_used = used_new
                improved = True
                if len(current_set) > best_len:
                    best_set = list(current_set)
                    best_S = current_S
                    best_used = current_used
                    best_len = len(best_set)
        
        if not improved:
            break
    
    # Try swap-based local search: replace an element with a larger one
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            
            current_set = list(best_set)
            current_S = best_S.copy()
            current_used = best_used.copy()
            current_set_set = set(current_set)
            improved = False
            
            # For each element in the set, try replacing with a larger number
            for i, elem in enumerate(current_set):
                if time.time() > deadline - 5:
                    break
                
                # Try replacing elem with a larger number
                for new_elem in range(elem + 1, max_val + 1):
                    if time.time() > deadline - 5:
                        break
                    if new_elem in current_set_set:
                        continue
                    
                    # Remove elem
                    new_set = current_set[:i] + current_set[i+1:]
                    new_set_set = set(new_set)
                    
                    # Try to add new_elem
                    success, S_new, used_new = try_add(new_elem, new_set, best_S[:len(new_set)], current_used)
                    
                    if success:
                        # Check if we can then add more elements
                        test_set = new_set + [new_elem]
                        test_set_set = set(test_set)
                        test_S = S_new
                        test_used = used_new
                        
                        # Try to fill in more elements
                        for fill in range(1, max_val + 1):
                            if time.time() > deadline - 5:
                                break
                            if fill in test_set_set:
                                continue
                            success2, S_fill, used_fill = try_add(fill, test_set, test_S, test_used)
                            if success2:
                                test_set.append(fill)
                                test_set_set.add(fill)
                                test_S = S_fill
                                test_used = used_fill
                        
                        if len(test_set) > best_len:
                            best_set = test_set
                            best_S = test_S
                            best_used = test_used
                            best_len = len(best_set)
                            current_set = list(test_set)
                            current_S = test_S
                            current_used = test_used
                            current_set_set = test_set_set
                            improved = True
                        break  # Only try first valid replacement for this elem
            
            if not improved:
                break
    
    return best_set
