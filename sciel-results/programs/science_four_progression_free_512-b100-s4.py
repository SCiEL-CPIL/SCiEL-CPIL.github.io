import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 58.0
    
    # Precompute all 4-term arithmetic progressions
    all_aps = []
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(all_aps)
    
    # Map each number to the indices of APs it belongs to
    ap_indices_by_num = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(all_aps):
        for x in ap:
            ap_indices_by_num[x].append(idx)
    
    # State for fast add/remove
    ap_count = [0] * num_aps
    in_set = [False] * (N + 1)
    current_size = 0
    
    def add_element(x):
        nonlocal current_size
        if in_set[x]:
            return True
        # Check if adding x creates a 4-AP
        for ap_idx in ap_indices_by_num[x]:
            if ap_count[ap_idx] == 3:
                return False
        # Add x
        in_set[x] = True
        current_size += 1
        for ap_idx in ap_indices_by_num[x]:
            ap_count[ap_idx] += 1
        return True

    def remove_element(x):
        nonlocal current_size
        if not in_set[x]:
            return True
        in_set[x] = False
        current_size -= 1
        for ap_idx in ap_indices_by_num[x]:
            ap_count[ap_idx] -= 1
        return True

    def reset_state():
        nonlocal ap_count, in_set, current_size
        ap_count = [0] * num_aps
        in_set = [False] * (N + 1)
        current_size = 0

    def greedy_fill(exclude=None):
        """
        Greedily add elements from a random order, skipping 'exclude' if provided.
        """
        if exclude is None:
            candidates = [i for i in range(1, N + 1) if not in_set[i]]
        else:
            candidates = [i for i in range(1, N + 1) if not in_set[i] and i != exclude]
        
        random.shuffle(candidates)
        for x in candidates:
            add_element(x)
        return current_size

    best_size = 0
    best_set = []

    # Phase 1: Multiple Randomized Greedy Starts
    for restart in range(50):
        if time.time() > deadline - 15:
            break
        reset_state()
        # Random order
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            add_element(x)
        
        if current_size > best_size:
            best_size = current_size
            best_set = [i for i in range(1, N + 1) if in_set[i]]
    
    # Phase 2: Local Search on the Best Set
    if best_size > 0:
        # Load best set into state
        reset_state()
        for x in best_set:
            add_element(x)
        
        # Local search loop
        for ls_iter in range(10000):
            if time.time() > deadline - 5:
                break
            
            # Try to add a random element
            # Pick a random element not in set
            candidates = [i for i in range(1, N + 1) if not in_set[i]]
            if not candidates:
                break
            
            # Try a few random additions
            added_any = False
            for _ in range(10):
                x = random.choice(candidates)
                if add_element(x):
                    added_any = True
                    # Update candidates? No, just continue. 
                    # But if we added x, it's in set now.
                    break
            
            if not added_any:
                # Stuck. Remove a random element and try to fill
                if current_size == 0:
                    break
                
                # Pick an element to remove. 
                # Heuristic: remove an element that is in many APs? 
                # Or just random.
                # Let's try random first.
                elem_to_remove = random.choice(best_set) # Wait, best_set is static. Use current in_set.
                # Generate list of in_set elements
                in_set_elems = [i for i in range(1, N + 1) if in_set[i]]
                if not in_set_elems:
                    break
                elem_to_remove = random.choice(in_set_elems)
                
                remove_element(elem_to_remove)
                
                # Now try to add elements
                # We can try to add the removed element back? No, that's a swap.
                # Let's try to add other elements.
                # Greedy fill with random order
                candidates = [i for i in range(1, N + 1) if not in_set[i]]
                random.shuffle(candidates)
                
                prev_size = current_size
                for x in candidates:
                    add_element(x)
                
                if current_size > best_size:
                    best_size = current_size
                    best_set = [i for i in range(1, N + 1) if in_set[i]]
                    # Reset the "stuck" state by continuing
                else:
                    # If we didn't improve, revert to previous best state?
                    # Or keep the current state if it's the same size?
                    # If we removed 1 and added 1, size is same.
                    # If we removed 1 and added 0, size decreased.
                    # We should revert if size decreased.
                    if current_size < prev_size:
                        # Revert: remove all added, add back removed
                        # This is complex. Easier: restart local search from best_set
                        reset_state()
                        for x in best_set:
                            add_element(x)
                        # Break out of inner loop to restart outer?
                        # Or just continue outer loop, which will pick a new element to remove
                        pass

    return sorted(best_set)
