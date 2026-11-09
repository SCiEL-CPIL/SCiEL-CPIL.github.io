"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # safety margin before 90s kill
    
    N = 512
    
    import numpy as np
    
    # Precompute all 4-term APs as triples for each element
    # ap_triples[x] = list of (y, z, w) such that {x,y,z,w} is a 4-AP
    ap_triples = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap_triples[a].append((b, c, e))
            ap_triples[b].append((a, c, e))
            ap_triples[c].append((a, b, e))
            ap_triples[e].append((a, b, c))
    
    # Convert to numpy arrays for faster operations
    # For each x, store the triples as a 2D array
    ap_arrays = [None] * (N + 1)
    for i in range(1, N + 1):
        if ap_triples[i]:
            ap_arrays[i] = np.array(ap_triples[i], dtype=np.int32)
        else:
            ap_arrays[i] = np.empty((0, 3), dtype=np.int32)
    
    def can_add(x, selected_np):
        """Check if adding x to selected_np creates a 4-AP. Returns True if safe to add."""
        arr = ap_arrays[x]
        if len(arr) == 0:
            return True
        # For each triple (y,z,w), check if all three are in selected
        # selected_np is a 1-indexed boolean numpy array
        # We need to check if selected_np[y] & selected_np[z] & selected_np[w] for any row
        ys = arr[:, 0]
        zs = arr[:, 1]
        ws = arr[:, 2]
        # Check if any row has all three selected
        all_selected = (selected_np[ys] & selected_np[zs] & selected_np[ws])
        return not np.any(all_selected)
    
    def greedy_construct(order, selected_np=None):
        if selected_np is None:
            selected_np = np.zeros(N + 1, dtype=bool)
        count = 0
        for x in order:
            if can_add(x, selected_np):
                selected_np[x] = True
                count += 1
        return selected_np, count
    
    def local_search(selected_np, count, time_limit):
        """Try to improve by removing one element and adding two others."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try removing each element in the set
            in_set = [i for i in range(1, N + 1) if selected_np[i]]
            not_in_set = [i for i in range(1, N + 1) if not selected_np[i]]
            
            for rem in in_set:
                if time.time() > time_limit:
                    break
                # Remove rem
                selected_np[rem] = False
                temp_count = count - 1
                
                # Try to add elements from not_in_set
                added = []
                for add in not_in_set:
                    if time.time() > time_limit:
                        break
                    if can_add(add, selected_np):
                        selected_np[add] = True
                        temp_count += 1
                        added.append(add)
                
                # If we added more than 1 element, we improved
                if len(added) >= 2:
                    count = temp_count
                    improved = True
                    # Update not_in_set
                    not_in_set = [i for i in range(1, N + 1) if not selected_np[i]]
                    in_set = [i for i in range(1, N + 1) if selected_np[i]]
                    break
                else:
                    # Revert: put rem back, remove any added
                    for a in added:
                        selected_np[a] = False
                    selected_np[rem] = True
        
        return selected_np, count
    
    best_set = None
    best_count = 0
    
    # Natural order
    order = list(range(1, N + 1))
    selected, count = greedy_construct(order)
    if count > best_count:
        best_count = count
        best_set = selected.copy()
    
    # Try random orders
    random.seed(42)
    for _ in range(300):
        if time.time() > deadline - 15:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        selected, count = greedy_construct(order)
        if count > best_count:
            best_count = count
            best_set = selected.copy()
    
    # Local search on best
    if best_set is not None:
        ls_deadline = min(deadline - 2, time.time() + 30)
        best_set, best_count = local_search(best_set.copy(), best_count, ls_deadline)
    
    # More random restarts with local search
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        selected, count = greedy_construct(order)
        
        # Quick local search
        if count >= best_count - 2:
            ls_deadline = min(deadline - 3, time.time() + 5)
            selected, count = local_search(selected, count, ls_deadline)
        
        if count > best_count:
            best_count = count
            best_set = selected.copy()
    
    # Convert best_set to a sorted list
    result = [i for i in range(1, N + 1) if best_set[i]]
    
    return result

# EVOLVE-BLOCK-END
