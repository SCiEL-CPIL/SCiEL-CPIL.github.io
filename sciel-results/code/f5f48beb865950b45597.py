"""B_3[1] construction: NumPy-accelerated greedy with batched collision checking."""
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85.0
    N = 4096
    MAX_SUM = 3 * N  # Max possible triple sum = 3*4096 = 12288
    
    # Use a numpy boolean array for sums tracking
    sums_arr = np.zeros(MAX_SUM + 1, dtype=np.bool_)
    
    def can_add_vectorized(x, selected_np):
        """Check if x can be added using vectorized numpy operations."""
        # 3x
        if sums_arr[3 * x]:
            return False
        # 2x + s for s in selected
        if np.any(sums_arr[2 * x + selected_np]):
            return False
        # x + 2s for s in selected
        if np.any(sums_arr[x + 2 * selected_np]):
            return False
        # x + s + t for s <= t in selected
        n = len(selected_np)
        if n > 0:
            # Create all pairs s <= t
            i_idx, j_idx = np.triu_indices(n)
            pair_sums = selected_np[i_idx] + selected_np[j_idx]
            if np.any(sums_arr[x + pair_sums]):
                return False
        return True

    def add_x(x, selected_list, selected_np):
        """Add x to selected and update sums."""
        selected_list.append(x)
        sums_arr[3 * x] = True
        if len(selected_np) > 0:
            sums_arr[2 * x + selected_np] = True
            sums_arr[x + 2 * selected_np] = True
            n = len(selected_np)
            i_idx, j_idx = np.triu_indices(n)
            sums_arr[x + selected_np[i_idx] + selected_np[j_idx]] = True
        return np.append(selected_np, x)

    def remove_x(x, selected_list, selected_np):
        """Remove x from selected and update sums."""
        idx = selected_list.index(x)
        selected_list.pop(idx)
        sums_arr[3 * x] = False
        remaining_np = np.delete(selected_np, idx)
        if len(remaining_np) > 0:
            sums_arr[2 * x + remaining_np] = False
            sums_arr[x + 2 * remaining_np] = False
            n = len(remaining_np)
            i_idx, j_idx = np.triu_indices(n)
            sums_arr[x + remaining_np[i_idx] + remaining_np[j_idx]] = False
        return remaining_np

    # Phase 1: Greedy fill from 1 to N
    selected_list = []
    selected_np = np.array([], dtype=np.int32)
    sel_set = set()
    
    for x in range(1, N + 1):
        if time.time() > deadline - 20.0:
            break
        if x not in sel_set and can_add_vectorized(x, selected_np):
            selected_np = add_x(x, selected_list, selected_np)
            sel_set.add(x)
    
    selected_list.sort()
    best_selected = selected_list[:]
    best_size = len(selected_list)
    
    # Phase 2: Swap improvement
    if time.time() < deadline - 8.0:
        sel_set = set(selected_list)
        candidates = sorted([x for x in range(1, N + 1) if x not in sel_set], reverse=True)
        
        cand_attempts = {c: 0 for c in candidates}
        max_attempts_per_cand = 3
        
        improved = True
        while improved and time.time() < deadline - 6.0:
            improved = False
            for i in range(len(selected_list)):
                if time.time() > deadline - 6.0:
                    break
                orig = selected_list[i]
                for idx, cand in enumerate(candidates):
                    if time.time() > deadline - 6.0:
                        break
                    if cand_attempts[cand] >= max_attempts_per_cand:
                        continue
                    if cand <= orig:
                        continue
                    # Remove orig, try adding cand
                    remaining_np = remove_x(orig, selected_list, selected_np)
                    if can_add_vectorized(cand, remaining_np):
                        selected_np = add_x(cand, selected_list, remaining_np)
                        candidates.pop(idx)
                        improved = True
                        break
                    else:
                        # Restore
                        selected_np = add_x(orig, selected_list, remaining_np)
                        cand_attempts[cand] += 1
                selected_list.sort()
        
        if len(selected_list) > best_size:
            best_selected = selected_list[:]
            best_size = len(selected_list)
    
    # Phase 3: Try adding more elements with randomized orderings
    if time.time() < deadline - 3.0:
        for attempt in range(15):
            if time.time() > deadline - 3.0:
                break
            
            sel_set = set(best_selected)
            candidates = [x for x in range(N, 0, -1) if x not in sel_set]
            if attempt > 0:
                random.shuffle(candidates)
            
            # Rebuild sums for best_selected
            best_np = np.array(best_selected, dtype=np.int32)
            temp_sums = np.zeros(MAX_SUM + 1, dtype=np.bool_)
            n = len(best_np)
            for i in range(n):
                for j in range(i, n):
                    for k in range(j, n):
                        temp_sums[best_np[i] + best_np[j] + best_np[k]] = True
            
            for x in candidates:
                if time.time() > deadline - 3.0:
                    break
                
                # Check if x can be added to best_selected
                ok = True
                if temp_sums[3 * x]:
                    ok = False
                elif n > 0:
                    if np.any(temp_sums[2 * x + best_np]) or np.any(temp_sums[x + 2 * best_np]):
                        ok = False
                    elif np.any(temp_sums[x + best_np[np.triu_indices(n)[0]] + best_np[np.triu_indices(n)[1]]]):
                        ok = False
                
                if ok:
                    temp_selected = best_selected + [x]
                    if len(temp_selected) > best_size:
                        best_selected = temp_selected
                        best_size = len(temp_selected)
                        # Rebuild sums for new best
                        best_np = np.array(best_selected, dtype=np.int32)
                        temp_sums = np.zeros(MAX_SUM + 1, dtype=np.bool_)
                        n = len(best_np)
                        for i in range(n):
                            for j in range(i, n):
                                for k in range(j, n):
                                    temp_sums[best_np[i] + best_np[j] + best_np[k]] = True
                        break
    
    return best_selected[:80]
