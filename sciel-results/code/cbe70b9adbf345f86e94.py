import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

    total_space = q ** n
    
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    digits = np.zeros((total_space, n), dtype=np.int32)
    for pos in range(n):
        for i in range(total_space):
            digits[i, pos] = (i // powers[pos]) % q

    neighbors = []
    for i in range(total_space):
        nbrs = [i]
        for pos in range(n):
            dval = digits[i, pos]
            for nv in range(q):
                if nv == dval:
                    continue
                j = i + (nv - dval) * powers[pos]
                nbrs.append(j)
        neighbors.append(nbrs)

    mask = np.zeros((total_space, total_space), dtype=np.bool_)
    for i in range(total_space):
        mask[i, neighbors[i]] = True

    full_mask = np.ones(total_space, dtype=np.bool_)
    
    def run_greedy():
        covered = np.zeros(total_space, dtype=np.bool_)
        selected = []
        remaining = full_mask.copy()
        selected_set = set()
        
        while remaining.any() and time.time() < deadline - 10:
            gains = (mask & remaining[None, :]).sum(axis=1)
            gains[list(selected_set)] = -1
            best_word = int(np.argmax(gains))
            if gains[best_word] <= 0:
                break
            selected.append(best_word)
            selected_set.add(best_word)
            covered |= mask[best_word]
            remaining = full_mask & ~covered
        return selected

    def polish(selected_list):
        if not selected_list:
            return selected_list
        sel_arr = np.array(selected_list, dtype=np.int32)
        counts = mask[sel_arr].sum(axis=0).astype(np.int32)
        
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            sel_arr = np.array(selected_list, dtype=np.int32)
            m = mask[sel_arr]
            unique_pos = (counts == 1)
            has_unique = (m & unique_pos).any(axis=1)
            removable = ~has_unique
            
            if removable.any():
                keep_mask = np.ones(len(selected_list), dtype=np.bool_)
                keep_mask[removable] = False
                removed_arr = sel_arr[removable]
                counts -= mask[removed_arr].sum(axis=0).astype(np.int32)
                # Update list
                new_list = []
                for i, w in enumerate(selected_list):
                    if keep_mask[i]:
                        new_list.append(w)
                selected_list = new_list
                changed = True
        return selected_list

    sol = run_greedy()
    sol = polish(sol)
    best_size = len(sol)
    best_selected = sol[:]
    
    # State
    selected_list = sol[:]
    selected_bool = np.zeros(total_space, dtype=np.bool_)
    for w in selected_list:
        selected_bool[w] = True
    counts = mask[np.array(selected_list, dtype=np.int32)].sum(axis=0).astype(np.int32)
    U = np.sum(counts == 0)
    
    all_words = np.arange(total_space)
    
    # SA Parameters
    T = 5.0
    cooling = 0.995
    
    while time.time() < deadline - 5:
        # 1. Polish
        if U == 0:
            polished = polish(selected_list)
            if len(polished) < len(selected_list):
                selected_list = polished
                for i in range(len(selected_list)):
                    selected_bool[selected_list[i]] = True
                # Reset bools for removed
                # Easier: rebuild
                selected_bool = np.zeros(total_space, dtype=np.bool_)
                for w in selected_list:
                    selected_bool[w] = True
                counts = mask[np.array(selected_list, dtype=np.int32)].sum(axis=0).astype(np.int32)
                U = np.sum(counts == 0)
                if len(selected_list) < best_size:
                    best_size = len(selected_list)
                    best_selected = selected_list[:]

        # 2. 2-for-1 Search
        # Try to find a pair to remove and one to add
        # We sample pairs from selected_list
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            if len(selected_list) < 2:
                break
            idx1 = random.randint(0, len(selected_list)-1)
            idx2 = random.randint(0, len(selected_list)-1)
            if idx1 == idx2:
                continue
            w1 = selected_list[idx1]
            w2 = selected_list[idx2]
            
            # Check feasibility of removing w1, w2
            mask_sum = mask[w1].astype(np.int8) + mask[w2].astype(np.int8)
            new_counts = counts.astype(np.int8) - mask_sum
            uncovered_mask = (new_counts == 0)
            
            if not uncovered_mask.any():
                # 2-for-0
                # Remove w1, w2
                counts -= mask[w1]
                counts -= mask[w2]
                # Update list
                # Remove by index
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                selected_list.pop(idx2)
                selected_list.pop(idx1)
                selected_bool[w1] = False
                selected_bool[w2] = False
                U = np.sum(counts == 0)
                if len(selected_list) < best_size:
                    best_size = len(selected_list)
                    best_selected = selected_list[:]
                continue
            
            # Find w3
            valid_w3 = (mask[:, uncovered_mask]).all(axis=1)
            valid_w3[selected_bool] = False
            
            if valid_w3.any():
                candidates = np.where(valid_w3)[0]
                w3 = candidates[random.randint(0, len(candidates)-1)]
                
                # Perform swap
                counts -= mask[w1]
                counts -= mask[w2]
                counts += mask[w3]
                
                # Update list
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                selected_list.pop(idx2)
                selected_list.pop(idx1)
                selected_list.append(w3)
                selected_bool[w1] = False
                selected_bool[w2] = False
                selected_bool[w3] = True
                
                U = np.sum(counts == 0)
                
                if U == 0 and len(selected_list) < best_size:
                    best_size = len(selected_list)
                    best_selected = selected_list[:]

        # 3. 1-for-1 Swap (Escape)
        if time.time() < deadline - 5 and len(selected_list) > 0:
            idx = random.randint(0, len(selected_list)-1)
            w1 = selected_list[idx]
            
            unique_pos = (counts == 1) & mask[w1]
            
            if unique_pos.any():
                valid_w2 = (mask[:, unique_pos]).all(axis=1)
                valid_w2[selected_bool] = False
                if valid_w2.any():
                    candidates = np.where(valid_w2)[0]
                    w2 = candidates[random.randint(0, len(candidates)-1)]
                    
                    # Check new U
                    counts -= mask[w1]
                    counts += mask[w2]
                    new_U = np.sum(counts == 0)
                    
                    # Accept if new_U == 0 or with prob
                    if new_U == 0:
                        selected_list[idx] = w2
                        selected_bool[w1] = False
                        selected_bool[w2] = True
                        U = new_U
                    else:
                        # Rollback
                        counts -= mask[w2]
                        counts += mask[w1]
                        # U unchanged

        T *= cooling

    return [int_to_tuple(v) for v in best_selected]
