import time
import random
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 75  # Safety margin before 90s kill

    def is_comparable(a, b):
        if a == b:
            return False
        return (a & ~b) == 0 or (b & ~a) == 0

    def count_comparable_pairs(config):
        cnt = 0
        for i in range(len(config)):
            for j in range(i + 1, len(config)):
                if is_comparable(config[i], config[j]):
                    cnt += 1
        return cnt

    # Precompute comparability as a boolean matrix (4096 x 4096) using numpy
    # comp_mat[i][j] = True if mask i is comparable to mask j (and i != j)
    masks = np.arange(total_masks, dtype=np.int32)
    # For each pair, check if one is subset of the other
    # a subset b iff (a & ~b) == 0
    # We can vectorize: for all a, b: (a & ~b) == 0 or (b & ~a) == 0
    # This is 4096*4096 = 16M entries, manageable
    # Build in chunks to manage memory
    comp_mat = np.zeros((total_masks, total_masks), dtype=bool)
    chunk_size = 512
    for start in range(0, total_masks, chunk_size):
        end = min(start + chunk_size, total_masks)
        a_chunk = masks[start:end, np.newaxis]  # (chunk, 1)
        b_chunk = masks[np.newaxis, :]  # (1, total)
        # a subset b: (a & ~b) == 0
        a_sub_b = (a_chunk & ~b_chunk) == 0
        b_sub_a = (b_chunk & ~a_chunk) == 0
        comp_mat[start:end, :] = (a_sub_b | b_sub_a)
    # Remove self-comparability (diagonal)
    np.fill_diagonal(comp_mat, False)
    
    # Convert to int array for fast summing
    comp_int = comp_mat.astype(np.int32)

    # Greedy Initialization using numpy
    selected = []
    available = set(range(total_masks))
    
    for _ in range(m):
        best_mask = -1
        best_score = -1
        for mask in available:
            score = int(comp_int[mask, selected].sum()) if selected else 0
            if score > best_score:
                best_score = score
                best_mask = mask
        selected.append(best_mask)
        available.remove(best_mask)
        
    current_score = count_comparable_pairs(selected)
    best_config = selected[:]
    best_score = current_score
    
    # Build compat array: compat[mask] = number of selected sets comparable to mask
    def build_compat(selected_arr):
        sel_np = np.array(selected_arr, dtype=np.int32)
        # For each mask, count how many selected are comparable
        # comp_int[mask, sel] gives boolean for each selected
        compat = comp_int[:, sel_np].sum(axis=1)
        return compat
    
    compat = build_compat(selected)
    selected_set = set(selected)
    
    no_improve_passes = 0
    max_no_improve = 30
    
    while time.time() < deadline:
        improved = False
        # Iterate through indices for steepest descent
        for i in range(m):
            old_mask = selected[i]
            loss = int(compat[old_mask])
            
            # Find best new_mask: maximize compat[new_mask] - loss, excluding selected and old_mask
            # We need to exclude masks that are already in selected_set
            # Create a mask of valid candidates
            valid = np.ones(total_masks, dtype=bool)
            valid[old_mask] = False
            for s in selected_set:
                valid[s] = False
            
            gains = compat[valid]
            if len(gains) == 0:
                continue
            best_idx = int(np.argmax(gains))
            best_delta = int(gains[best_idx]) - loss
            
            if best_delta <= 0:
                continue
            
            # Find the actual mask
            valid_masks = np.where(valid)[0]
            best_new_mask = int(valid_masks[best_idx])
            
            # Perform swap
            selected_set.remove(old_mask)
            selected_set.add(best_new_mask)
            selected[i] = best_new_mask
            current_score += best_delta
            
            # Update compat array using numpy vectorized ops
            # compat -= comp_int[:, old_mask] (removing old_mask's contribution)
            # compat += comp_int[:, best_new_mask] (adding new_mask's contribution)
            compat -= comp_int[:, old_mask]
            compat += comp_int[:, best_new_mask]
            
            improved = True
        
        if current_score > best_score:
            best_score = current_score
            best_config = selected[:]
        
        if improved:
            no_improve_passes = 0
        else:
            no_improve_passes += 1
            if no_improve_passes >= max_no_improve:
                # Perturbation: try multiple random 2-swap perturbations
                success = False
                for _attempt in range(50):
                    i1 = random.randint(0, m - 1)
                    i2 = random.randint(0, m - 1)
                    if i1 == i2:
                        continue
                    old1 = selected[i1]
                    old2 = selected[i2]
                    
                    # Pick random masks not in selected_set
                    m1 = random.randint(0, total_masks - 1)
                    m2 = random.randint(0, total_masks - 1)
                    if m1 in selected_set or m2 in selected_set:
                        continue
                    if m1 == m2:
                        continue
                    if m1 == old1 or m2 == old2:
                        continue
                    
                    old_score = current_score
                    selected_set.remove(old1)
                    selected_set.remove(old2)
                    selected_set.add(m1)
                    selected_set.add(m2)
                    selected[i1] = m1
                    selected[i2] = m2
                    new_score = count_comparable_pairs(selected)
                    
                    if new_score > old_score:
                        current_score = new_score
                        compat = build_compat(selected)
                        no_improve_passes = 0
                        success = True
                        break
                    else:
                        selected_set.remove(m1)
                        selected_set.remove(m2)
                        selected_set.add(old1)
                        selected_set.add(old2)
                        selected[i1] = old1
                        selected[i2] = old2
                        current_score = old_score
                if not success:
                    # Try single-swap perturbations
                    success2 = False
                    for _attempt in range(100):
                        i1 = random.randint(0, m - 1)
                        old1 = selected[i1]
                        m1 = random.randint(0, total_masks - 1)
                        if m1 in selected_set or m1 == old1:
                            continue
                        
                        old_score = current_score
                        selected_set.remove(old1)
                        selected_set.add(m1)
                        selected[i1] = m1
                        new_score = count_comparable_pairs(selected)
                        
                        if new_score > old_score:
                            current_score = new_score
                            compat = build_compat(selected)
                            no_improve_passes = 0
                            success2 = True
                            break
                        else:
                            selected_set.remove(m1)
                            selected_set.add(old1)
                            selected[i1] = old1
                            current_score = old_score
                    if not success2:
                        break
    
    # Convert to lists of ints
    result = []
    for mask in best_config:
        s = [i for i in range(n) if (mask >> i) & 1]
        result.append(s)
        
    return result
