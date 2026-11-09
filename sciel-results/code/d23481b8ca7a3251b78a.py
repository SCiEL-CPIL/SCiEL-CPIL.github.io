import numpy as np
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 620  # Safety margin for 630s limit
    
    def generate_candidates(R):
        """Generate all integer vectors with squared norm <= R"""
        candidates = []
        
        # Norm 1: e_i, -e_i
        for i in range(d):
            v = [0]*d
            v[i] = 1
            candidates.append(tuple(v))
            v[i] = -1
            candidates.append(tuple(v))
        
        # Norm 2: e_i +/- e_j
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = [0]*d
                        v[i] = s1
                        v[j] = s2
                        candidates.append(tuple(v))
        
        # Norm 3: e_i +/- e_j +/- e_k
        if R >= 3:
            for i in range(d):
                for j in range(i + 1, d):
                    for k in range(j + 1, d):
                        for s1 in [1, -1]:
                            for s2 in [1, -1]:
                                for s3 in [1, -1]:
                                    v = [0]*d
                                    v[i] = s1
                                    v[j] = s2
                                    v[k] = s3
                                    candidates.append(tuple(v))
        
        # Norm 4: 2e_i and e_i +/- e_j +/- e_k +/- e_l
        if R >= 4:
            for i in range(d):
                v = [0]*d
                v[i] = 2
                candidates.append(tuple(v))
                v[i] = -2
                candidates.append(tuple(v))
            for i in range(d):
                for j in range(i + 1, d):
                    for k in range(j + 1, d):
                        for l in range(k + 1, d):
                            for s1 in [1, -1]:
                                for s2 in [1, -1]:
                                    for s3 in [1, -1]:
                                        for s4 in [1, -1]:
                                            v = [0]*d
                                            v[i] = s1
                                            v[j] = s2
                                            v[k] = s3
                                            v[l] = s4
                                            candidates.append(tuple(v))
        
        # Remove duplicates
        candidates = list(set(candidates))
        arr = np.array(candidates, dtype=np.int64)
        return arr
    
    def solve_with_local_search(candidates, R, time_limit=30):
        """
        Find maximum set of points with pairwise dist_sq >= R.
        Uses greedy initialization + aggressive local search with swaps.
        """
        n = len(candidates)
        if n < 2:
            return []
        
        norms_sq = np.sum(candidates**2, axis=1)
        valid_mask = norms_sq <= R
        if not np.all(valid_mask):
            candidates = candidates[valid_mask]
            norms_sq = norms_sq[valid_mask]
            n = len(candidates)
            if n < 2:
                return []
        
        # Precompute conflict matrix
        dots = candidates @ candidates.T
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        conflict = dist_sq < R
        np.fill_diagonal(conflict, False)
        
        # Greedy initialization: pick points in order of minimum conflict degree
        deg = conflict.sum(axis=1)
        order = np.argsort(deg)
        
        selected = set()
        for idx in order:
            # Check if idx conflicts with any selected
            can_add = True
            for s in selected:
                if conflict[idx, s]:
                    can_add = False
                    break
            if can_add:
                selected.add(idx)
        
        best_set = set(selected)
        best_size = len(selected)
        
        # Local search: try swaps and additions
        end_time = time.time() + time_limit
        improved = True
        while improved and time.time() < end_time:
            improved = False
            
            # Try adding any non-selected point
            for idx in range(n):
                if idx in best_set:
                    continue
                can_add = True
                for s in best_set:
                    if conflict[idx, s]:
                        can_add = False
                        break
                if can_add:
                    best_set.add(idx)
                    best_size += 1
                    improved = True
            
            # Try 1-for-1 swaps
            if time.time() > end_time:
                break
            for remove_idx in list(best_set):
                for add_idx in range(n):
                    if add_idx in best_set or add_idx == remove_idx:
                        continue
                    # Check if add_idx conflicts with (best_set - {remove_idx})
                    can_add = True
                    for s in best_set:
                        if s != remove_idx and conflict[add_idx, s]:
                            can_add = False
                            break
                    if can_add:
                        # This swap is neutral (same size), but might enable further improvements
                        # Only do it if it reduces total conflict potential
                        old_conflict = sum(conflict[remove_idx, s] for s in best_set if s != remove_idx)
                        new_conflict = sum(conflict[add_idx, s] for s in best_set if s != remove_idx)
                        if new_conflict < old_conflict:
                            best_set.discard(remove_idx)
                            best_set.add(add_idx)
                            improved = True
                            break
                if improved:
                    break
            
            # Try 1-for-2 swaps (remove one, add two)
            if time.time() > end_time:
                break
            for remove_idx in list(best_set):
                remaining = best_set - {remove_idx}
                # Find up to 2 points that can be added
                addable = []
                for idx in range(n):
                    if idx in remaining or idx == remove_idx:
                        continue
                    can_add = True
                    for s in remaining:
                        if conflict[idx, s]:
                            can_add = False
                            break
                    if can_add:
                        addable.append(idx)
                        if len(addable) >= 2:
                            break
                if len(addable) >= 2:
                    best_set.discard(remove_idx)
                    for idx in addable:
                        best_set.add(idx)
                    best_size = len(best_set)
                    improved = True
                    break
            
            # Try 2-for-1 swaps (remove two, add one)
            if time.time() > end_time:
                break
            best_list = list(best_set)
            for i in range(len(best_list)):
                for j in range(i + 1, len(best_list)):
                    remove1, remove2 = best_list[i], best_list[j]
                    remaining = best_set - {remove1, remove2}
                    for idx in range(n):
                        if idx in remaining or idx == remove1 or idx == remove2:
                            continue
                        can_add = True
                        for s in remaining:
                            if conflict[idx, s]:
                                can_add = False
                                break
                        if can_add:
                            best_set.discard(remove1)
                            best_set.discard(remove2)
                            best_set.add(idx)
                            best_size = len(best_set)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return list(best_set)
    
    best_set_global = []
    best_size_global = 0
    
    for R in [3, 4, 5]:
        if time.time() > deadline - 15:
            break
        
        candidates = generate_candidates(R)
        if len(candidates) < 2:
            continue
        
        result_indices = solve_with_local_search(candidates, R, time_limit=180)
        
        if len(result_indices) > best_size_global:
            best_size_global = len(result_indices)
            best_set_global = [candidates[i] for i in result_indices]
    
    if best_size_global >= 2:
        result = np.array(best_set_global, dtype=np.int64)
        max_norm = np.max(np.sum(result**2, axis=1))
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                d2 = np.dot(result[i] - result[j], result[i] - result[j])
                if d2 < max_norm:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return result
    
    return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

if __name__ == "__main__":
    res = kissing_number11()
    print(f"Size: {len(res)}")
