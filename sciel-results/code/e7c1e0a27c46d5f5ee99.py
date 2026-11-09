import numpy as np
import time
import random
import sys
import heapq
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 620  # Safety margin for 630s limit

    def generate_candidates(R):
        """Generate all integer vectors with squared norm <= R"""
        candidates = set()
        
        # Norm 1: e_i, -e_i
        for i in range(d):
            v = [0]*d
            v[i] = 1
            candidates.add(tuple(v))
            v[i] = -1
            candidates.add(tuple(v))
        
        # Norm 2: e_i +/- e_j
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = [0]*d
                        v[i] = s1
                        v[j] = s2
                        candidates.add(tuple(v))
        
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
                                    candidates.add(tuple(v))
        
        # Norm 4: 2e_i and e_i +/- e_j +/- e_k +/- e_l
        if R >= 4:
            # Type 1: 2e_i
            for i in range(d):
                v = [0]*d
                v[i] = 2
                candidates.add(tuple(v))
                v[i] = -2
                candidates.add(tuple(v))
            # Type 2: e_i +/- e_j +/- e_k +/- e_l
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
                                            candidates.add(tuple(v))
        
        # Convert to numpy array
        if not candidates:
            return np.array([], dtype=np.int64).reshape(0, d)
        
        arr = np.array(list(candidates), dtype=np.int64)
        return arr

    def solve_mis(candidates, R, time_limit=10):
        """
        Find Maximum Independent Set in the conflict graph.
        Two vertices conflict if their squared distance < R.
        """
        n = len(candidates)
        if n < 2:
            return []
        
        # Compute norms
        norms_sq = np.sum(candidates**2, axis=1)
        
        # Filter for valid candidates (norm_sq <= R)
        valid_mask = norms_sq <= R
        if not np.all(valid_mask):
            candidates = candidates[valid_mask]
            norms_sq = norms_sq[valid_mask]
            n = len(candidates)
            if n < 2:
                return []
        
        # Compute distance matrix
        dots = candidates @ candidates.T
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        
        # Conflict if dist_sq < R
        conflict_mask = dist_sq < R
        np.fill_diagonal(conflict_mask, False)
        
        deg = conflict_mask.sum(axis=1)
        
        # Multiple strategies
        best_set = []
        best_size = 0
        
        # Strategy 1: Min-degree greedy
        sorted_indices = np.argsort(deg)
        available = np.ones(n, dtype=bool)
        selected = []
        
        while available.any():
            pick_idx = -1
            for idx in sorted_indices:
                if available[idx]:
                    pick_idx = idx
                    break
            
            if pick_idx == -1:
                break
            
            selected.append(pick_idx)
            available[pick_idx] = False
            neighbors = np.where(conflict_mask[pick_idx])[0]
            available[neighbors] = False
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
        
        # Strategy 2: Randomized greedy with restarts
        num_restarts = 50
        for restart in range(num_restarts):
            if time.time() > start_time + time_limit:
                break
            
            available = np.ones(n, dtype=bool)
            selected = []
            
            # Randomly shuffle the order of consideration
            order = np.random.permutation(n)
            
            while available.any():
                # Pick first available in random order
                pick_idx = -1
                for idx in order:
                    if available[idx]:
                        pick_idx = idx
                        break
                
                if pick_idx == -1:
                    break
                
                selected.append(pick_idx)
                available[pick_idx] = False
                neighbors = np.where(conflict_mask[pick_idx])[0]
                available[neighbors] = False
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected[:]
        
        # Strategy 3: Local search / improvement
        if best_size > 0:
            current_set = set(best_set)
            current_size = len(current_set)
            
            # Try to add points
            for idx in range(n):
                if idx in current_set:
                    continue
                
                # Check if idx conflicts with any point in current_set
                can_add = True
                for other in current_set:
                    if conflict_mask[idx, other]:
                        can_add = False
                        break
                
                if can_add:
                    current_set.add(idx)
                    current_size += 1
            
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            
            # Try to remove one point and add two
            for remove_idx in list(current_set):
                test_set = current_set - {remove_idx}
                added = 0
                for idx in range(n):
                    if idx in test_set or idx == remove_idx:
                        continue
                    
                    can_add = True
                    for other in test_set:
                        if conflict_mask[idx, other]:
                            can_add = False
                            break
                    
                    if can_add:
                        test_set.add(idx)
                        added += 1
                        if added >= 2:
                            break
                
                if len(test_set) > best_size:
                    best_size = len(test_set)
                    best_set = list(test_set)
                    current_set = test_set
        
        return best_set

    # Main search loop
    best_set_global = []
    best_size_global = 0
    
    # Try different R values
    for R in [3, 4, 5]:
        if time.time() > deadline - 10:
            break
        
        candidates = generate_candidates(R)
        if len(candidates) < 2:
            continue
        
        # Solve MIS
        result_indices = solve_mis(candidates, R, time_limit=120)
        
        if len(result_indices) > best_size_global:
            best_size_global = len(result_indices)
            best_set_global = [candidates[i] for i in result_indices]
    
    # Verify and return
    if best_size_global >= 2:
        result = np.array(best_set_global, dtype=np.int64)
        # Final verification
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
    
    # Fallback
    return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

if __name__ == "__main__":
    res = kissing_number11()
    print(f"Size: {len(res)}")
