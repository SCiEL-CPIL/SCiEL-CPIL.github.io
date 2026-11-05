import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters: simple greedy
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            neigh_set = set()
            neigh_set.add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    neigh_set.add(word_to_idx[new_w])
            neighbors.append(list(neigh_set))
        
        N = q ** n
        uncovered = np.ones(N, dtype=bool)
        C = []
        start_time = time.time()
        time_limit = 100.0
        
        while np.any(uncovered):
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0:
                break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit:
                break
            
        # Reduction
        coverage_counts = np.zeros(N, dtype=int)
        for c in C:
            coverage_counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(coverage_counts[neighbors[c]]) >= 2:
                    coverage_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(all_words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as numpy arrays
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_set = set()
        neigh_set.add(i)
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_set.add(word_to_idx[new_w])
        neighbors.append(np.array(list(neigh_set), dtype=int))
    
    # Start with a greedy construction to get an initial code
    uncovered = np.ones(N, dtype=bool)
    C = []
    start_time = time.time()
    time_limit = 115.0
    
    while np.any(uncovered):
        best_idx = -1
        best_count = -1
        for i in range(N):
            cnt = np.sum(uncovered[neighbors[i]])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_count == 0:
            break
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        if time.time() - start_time > time_limit:
            break
    
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts
    
    def reduce_code(code, counts):
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        
        while True:
            removed_any = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors[c]]) >= 2:
                    current_counts[neighbors[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts
    
    C, counts = reduce_code(C, get_counts(C))
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()
    
    # Local Search Loop with multiple swap attempts per iteration
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() - start_time < time_limit:
        # 1. Reduction Pass
        C, counts = reduce_code(C, counts)
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        if time.time() - start_time > time_limit:
            break
        
        # 2. Swap Pass - try multiple swaps in one pass
        improved = False
        swaps_made = 0
        max_swaps_per_iter = 5
        
        for c in C[:]:  # Iterate over a copy
            if time.time() - start_time > time_limit or swaps_made >= max_swaps_per_iter:
                break
                
            u_indices = neighbors[c]
            unique_mask = counts[u_indices] == 1
            unique_points = u_indices[unique_mask]
            
            if len(unique_points) == 0:
                continue
            
            if len(unique_points) > 3:
                continue
            
            # Find candidates c' that cover all unique_points
            if len(unique_points) == 1:
                candidates_set = set(neighbors[unique_points[0]].tolist())
            else:
                candidates_set = set(neighbors[unique_points[0]].tolist())
                for u in unique_points[1:]:
                    candidates_set &= set(neighbors[u].tolist())
                    if not candidates_set:
                        break
            
            if not candidates_set:
                continue
            
            candidates_set.difference_update(C)
            
            if not candidates_set:
                continue
            
            # Pick a candidate - try the one with highest overlap with C
            best_c_prime = None
            max_overlap = -1
            for c_prime in candidates_set:
                overlap = np.sum(counts[neighbors[c_prime]] > 0)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_c_prime = c_prime
            
            if best_c_prime is None:
                continue
                
            # Apply swap
            old_C = C.copy()
            old_counts = counts.copy()
            
            C.remove(c)
            C.append(best_c_prime)
            
            counts[neighbors[c]] -= 1
            counts[neighbors[best_c_prime]] += 1
            
            if np.any(counts < 1):
                C = old_C
                counts = old_counts
                continue
                
            new_C, new_counts = reduce_code(C, counts)
            
            if len(new_C) < len(old_C):
                C = new_C
                counts = new_counts
                if len(C) < best_size:
                    best_size = len(C)
                    best_C = C.copy()
                    best_counts = counts.copy()
                    no_improvement_count = 0
                improved = True
                swaps_made += 1
            else:
                # Keep neutral swap to change structure
                C = new_C
                counts = new_counts
                swaps_made += 1
        
        # 3. Perturbation if stuck
        if not improved and no_improvement_count > max_no_improvement:
            # Remove 3 random codewords and re-greedy fill
            if len(C) >= 3:
                to_remove = random.sample(C, 3)
                for c in to_remove:
                    C.remove(c)
                    counts[neighbors[c]] -= 1
                
                # Re-greedy fill
                uncovered = counts < 1
                while np.any(uncovered):
                    best_idx = -1
                    best_count = -1
                    for i in range(N):
                        if i in C:
                            continue
                        cnt = np.sum(uncovered[neighbors[i]])
                        if cnt > best_count:
                            best_count = cnt
                            best_idx = i
                    if best_count == 0:
                        break
                    C.append(best_idx)
                    counts[neighbors[best_idx]] += 1
                    uncovered = counts < 1
                
                C, counts = reduce_code(C, counts)
                no_improvement_count = 0
    
    # Final Reduction
    best_C, _ = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
