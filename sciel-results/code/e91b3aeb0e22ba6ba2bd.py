import itertools
import numpy as np
import random
import time
import heapq

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters
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
            if best_count == 0: break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit: break
            
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
    
    # Precompute neighbors as lists of ints for faster iteration if needed, 
    # but numpy arrays are good for vectorized ops.
    # For heap greedy, we need to calculate "gain" which is sum of uncovered neighbors.
    
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
        
    start_time = time.time()
    time_limit = 115.0
    
    best_size = float('inf')
    best_C = []
    
    # Multiple restarts
    # Each restart: Greedy + Local Search
    # Greedy with heap is O(N log N + K * N) roughly? 
    # Actually, updating the heap is the tricky part. 
    # A simpler "Fast Greedy" without heap but with numpy vectorization might be faster in Python due to overhead.
    # Let's stick to the vectorized scan but optimize the local search.
    # The vectorized scan: for each of ~400 steps, scan 6561 words. 400 * 6561 * (cost of sum) ~ 2.6M ops. Fast.
    # The bottleneck is likely the local search.
    
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

    num_restarts = 5 # Adjust based on time
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 20: # Leave 20s for last run
            break
            
        # 1. Greedy
        uncovered = np.ones(N, dtype=bool)
        C = []
        
        while np.any(uncovered):
            # Vectorized gain calculation
            # gain[i] = sum(uncovered[neighbors[i]])
            # This is a matrix multiplication if neighbors were a matrix, but they are sparse.
            # Simple loop is often faster in Python for small N if numpy overhead is high?
            # N=6561. neighbors[i] has 17 elements.
            # 6561 * 17 ~ 110k ops per step.
            # 400 steps * 110k = 44M ops.
            # This might be too slow in pure Python loop inside numpy?
            # Let's try to optimize the greedy step.
            
            # Alternative: 
            # We can maintain a list of "active" candidates?
            # Or just use the simple loop. 
            # 110k ops is very fast in C (numpy), but the loop is in Python.
            # for i in range(6561): cnt = np.sum(uncovered[neighbors[i]])
            # This involves 6561 numpy calls. Each call has overhead.
            # Better: 
            # Use a matrix? 6561x6561 is 43M ints. 344MB. Too big? Maybe.
            # Let's stick to the loop but check time.
            
            best_idx = -1
            best_count = -1
            # Optimization: Only check if uncovered exists? Yes, loop condition.
            
            # To speed up, we can sample or use a heuristic?
            # No, greedy needs exact best.
            
            # Let's try to reduce the constant factor.
            # neighbors is a list of arrays.
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            
            if best_count == 0:
                break
            
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            
            if time.time() - start_time > time_limit - 10:
                break
                
        if time.time() - start_time > time_limit - 5:
            break

        # 2. Local Search
        counts = get_counts(C)
        C, counts = reduce_code(C, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            
        # Local Search Loop
        # We have limited time per restart.
        # Estimate time per restart: Greedy ~ 2-5s, LS ~ ?
        # If we have 115s total, and 5 restarts, we have ~20s per restart.
        # Greedy takes ~5s. So 15s for LS.
        
        ls_start = time.time()
        ls_limit = min(time_limit - 5 - (time.time() - start_time), 15.0)
        
        while time.time() - ls_start < ls_limit:
            old_size = len(C)
            C, counts = reduce_code(C, counts)
            
            if len(C) < best_size:
                best_size = len(C)
                best_C = C.copy()
                
            if len(C) == old_size:
                # No reduction. Try swap.
                improved = False
                indices = list(range(len(C)))
                random.shuffle(indices)
                
                for idx in indices:
                    if time.time() - ls_start > ls_limit:
                        break
                    c = C[idx]
                    u_indices = neighbors[c]
                    unique_mask = counts[u_indices] == 1
                    unique_points = u_indices[unique_mask]
                    
                    if len(unique_points) == 0:
                        # Should be removed
                        C.remove(c)
                        counts[neighbors[c]] -= 1
                        improved = True
                        break
                    
                    if len(unique_points) > 3:
                        continue
                    
                    # Find candidates
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
                    
                    c_prime = random.choice(list(candidates_set))
                    
                    old_C = C.copy()
                    old_counts = counts.copy()
                    
                    C.remove(c)
                    C.append(c_prime)
                    counts[neighbors[c]] -= 1
                    counts[neighbors[c_prime]] += 1
                    
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
                        improved = True
                        break
                    else:
                        # Accept neutral swap to escape
                        C = new_C
                        counts = new_counts
                        improved = True
                        break
                
                if not improved:
                    # Kick
                    if C:
                        c_idx = random.randint(0, len(C)-1)
                        c = C[c_idx]
                        cand = neighbors[c]
                        c_prime = cand[random.randint(0, len(cand)-1)]
                        if c_prime in C:
                            continue
                        
                        old_C = C.copy()
                        old_counts = counts.copy()
                        
                        C.remove(c)
                        C.append(c_prime)
                        counts[neighbors[c]] -= 1
                        counts[neighbors[c_prime]] += 1
                        
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
                        else:
                            C = new_C
                            counts = new_counts

    # Final Reduction
    if best_C:
        best_C, _ = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
