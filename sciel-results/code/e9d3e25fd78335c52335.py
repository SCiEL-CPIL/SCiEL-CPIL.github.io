import itertools
import numpy as np
import random
import time

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
    
    # Precompute neighbors as numpy arrays for fast operations
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
    
    # Precompute neighbor sets as Python sets for fast intersection
    neighbor_sets = [set(arr.tolist()) for arr in neighbors]
    
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    def reduce_code(C, counts):
        """Peel: iteratively remove codewords that are redundant."""
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(counts[neighbors[c]]) >= 2:
                    counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return C, counts
    
    def randomized_greedy(rng):
        """Randomized greedy set cover with tie-breaking."""
        uncovered = np.ones(N, dtype=bool)
        C = []
        while np.any(uncovered):
            # Compute coverage counts for all candidates
            # Vectorized: for each candidate, count how many uncovered points it covers
            best_idx = -1
            best_count = -1
            # Sample a subset of candidates for speed, or use full scan
            # Use full scan but with vectorization
            # Count uncovered neighbors for each candidate
            # This is O(N * 17) which is ~111k ops per iteration
            # Number of iterations ~ 486, so ~54M ops total - too slow in Python
            # Instead, maintain a priority structure or use numpy
            # For speed, we'll use a sampling approach for ties
            # Compute all coverage counts at once using matrix multiply
            # uncovered is boolean array of size N
            # neighbors is list of arrays
            # We need: for each i, sum(uncovered[neighbors[i]])
            # This is equivalent to: for each j in uncovered, add 1 to all i where j in neighbors[i]
            # i.e., for each j in uncovered, add 1 to all i in neighbor_sets[j]
            # Let's use a different approach: maintain a count array
            # Actually, let's just do the direct computation with numpy
            # Precompute: neighbors_matrix is N x 17 binary matrix
            # But that's 6561*17 = 111k entries, fine
            # coverage = neighbors_matrix @ uncovered  (but neighbors_matrix is sparse)
            # Let's just loop but with numpy sums
            # Actually the original approach works, let's keep it but optimize
            # Use a precomputed neighbor matrix as sparse-like structure
            
            # For now, use the direct approach with numpy
            counts_arr = np.array([np.sum(uncovered[neighbors[i]]) for i in range(N)])
            
            if counts_arr.max() == 0:
                break
            
            # Find all candidates with max count
            max_count = counts_arr.max()
            max_indices = np.where(counts_arr == max_count)[0]
            
            if len(max_indices) == 1:
                best_idx = max_indices[0]
            else:
                # Random tie-breaking
                best_idx = max_indices[rng.randint(0, len(max_indices) - 1)]
            
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            
            if time.time() - start_time > time_limit:
                break
        
        return C
    
    def local_search(C, counts, time_budget):
        """Swap-based local search to improve the code."""
        best_C = C.copy()
        best_counts = counts.copy()
        best_size = len(C)
        local_start = time.time()
        
        while time.time() - local_start < time_budget:
            # Pass 1: Peeling
            C, counts = reduce_code(C, counts)
            
            if len(C) < best_size:
                best_size = len(C)
                best_C = C.copy()
                best_counts = counts.copy()
            
            if time.time() - local_start > time_budget:
                break
            
            # Pass 2: Try swaps
            improved = False
            indices = list(range(len(C)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() - local_start > time_budget:
                    break
                c = C[idx]
                u_indices = neighbors[c]
                unique_mask = counts[u_indices] == 1
                unique_points = u_indices[unique_mask]
                
                if len(unique_points) == 0:
                    # Remove
                    C.remove(c)
                    counts[neighbors[c]] -= 1
                    improved = True
                    break
                
                if len(unique_points) > 3:
                    continue
                
                # Find candidates that cover all unique points
                if len(unique_points) == 1:
                    candidates_set = set(neighbor_sets[unique_points[0]])
                else:
                    candidates_set = set(neighbor_sets[unique_points[0]])
                    for u in unique_points[1:]:
                        candidates_set &= neighbor_sets[u]
                        if not candidates_set:
                            break
                
                if not candidates_set:
                    continue
                
                candidates_set.difference_update(C)
                
                if not candidates_set:
                    continue
                
                # Pick candidate that maximizes overlap with existing code
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
                
                # Try to reduce
                new_C, new_counts = reduce_code(C, counts)
                
                if len(new_C) < len(old_C):
                    C = new_C
                    counts = new_counts
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_counts = counts.copy()
                    improved = True
                    break
                else:
                    # Keep the swap to escape local minimum
                    C = new_C
                    counts = new_counts
                    improved = True
                    break
            
            if not improved:
                # Try random kicks
                for _ in range(5):
                    if time.time() - local_start > time_budget:
                        break
                    if not C:
                        break
                    c_idx = random.randint(0, len(C) - 1)
                    c = C[c_idx]
                    cand = neighbors[c]
                    c_prime = cand[random.randint(0, len(cand) - 1)]
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
                    
                    # Try reduce
                    new_C, new_counts = reduce_code(C, counts)
                    
                    if len(new_C) < len(old_C):
                        C = new_C
                        counts = new_counts
                        if len(C) < best_size:
                            best_size = len(C)
                            best_C = C.copy()
                            best_counts = counts.copy()
                        break
                    else:
                        C = new_C
                        counts = new_counts
        
        return best_C, best_counts
    
    # Main loop: multiple randomized greedy restarts with local search
    best_C = None
    best_counts = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # First run: deterministic greedy (no random tie-breaking)
    uncovered = np.ones(N, dtype=bool)
    C = []
    while np.any(uncovered) and time.time() - start_time < time_limit:
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
    
    counts = np.zeros(N, dtype=int)
    for c in C:
        counts[neighbors[c]] += 1
    
    # Ensure coverage
    if np.any(counts < 1):
        while np.any(counts < 1) and time.time() - start_time < time_limit:
            uncovered_mask = counts < 1
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered_mask[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0:
                break
            C.append(best_idx)
            counts[neighbors[best_idx]] += 1
    
    # Reduce
    C, counts = reduce_code(C, counts)
    
    if len(C) < best_size:
        best_size = len(C)
        best_C = C.copy()
        best_counts = counts.copy()
    
    # Local search on the first solution
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        best_C, best_counts = local_search(best_C, best_counts, remaining_time * 0.6)
        if len(best_C) < best_size:
            best_size = len(best_C)
    
    # Multiple randomized restarts
    remaining_time = time_limit - (time.time() - start_time)
    num_restarts = min(10, max(1, int(remaining_time / 15)))
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Randomized greedy
        C = randomized_greedy(rng)
        
        if not C:
            continue
        
        counts = np.zeros(N, dtype=int)
        for c in C:
            counts[neighbors[c]] += 1
        
        # Ensure coverage
        if np.any(counts < 1):
            while np.any(counts < 1) and time.time() - start_time < time_limit - 5:
                uncovered_mask = counts < 1
                best_idx = -1
                best_count = -1
                for i in range(N):
                    cnt = np.sum(uncovered_mask[neighbors[i]])
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
                if best_count == 0:
                    break
                C.append(best_idx)
                counts[neighbors[best_idx]] += 1
        
        # Reduce
        C, counts = reduce_code(C, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
        
        # Local search if we have time
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 10 and len(C) <= best_size + 5:
            C, counts = local_search(C, counts, remaining_time * 0.5)
            if len(C) < best_size:
                best_size = len(C)
                best_C = C.copy()
                best_counts = counts.copy()
    
    # Final reduction
    if best_C is not None:
        final_counts = np.zeros(N, dtype=int)
        for c in best_C:
            final_counts[neighbors[c]] += 1
        best_C, final_counts = reduce_code(best_C, final_counts)
    
    if best_C is None:
        # Should not happen, but fallback
        best_C = list(range(N))
    
    return [list(all_words[i]) for i in best_C]
