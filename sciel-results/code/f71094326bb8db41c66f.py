"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill

    space_size = q ** n

    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx, q, n):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute all neighbors (words within radius 1) for every word in the space
    neighbors = []
    for idx in range(space_size):
        w = idx_to_word(idx, q, n)
        nbrs = [idx]
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for v in range(q):
                    if v == orig:
                        continue
                    new_w = w.copy()
                    new_w[pos] = v
                    nbrs.append(word_to_idx(new_w))
        neighbors.append(nbrs)

    # Convert neighbors to numpy arrays for vectorized operations
    neighbors_np = np.array(neighbors, dtype=np.int32)
    # neighbors_np[i] is the list of indices covered by codeword i

    # Greedy set cover with vectorized counting
    def greedy_set_cover(seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = np.ones(space_size, dtype=bool)
        selected_indices = []
        
        while uncovered.any():
            if time.time() > deadline - 15:
                break
            
            # For each candidate, count how many uncovered words it covers
            # Use matrix multiplication style: for each candidate, sum over its neighbors
            # that are uncovered
            
            # Vectorized approach: for each candidate i, count = sum(uncovered[neighbors_np[i]])
            # We can do this efficiently by iterating candidates but using numpy for the inner sum
            
            best_idx = -1
            best_count = -1
            
            # Process in batches for speed
            batch_size = 100
            for start in range(0, space_size, batch_size):
                end = min(start + batch_size, space_size)
                # For each candidate in this batch, compute coverage count
                for cand_idx in range(start, end):
                    count = uncovered[neighbors_np[cand_idx]].sum()
                    if count > best_count:
                        best_count = count
                        best_idx = cand_idx
            
            if best_idx == -1 or best_count <= 0:
                break
            
            selected_indices.append(best_idx)
            uncovered[neighbors_np[best_idx]] = False
        
        return selected_indices

    # Iterative removal of redundant codewords
    def remove_redundant(selected_indices, max_passes=100):
        if not selected_indices:
            return selected_indices
        
        coverage_count = np.zeros(space_size, dtype=np.int32)
        for ci in selected_indices:
            coverage_count[neighbors_np[ci]] += 1
        
        removed = [False] * len(selected_indices)
        
        for _pass in range(max_passes):
            if time.time() > deadline - 15:
                break
            removed_any = False
            for ci in range(len(selected_indices)):
                if removed[ci]:
                    continue
                # Check if removing selected[ci] leaves any word uncovered
                can_remove = True
                for nbr_idx in neighbors_np[selected_indices[ci]]:
                    if coverage_count[nbr_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    removed[ci] = True
                    removed_any = True
                    coverage_count[neighbors_np[selected_indices[ci]]] -= 1
            if not removed_any:
                break
        
        return [selected_indices[i] for i in range(len(selected_indices)) if not removed[i]]

    # Local search: try swapping codewords to reduce size
    def local_search(selected_indices, max_iterations=500):
        if not selected_indices:
            return selected_indices
        
        coverage_count = np.zeros(space_size, dtype=np.int32)
        for ci in selected_indices:
            coverage_count[neighbors_np[ci]] += 1
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            improved = False
            
            # Try adding a random candidate and removing 2+ existing
            for trial in range(20):
                if time.time() > deadline - 10:
                    break
                
                cand_idx = random.randint(0, space_size - 1)
                if cand_idx in selected_indices:
                    continue
                
                # Simulate adding cand_idx
                new_cov = coverage_count.copy()
                new_cov[neighbors_np[cand_idx]] += 1
                
                # Try to greedily remove codewords
                temp_removed = [False] * len(selected_indices)
                removable = 0
                for ri, rcw_idx in enumerate(selected_indices):
                    if temp_removed[ri]:
                        continue
                    can_remove = True
                    for nbr_idx in neighbors_np[rcw_idx]:
                        if new_cov[nbr_idx] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        temp_removed[ri] = True
                        removable += 1
                        new_cov[neighbors_np[rcw_idx]] -= 1
                
                if removable >= 2:
                    # Apply
                    selected_indices = [selected_indices[ri] for ri in range(len(selected_indices)) if not temp_removed[ri]]
                    selected_indices.append(cand_idx)
                    coverage_count = new_cov
                    improved = True
                    break
            
            # Also try simple removals
            if not improved:
                removed_any = False
                for ri in range(len(selected_indices)):
                    if time.time() > deadline - 10:
                        break
                    # Check if removing selected_indices[ri] leaves any word uncovered
                    can_remove = True
                    for nbr_idx in neighbors_np[selected_indices[ri]]:
                        if coverage_count[nbr_idx] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        # Remove it
                        coverage_count[neighbors_np[selected_indices[ri]]] -= 1
                        selected_indices.pop(ri)
                        removed_any = True
                        break
                if removed_any:
                    improved = True
            
            if not improved:
                break
        
        return selected_indices

    # Main search: multiple restarts
    best_solution = None
    best_size = float('inf')
    
    # First, do a deterministic greedy
    greedy_sol = greedy_set_cover(seed=None)
    greedy_sol = remove_redundant(greedy_sol)
    greedy_sol = local_search(greedy_sol)
    
    if len(greedy_sol) < best_size:
        best_size = len(greedy_sol)
        best_solution = greedy_sol
    
    # Multiple randomized restarts
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        seed = random.randint(0, 1000000)
        selected = greedy_set_cover(seed=seed)
        
        if not selected:
            continue
        
        selected = remove_redundant(selected)
        selected = local_search(selected)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = selected
            
            # Early stop if we hit the known optimal
            if best_size <= 73:
                if best_size == 71:
                    break
                # Try a few more to see if we can get 72
                if r >= 10:
                    break
    
    # Fallback if no solution found (should not happen)
    if best_solution is None:
        # Use direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    else:
        result = [idx_to_word(idx, q, n) for idx in best_solution]
        
    # Final validity check
    final_cov = np.zeros(space_size, dtype=np.int32)
    for cw in result:
        ci = word_to_idx(cw)
        final_cov[neighbors_np[ci]] += 1
    
    if not (final_cov > 0).all():
        # Fallback to tetracode direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    return result
# EVOLVE-BLOCK-END
