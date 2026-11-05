import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters
        # Simple greedy for other cases, not optimized
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
            neighbors.append(np.array(list(neigh_set), dtype=int))
        
        N = q ** n
        uncovered = np.ones(N, dtype=bool)
        C = []
        start_time = time.time()
        time_limit = 100.0
        
        # Precompute neighbor matrix if sizes are uniform
        # For q=3, n=8, size is 17. For general, might vary? No, for fixed q,n, size is 1 + n*(q-1).
        # So we can use a matrix.
        neigh_size = 1 + n * (q - 1)
        neigh_mat = np.zeros((N, neigh_size), dtype=int)
        for i in range(N):
            neigh_mat[i] = neighbors[i]
            
        while np.any(uncovered):
            if not np.any(uncovered): break
            # Vectorized gain calculation
            # uncovered is bool array. neigh_mat is int array.
            # We need sum of uncovered at neighbor positions.
            # This requires indexing.
            # gains = np.sum(uncovered[neigh_mat], axis=1)
            # This creates a huge temporary array (N * neigh_size).
            # For N=6561, size 17, it's ~111k elements. Very fast.
            gains = np.sum(uncovered[neigh_mat], axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0: break
            C.append(best_idx)
            uncovered[neigh_mat[best_idx]] = False
            if time.time() - start_time > time_limit: break
            
        # Reduction
        coverage_counts = np.zeros(N, dtype=int)
        for c in C:
            coverage_counts[neigh_mat[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(coverage_counts[neigh_mat[c]]) >= 2:
                    coverage_counts[neigh_mat[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(all_words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as a (N, 17) matrix
    neigh_size = 17
    neigh_mat = np.zeros((N, neigh_size), dtype=int)
    
    # Fast construction of neighbor matrix
    # Index of word (d0, ..., d7) is d0*3^7 + ... + d7
    # We can generate neighbors by modifying one digit.
    for i in range(N):
        w = all_words[i]
        neigh_set = [i]
        for pos in range(8):
            orig_val = w[pos]
            for val in range(3):
                if val == orig_val:
                    continue
                # Calculate new index
                # i + (val - orig_val) * (3 ** (7 - pos))
                diff = (val - orig_val) * (3 ** (7 - pos))
                neigh_set.append(i + diff)
        neigh_mat[i] = neigh_set
        
    start_time = time.time()
    time_limit = 118.0 # Leave 2s for output conversion
    
    # 1. Fast Greedy Initialization using Vectorized Gains
    uncovered = np.ones(N, dtype=bool)
    C = []
    
    while np.any(uncovered):
        # Calculate gain for each word: number of uncovered neighbors
        # uncovered[neigh_mat] creates an array of shape (N, 17)
        # Summing along axis 1 gives gain for each word
        gains = np.sum(uncovered[neigh_mat], axis=1)
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            break
            
        C.append(best_idx)
        uncovered[neigh_mat[best_idx]] = False
        
        if time.time() - start_time > time_limit * 0.2: # Spend only 20% time on greedy
            break

    # Ensure full coverage if greedy stopped early (shouldn't happen with 20% time for 6561 points)
    # If it did, we might have gaps. But greedy is fast.
    # Let's verify coverage.
    coverage_counts = np.zeros(N, dtype=int)
    for c in C:
        coverage_counts[neigh_mat[c]] += 1
        
    if np.any(coverage_counts == 0):
        # Fill gaps
        while np.any(coverage_counts == 0):
            # Find uncovered points
            uncovered_mask = (coverage_counts == 0)
            # Find best word to cover most uncovered
            gains = np.sum(uncovered_mask[neigh_mat], axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0: break
            C.append(best_idx)
            coverage_counts[neigh_mat[best_idx]] += 1
            if time.time() - start_time > time_limit * 0.3:
                 break

    # 2. Local Search
    # State: C (list of int), counts (array of int)
    
    def reduce_code(C, counts):
        """
        Removes redundant codewords.
        Returns (new_C, new_counts)
        """
        if not C:
            return C, counts
        current_code = list(C)
        current_counts = counts.copy()
        changed = True
        while changed:
            changed = False
            next_code = []
            for c in current_code:
                # If all neighbors are covered at least twice, c is redundant
                if np.min(current_counts[neigh_mat[c]]) >= 2:
                    current_counts[neigh_mat[c]] -= 1
                    changed = True
                else:
                    next_code.append(c)
            current_code = next_code
        return current_code, current_counts

    C, counts = reduce_code(C, counts)
    best_size = len(C)
    best_C = C.copy()
    
    # Precompute sets of neighbors for faster intersection?
    # neigh_mat rows are arrays.
    
    while time.time() - start_time < time_limit:
        # 1. Reduction Pass
        old_size = len(C)
        C, counts = reduce_code(C, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            
        if len(C) == old_size:
            # No reduction. Try swaps.
            # Shuffle C to avoid bias
            indices = list(range(len(C)))
            random.shuffle(indices)
            improved = False
            
            for idx in indices:
                if time.time() - start_time > time_limit:
                    break
                
                c = C[idx]
                # Get unique points covered by c
                c_neighs = neigh_mat[c]
                unique_mask = (counts[c_neighs] == 1)
                unique_points = c_neighs[unique_mask]
                
                if len(unique_points) == 0:
                    # Should have been reduced, but maybe race condition?
                    # Force remove
                    C.remove(c)
                    counts[neigh_mat[c]] -= 1
                    improved = True
                    break
                
                # If too many unique points, hard to find a single replacement
                if len(unique_points) > 4:
                    continue
                
                # Find candidates c' that cover all unique_points
                # Start with neighbors of first unique point
                # We want c' such that c' is in neighbors[u] for all u in unique_points
                # i.e., c' in intersection of neighbors[u]
                
                # Optimization: If len(unique_points) is small, intersect sets
                # Convert first neighbor list to set
                # Note: neigh_mat[unique_points[0]] is an array
                candidates_set = set(neigh_mat[unique_points[0]].tolist())
                
                for u in unique_points[1:]:
                    candidates_set.intersection_update(neigh_mat[u].tolist())
                    if not candidates_set:
                        break
                
                if not candidates_set:
                    continue
                
                # Remove existing codewords from candidates
                # C is a list. Converting to set each time is expensive if C is large.
                # But C is ~486.
                C_set = set(C)
                candidates_set.difference_update(C_set)
                
                if not candidates_set:
                    continue
                
                # Score candidates: We want to maximize redundancy (counts) after swap
                # Or just pick one that allows reduction.
                # Heuristic: Pick candidate that covers the most points where counts == 1 (other than unique_points)?
                # Actually, we just want to enable reduction.
                # Let's pick the candidate that maximizes sum(counts[neigh_mat[c_prime]])
                # This increases redundancy.
                
                best_c_prime = -1
                best_score = -1
                for c_prime in candidates_set:
                    # Score: how many neighbors of c_prime have count >= 2?
                    # Or just sum of counts?
                    # If we swap c -> c_prime, counts change:
                    # counts[neigh_mat[c]] -= 1
                    # counts[neigh_mat[c_prime]] += 1
                    # We want to create new redundancies.
                    # So we want c_prime to overlap with points that already have count >= 2?
                    # Or points with count == 1?
                    # If c_prime covers a point with count 1, it becomes 2 (redundant).
                    # So we want to maximize the number of points in neigh_mat[c_prime] with count == 1.
                    score = np.sum(counts[neigh_mat[c_prime]] == 1)
                    if score > best_score:
                        best_score = score
                        best_c_prime = c_prime
                
                if best_c_prime == -1:
                    continue
                    
                # Apply swap
                old_C = C.copy()
                old_counts = counts.copy()
                
                C.remove(c)
                C.append(best_c_prime)
                
                counts[neigh_mat[c]] -= 1
                counts[neigh_mat[best_c_prime]] += 1
                
                # Check validity
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
                    improved = True
                    break # Restart loop with new state
                else:
                    # No immediate reduction.
                    # Accept the swap to change state?
                    # This is a "neutral" move.
                    # To avoid cycling, we only accept if it changes the "fingerprint" or randomly.
                    # For now, accept it to explore.
                    C = new_C
                    counts = new_counts
                    improved = True
                    break # Restart loop
            
            if not improved:
                # Stuck. Try random kicks.
                # Randomly pick a codeword and swap with a random neighbor.
                for _ in range(20):
                    if time.time() - start_time > time_limit: break
                    if not C: break
                    
                    c_idx = random.randint(0, len(C)-1)
                    c = C[c_idx]
                    
                    # Pick random neighbor
                    c_prime = neigh_mat[c][random.randint(0, neigh_size-1)]
                    
                    if c_prime in C:
                        continue
                        
                    old_C = C.copy()
                    old_counts = counts.copy()
                    
                    C.remove(c)
                    C.append(c_prime)
                    counts[neigh_mat[c]] -= 1
                    counts[neigh_mat[c_prime]] += 1
                    
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
                        break
                    else:
                        # Accept kick
                        C = new_C
                        counts = new_counts
                        break # Restart main loop

    # Final Reduction
    best_C, _ = reduce_code(best_C, np.zeros(N, dtype=int)) # Wait, reduce_code needs counts
    # Recount for best_C
    final_counts = np.zeros(N, dtype=int)
    for c in best_C:
        final_counts[neigh_mat[c]] += 1
    best_C, _ = reduce_code(best_C, final_counts)
    
    return [list(all_words[i]) for i in best_C]
