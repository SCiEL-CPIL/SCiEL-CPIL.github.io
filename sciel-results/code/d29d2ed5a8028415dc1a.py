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
    
    start_time = time.time()
    time_limit = 115.0
    
    # Initialize with a greedy code
    uncovered = np.ones(N, dtype=bool)
    C = []
    
    # Use a faster greedy: pick the word that covers the most uncovered points
    while np.any(uncovered):
        best_idx = -1
        best_count = -1
        for i in range(N):
            if not uncovered[i]:
                continue
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
    
    # Now apply local search to reduce the code
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts
    
    def reduce_code(code, counts):
        """
        Reduces code by removing redundant words.
        Returns new_code and new_counts.
        """
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        
        while True:
            removed_any = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors[c]]) >= 2:
                    # Remove c
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
    
    # Local Search Loop with improved strategy
    # Strategy:
    # 1. Try to remove a codeword if possible (reduction).
    # 2. If not, try to swap a codeword for a neighbor that enables a reduction.
    # 3. Use simulated annealing-like acceptance of neutral moves.
    
    temperature = 1.0
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit:
        # 1. Reduction Pass
        old_size = len(C)
        C, counts = reduce_code(C, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
        
        if time.time() - start_time > time_limit:
            break
        
        # 2. Swap Pass
        if len(C) == old_size:
            # No reduction happened. Try to find a move.
            improved = False
            # Randomize the order of C to avoid bias
            indices = list(range(len(C)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() - start_time > time_limit:
                    break
                c = C[idx]
                u_indices = neighbors[c]
                unique_mask = counts[u_indices] == 1
                unique_points = u_indices[unique_mask]
                
                if len(unique_points) == 0:
                    # Should have been removed, but maybe logic missed it?
                    # Force removal
                    C.remove(c)
                    counts[neighbors[c]] -= 1
                    improved = True
                    break
                
                # If too many unique points, unlikely to find a single replacement
                if len(unique_points) > 3:
                    continue
                
                # Find candidates c' that cover all unique_points
                # Intersection of neighbors[u] for u in unique_points
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
                
                # Filter candidates: must not be in C
                candidates_set.difference_update(C)
                
                if not candidates_set:
                    continue
                    
                # Pick the best candidate based on overlap with existing codewords
                best_candidate = None
                best_score = -1
                for cand in list(candidates_set)[:15]:  # Limit to 15 candidates for speed
                    # Score: how much overlap with existing codewords (higher is better for future reductions)
                    overlap = np.sum(counts[neighbors[cand]] > 0)
                    # Also consider if this candidate covers many points that are currently covered only once
                    new_unique_reduction = np.sum((counts[neighbors[cand]] == 1) & (counts[neighbors[cand]] < counts[neighbors[c]]))
                    score = overlap + 0.1 * new_unique_reduction
                    if score > best_score:
                        best_score = score
                        best_candidate = cand
                
                if best_candidate is None:
                    continue
                    
                c_prime = best_candidate
                
                # Apply swap
                old_C = C.copy()
                old_counts = counts.copy()
                
                C.remove(c)
                C.append(c_prime)
                
                counts[neighbors[c]] -= 1
                counts[neighbors[c_prime]] += 1
                
                # Check validity (should be valid by construction of U_c)
                # But let's be safe
                if np.any(counts < 1):
                    C = old_C
                    counts = old_counts
                    continue
                
                # Try to reduce
                new_C, new_counts = reduce_code(C, counts)
                
                if len(new_C) < len(old_C):
                    # Improvement!
                    C = new_C
                    counts = new_counts
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_counts = counts.copy()
                    improved = True
                    break # Restart loop
                    
                else:
                    # No reduction. Use simulated annealing to decide whether to accept
                    # Accept with probability exp(-delta/T) where delta is the change in "cost"
                    # For simplicity, accept neutral moves with some probability
                    if random.random() < 0.3:  # 30% chance to accept neutral moves
                        improved = True
                        break # Restart loop to see if this new state allows reduction
            
            if not improved:
                # No move found in this pass.
                # To escape local minima, we can try a "kick":
                # Randomly swap a word with a random neighbor, even if it doesn't enable reduction immediately.
                # Or just break and return best.
                # Let's try a few random kicks.
                for _ in range(30):
                    if time.time() - start_time > time_limit: break
                    if not C: break
                    c_idx = random.randint(0, len(C)-1)
                    c = C[c_idx]
                    # Pick a random neighbor of c
                    cand = neighbors[c]
                    c_prime = cand[random.randint(0, len(cand)-1)]
                    if c_prime in C: continue
                    
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
                        # Accept the kick to change state with some probability
                        if random.random() < 0.2:
                            C = new_C
                            counts = new_counts

    # Final Reduction
    best_C, _ = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
