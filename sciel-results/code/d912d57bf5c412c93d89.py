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
    
    # Build the van Laarhoven 486-word code as starting point
    # This code is based on the structure: words whose digits sum to 0 mod 3 in certain patterns
    # Actually, let's use a known construction: the 486-word code can be generated
    # using the fact that 486 = 2 * 3^5 * ... Let me use a different approach.
    # 
    # The van Laarhoven code for K_3(8,1) with 486 words:
    # It's based on taking all words where certain parity/sum conditions hold.
    # A known construction: take all (a1,...,a8) where sum(ai) ≡ 0 (mod 3) 
    # gives 3^7 = 2187 words - too many.
    #
    # Actually, let me just use greedy but with vectorized computation for speed,
    # then do aggressive local search. The key improvement is making local search faster.
    
    start_time = time.time()
    time_limit = 112.0
    
    # Fast vectorized greedy initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    
    # Precompute neighbor matrix as list of arrays for fast indexing
    # Use a different strategy: maintain uncovered count per candidate
    # For speed, use numpy to compute marginal gains in batches
    
    while np.any(uncovered) and time.time() - start_time < time_limit - 30:
        # Compute marginal gain for all candidates using vectorized ops
        # This is O(N * avg_degree) per iteration but done in numpy
        gains = np.zeros(N, dtype=int)
        for i in range(0, N, 100):
            batch_end = min(i + 100, N)
            for j in range(i, batch_end):
                gains[j] = np.sum(uncovered[neighbors[j]])
        
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = False
    
    # If greedy didn't finish, fall back to just using what we have
    # and rely on local search
    
    # Helper functions
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts

    def reduce_code(code, counts):
        """Remove redundant codewords whose coverage is >= 2 everywhere."""
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
    
    # Local search: try removals and swaps
    # Key improvement: more aggressive swap strategy with better candidate selection
    no_improve_streak = 0
    max_streak = 50
    
    while time.time() - start_time < time_limit:
        # 1. Try to remove any redundant codeword
        old_size = len(C)
        C, counts = reduce_code(C, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            no_improve_streak = 0
        
        if time.time() - start_time > time_limit:
            break
        
        # 2. If no reduction, try swaps to enable future reductions
        if len(C) == old_size:
            improved = False
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
                    # This codeword is fully redundant - remove it
                    C.remove(c)
                    counts[neighbors[c]] -= 1
                    improved = True
                    break
                
                # Allow up to 4 unique points for more flexibility
                if len(unique_points) > 4:
                    continue
                
                # Find candidates that cover all unique points
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
                    
                # Try multiple candidates to find the best one
                candidates_list = list(candidates_set)
                random.shuffle(candidates_list)
                
                found_improvement = False
                for c_prime in candidates_list[:5]:  # Try up to 5 candidates
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
                            best_counts = counts.copy()
                        improved = True
                        found_improvement = True
                        break
                    else:
                        C = old_C
                        counts = old_counts
                
                if found_improvement:
                    break
                elif improved:
                    break
            
            if not improved:
                no_improve_streak += 1
                # Random kicks to escape local minima - more aggressive when stuck
                kick_count = min(10 + no_improve_streak // 5, 30)
                for _ in range(kick_count):
                    if time.time() - start_time > time_limit: break
                    if not C: break
                    c_idx = random.randint(0, len(C)-1)
                    c = C[c_idx]
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
                        
                    new_C, new_counts = reduce_code(C, counts)
                    if len(new_C) < len(old_C):
                        C = new_C
                        counts = new_counts
                        if len(C) < best_size:
                            best_size = len(C)
                            best_C = C.copy()
                            best_counts = counts.copy()
                        no_improve_streak = 0
                        break
                    else:
                        # Accept neutral moves with some probability to explore
                        if random.random() < 0.3:
                            C = new_C
                            counts = new_counts
                        else:
                            C = old_C
                            counts = old_counts
        else:
            no_improve_streak = 0

    # Final reduction
    best_C, _ = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
