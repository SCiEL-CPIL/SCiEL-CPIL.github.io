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
        neighbors.append(np.array(list(neigh_set), dtype=np.int32))
    
    # Precompute neighbor index matrix for vectorized greedy
    # nbr_matrix[i, j] = 1 if j is in neighbors[i]
    nbr_matrix = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        nbr_matrix[i, neighbors[i]] = 1
    
    start_time = time.time()
    time_limit = 115.0
    
    # 1. Fast Vectorized Greedy Initialization
    uncovered = np.ones(N, dtype=np.int8)
    C = []
    
    while np.any(uncovered):
        # counts[i] = number of uncovered points covered by word i
        counts_vec = nbr_matrix @ uncovered
        best_idx = int(np.argmax(counts_vec))
        if counts_vec[best_idx] == 0:
            break
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = 0
        if time.time() - start_time > time_limit:
            break
    
    # 2. Local Search with improved swap strategy
    def get_counts(code):
        counts = np.zeros(N, dtype=np.int32)
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
    
    C_set = set(C)
    
    max_unique_for_swap = 4
    
    def try_swap(c, C_list, C_set_local, counts_local):
        """Try to swap codeword c with a better candidate. Returns (new_C, new_C_set, new_counts, improved) or None."""
        u_indices = neighbors[c]
        unique_mask = counts_local[u_indices] == 1
        unique_points = u_indices[unique_mask]
        
        if len(unique_points) == 0:
            # Remove c
            new_C = [x for x in C_list if x != c]
            new_C_set = set(new_C)
            new_counts = counts_local.copy()
            new_counts[neighbors[c]] -= 1
            return new_C, new_C_set, new_counts, True
        
        if len(unique_points) > max_unique_for_swap:
            return None
        
        # Find candidates that cover all unique points
        if len(unique_points) == 1:
            candidates_set = set(neighbors[unique_points[0]].tolist())
        else:
            candidates_set = set(neighbors[unique_points[0]].tolist())
            for u in unique_points[1:]:
                candidates_set &= set(neighbors[u].tolist())
                if not candidates_set:
                    break
        
        if not candidates_set:
            return None
        
        candidates_set.difference_update(C_set_local)
        
        if not candidates_set:
            return None
        
        # Pick best candidate: maximize overlap with already-covered points
        best_candidate = None
        best_overlap = -1
        for c_prime in candidates_set:
            overlap = np.sum(counts_local[neighbors[c_prime]] >= 2)
            if overlap > best_overlap:
                best_overlap = overlap
                best_candidate = c_prime
        
        if best_candidate is None:
            return None
        
        # Apply swap
        new_C = [x for x in C_list if x != c]
        new_C.append(best_candidate)
        new_C_set = set(new_C)
        new_counts = counts_local.copy()
        new_counts[neighbors[c]] -= 1
        new_counts[neighbors[best_candidate]] += 1
        
        if np.any(new_counts < 1):
            return None
        
        # Try to reduce
        reduced_C, reduced_counts = reduce_code(new_C, new_counts)
        reduced_C_set = set(reduced_C)
        
        if len(reduced_C) < len(C_list):
            return reduced_C, reduced_C_set, reduced_counts, True
        else:
            # Accept swap to change state even without immediate reduction
            return new_C, new_C_set, new_counts, False
    
    def try_two_swap(c1, c2, C_list, C_set_local, counts_local):
        """Try to remove two codewords and add one. Returns (new_C, new_C_set, new_counts, improved) or None."""
        # Unique points of c1 and c2 combined
        u1 = neighbors[c1]
        u2 = neighbors[c2]
        
        # Points uniquely covered by c1 (after removing both)
        # We need to check: points covered only by c1 or only by c2
        # Combined unique set: points where counts==1 and covered by c1 or c2
        # But we need to be careful: a point might be covered by both c1 and c2
        
        # Simpler approach: compute what happens if we remove both
        temp_counts = counts_local.copy()
        temp_counts[neighbors[c1]] -= 1
        temp_counts[neighbors[c2]] -= 1
        
        # Find points that become uncovered
        uncovered_by_removal = temp_counts == 0
        if np.any(uncovered_by_removal):
            return None
        
        # Find points that become unique (count==1) after removal
        # These are the points that a replacement must cover
        # Actually, we need points where count was 2 and one of c1,c2 covers them
        # Let's find all points where count==1 after removal that are in neighbors[c1] or neighbors[c2]
        unique_after = (temp_counts == 1) & (nbr_matrix[:, c1] | nbr_matrix[:, c2])
        
        unique_points = np.where(unique_after)[0]
        
        if len(unique_points) > max_unique_for_swap + 2:
            return None
        
        # Find candidates that cover all these unique points
        if len(unique_points) == 0:
            # Both can be removed without replacement? That means they're redundant
            new_C = [x for x in C_list if x != c1 and x != c2]
            new_C_set = set(new_C)
            new_counts = temp_counts
            return new_C, new_C_set, new_counts, True
        
        if len(unique_points) == 1:
            candidates_set = set(neighbors[unique_points[0]].tolist())
        else:
            candidates_set = set(neighbors[unique_points[0]].tolist())
            for u in unique_points[1:]:
                candidates_set &= set(neighbors[u].tolist())
                if not candidates_set:
                    break
        
        if not candidates_set:
            return None
        
        candidates_set.difference_update(C_set_local)
        
        if not candidates_set:
            return None
        
        # Pick best candidate
        best_candidate = None
        best_overlap = -1
        for c_prime in candidates_set:
            overlap = np.sum(temp_counts[neighbors[c_prime]] >= 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_candidate = c_prime
        
        if best_candidate is None:
            return None
        
        # Apply: remove c1, c2, add best_candidate
        new_C = [x for x in C_list if x != c1 and x != c2]
        new_C.append(best_candidate)
        new_C_set = set(new_C)
        new_counts = temp_counts.copy()
        new_counts[neighbors[best_candidate]] += 1
        
        if np.any(new_counts < 1):
            return None
        
        # Try to reduce
        reduced_C, reduced_counts = reduce_code(new_C, new_counts)
        reduced_C_set = set(reduced_C)
        
        if len(reduced_C) < len(C_list) - 1:
            return reduced_C, reduced_C_set, reduced_counts, True
        else:
            return new_C, new_C_set, new_counts, False
    
    # Local Search Loop
    while time.time() - start_time < time_limit:
        # 1. Reduction Pass
        old_size = len(C)
        C, counts = reduce_code(C, counts)
        C_set = set(C)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            
        if time.time() - start_time > time_limit:
            break
            
        # 2. Try moves
        if len(C) == old_size:
            improved = False
            indices = list(range(len(C)))
            random.shuffle(indices)
            
            # Try single swaps
            for idx in indices:
                if time.time() - start_time > time_limit:
                    break
                c = C[idx]
                result = try_swap(c, C, C_set, counts)
                if result is not None:
                    new_C, new_C_set, new_counts, was_improved = result
                    if was_improved:
                        C = new_C
                        C_set = new_C_set
                        counts = new_counts
                        if len(C) < best_size:
                            best_size = len(C)
                            best_C = C.copy()
                            best_counts = counts.copy()
                        improved = True
                        break
                    else:
                        # Accept non-improving swap to change state
                        C = new_C
                        C_set = new_C_set
                        counts = new_counts
                        improved = True
                        break
            
            # Try two-swaps if no single swap worked
            if not improved:
                # Pick pairs of codewords to try removing
                max_pairs_to_try = 50
                pair_attempts = 0
                for i in range(len(C)):
                    if time.time() - start_time > time_limit or pair_attempts >= max_pairs_to_try:
                        break
                    for j in range(i + 1, len(C)):
                        if pair_attempts >= max_pairs_to_try:
                            break
                        c1 = C[i]
                        c2 = C[j]
                        result = try_two_swap(c1, c2, C, C_set, counts)
                        if result is not None:
                            new_C, new_C_set, new_counts, was_improved = result
                            if was_improved:
                                C = new_C
                                C_set = new_C_set
                                counts = new_counts
                                if len(C) < best_size:
                                    best_size = len(C)
                                    best_C = C.copy()
                                    best_counts = counts.copy()
                                improved = True
                                break
                            else:
                                C = new_C
                                C_set = new_C_set
                                counts = new_counts
                                improved = True
                                break
                        pair_attempts += 1
                    if improved:
                        break
            
            # Random kicks if nothing else worked
            if not improved:
                for _ in range(10):
                    if time.time() - start_time > time_limit:
                        break
                    if not C:
                        break
                    c_idx = random.randint(0, len(C) - 1)
                    c = C[c_idx]
                    cand = neighbors[c]
                    valid_cands = [x for x in cand if x not in C_set]
                    if not valid_cands:
                        continue
                    c_prime = random.choice(valid_cands)
                    
                    old_C = C.copy()
                    old_C_set = C_set.copy()
                    old_counts = counts.copy()
                    
                    C = [x for x in C if x != c]
                    C.append(c_prime)
                    C_set = set(C)
                    counts = old_counts.copy()
                    counts[neighbors[c]] -= 1
                    counts[neighbors[c_prime]] += 1
                    
                    if np.any(counts < 1):
                        C = old_C
                        C_set = old_C_set
                        counts = old_counts
                        continue
                        
                    new_C, new_counts = reduce_code(C, counts)
                    new_C_set = set(new_C)
                    if len(new_C) < len(old_C):
                        C = new_C
                        C_set = new_C_set
                        counts = new_counts
                        if len(C) < best_size:
                            best_size = len(C)
                            best_C = C.copy()
                            best_counts = counts.copy()
                        break
                    else:
                        C = new_C
                        C_set = new_C_set
                        counts = new_counts

    # Final Reduction
    best_C, _ = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
