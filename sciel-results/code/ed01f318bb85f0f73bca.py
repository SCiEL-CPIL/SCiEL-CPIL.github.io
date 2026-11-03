import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    if q != 3 or n != 6 or radius != 1:
        return _generic_greedy(q, n, radius, start_time, deadline)
    
    # Specific optimization for q=3, n=6, r=1
    total_words = 3 ** n
    
    def word_to_idx(w):
        idx = 0
        for digit in w:
            idx = idx * 3 + digit
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return tuple(w)
    
    # Precompute all neighbor indices for each word (13 neighbors: self + 12 at distance 1)
    neighbors = np.empty((total_words, 13), dtype=np.int32)
    for idx in range(total_words):
        w = idx_to_word(idx)
        nb_list = [idx]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = new_val
                nb_list.append(word_to_idx(tuple(new_w_list)))
        neighbors[idx] = nb_list
    
    # Precompute reverse neighbors: for each word, which words have it as a neighbor
    # This helps find candidate replacements quickly
    reverse_neighbors = [list() for _ in range(total_words)]
    for idx in range(total_words):
        for nb in neighbors[idx]:
            reverse_neighbors[nb].append(idx)
    # Convert to numpy arrays for fast access
    reverse_neighbors_arr = [np.array(rn, dtype=np.int32) for rn in reverse_neighbors]
    
    # Generate the seed code (tetracode direct sum) - 81 codewords
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_codewords = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Use NumPy for coverage tracking
    coverage = np.zeros(total_words, dtype=np.int32)
    current_code = list(seed_codewords)
    current_code_idx = [word_to_idx(cw) for cw in current_code]
    
    for cw_idx in current_code_idx:
        coverage[neighbors[cw_idx]] += 1
    
    # Helper: check if all neighbors of a codeword have coverage > 1
    def is_redundant(cw_idx):
        return np.all(coverage[neighbors[cw_idx]] > 1)
    
    # Helper: remove a codeword
    def remove_codeword(pos):
        nonlocal current_code, current_code_idx
        cw_idx = current_code_idx[pos]
        coverage[neighbors[cw_idx]] -= 1
        current_code.pop(pos)
        current_code_idx.pop(pos)
    
    # Helper: add a codeword
    def add_codeword(cw_idx):
        nonlocal current_code, current_code_idx
        coverage[neighbors[cw_idx]] += 1
        current_code.append(idx_to_word(cw_idx))
        current_code_idx.append(cw_idx)
    
    # 4. Redundancy Removal Loop - vectorized check
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        new_code_idx = []
        for i, cw_idx in enumerate(current_code_idx):
            if time.time() > deadline:
                # Keep remaining
                new_code_idx.extend(current_code_idx[i:])
                break
            if np.all(coverage[neighbors[cw_idx]] > 1):
                coverage[neighbors[cw_idx]] -= 1
                improved = True
            else:
                new_code_idx.append(cw_idx)
                
        current_code_idx = new_code_idx
        current_code = [idx_to_word(ci) for ci in current_code_idx]
    
    # 5. Local Search with faster swap moves
    code_set = set(current_code_idx)
    
    # Precompute for each word: how many codewords cover it (already in coverage)
    
    # Try multiple rounds of swap-based improvement
    swap_improved = True
    max_swap_rounds = 200
    for round_num in range(max_swap_rounds):
        if time.time() > deadline:
            break
        if not swap_improved:
            break
        swap_improved = False
        
        n_codewords = len(current_code_idx)
        if n_codewords == 0:
            break
        
        # Shuffle candidates
        candidates = list(range(n_codewords))
        random.shuffle(candidates)
        
        best_swap = None  # (remove_pos, add_idx, net_gain)
        best_gain = 0
        
        # Check all candidates (up to ~80, fast enough)
        for cand_pos in candidates:
            if time.time() > deadline:
                break
            
            remove_idx = current_code_idx[cand_pos]
            remove_nbs = neighbors[remove_idx]
            
            # Temporarily remove
            coverage[remove_nbs] -= 1
            uncovered_mask = (coverage == 0)
            n_uncovered = np.count_nonzero(uncovered_mask)
            
            if n_uncovered == 0:
                # Redundant - just remove
                current_code.pop(cand_pos)
                current_code_idx.pop(cand_pos)
                code_set.discard(remove_idx)
                swap_improved = True
                break
            else:
                # Find best replacement using reverse neighbors of uncovered words
                # A replacement word r covers uncovered word u if r is in reverse_neighbors[u]
                # We want r not in code_set, maximizing count of uncovered words it covers
                
                # Collect candidate replacements from uncovered words' reverse neighbors
                candidate_reps = set()
                uncovered_indices = np.where(uncovered_mask)[0]
                # Sample if too many uncovered
                if len(uncovered_indices) > 30:
                    sampled = random.sample(list(uncovered_indices), 30)
                else:
                    sampled = list(uncovered_indices)
                
                for uidx in sampled:
                    for r in reverse_neighbors_arr[uidx]:
                        if r not in code_set:
                            candidate_reps.add(r)
                
                # Score candidates
                if candidate_reps:
                    # Vectorize scoring: for each candidate, count how many uncovered it covers
                    rep_list = list(candidate_reps)
                    if len(rep_list) > 100:
                        rep_list = random.sample(rep_list, 100)
                    
                    best_rep_idx = -1
                    best_rep_covers = 0
                    
                    for rep_idx in rep_list:
                        rep_nbs = neighbors[rep_idx]
                        covers = np.count_nonzero(coverage[rep_nbs] == 0)
                        if covers > best_rep_covers:
                            best_rep_covers = covers
                            best_rep_idx = rep_idx
                    
                    # Gain: we remove one codeword (covers n_uncovered words uniquely) and add one
                    # Net: new codeword covers best_rep_covers of the uncovered
                    # We need best_rep_covers >= n_uncovered for a valid swap that doesn't break coverage
                    # Actually, we need the replacement to cover ALL uncovered words
                    if best_rep_idx >= 0 and best_rep_covers >= n_uncovered:
                        gain = best_rep_covers - n_uncovered  # >= 0
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (cand_pos, best_rep_idx)
                
                # Revert
                coverage[remove_nbs] += 1
        
        # Apply best swap
        if best_swap is not None and best_gain >= 0:
            cand_pos, add_idx = best_swap
            remove_idx = current_code_idx[cand_pos]
            
            # Remove old
            coverage[neighbors[remove_idx]] -= 1
            current_code.pop(cand_pos)
            current_code_idx.pop(cand_pos)
            code_set.discard(remove_idx)
            
            # Add new
            coverage[neighbors[add_idx]] += 1
            current_code.append(idx_to_word(add_idx))
            current_code_idx.append(add_idx)
            code_set.add(add_idx)
            
            swap_improved = True
    
    # 6. Final redundant removal pass
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        new_code_idx = []
        for i, cw_idx in enumerate(current_code_idx):
            if time.time() > deadline:
                new_code_idx.extend(current_code_idx[i:])
                break
            if np.all(coverage[neighbors[cw_idx]] > 1):
                coverage[neighbors[cw_idx]] -= 1
                improved = True
            else:
                new_code_idx.append(cw_idx)
                
        current_code_idx = new_code_idx
        current_code = [idx_to_word(ci) for ci in current_code_idx]
    
    # 7. Verify coverage (safety check)
    if time.time() < deadline - 3.0:
        final_coverage = np.zeros(total_words, dtype=np.int32)
        for cw_idx in current_code_idx:
            final_coverage[neighbors[cw_idx]] += 1
        if np.any(final_coverage == 0):
            # Coverage broken! Fallback to seed code
            current_code = list(seed_codewords)
            current_code_idx = [word_to_idx(cw) for cw in current_code]
    
    # Convert back to list of lists
    return [list(cw) for cw in current_code]

def _generic_greedy(q, n, radius, start_time, deadline):
    # Fallback implementation for other parameters
    # Basic greedy set cover
    total_words = q ** n
    
    def word_to_idx(w):
        idx = 0
        for digit in w:
            idx = idx * q + digit
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return tuple(w)
    
    # Precompute neighbors (words within Hamming distance radius)
    # For small n and q, enumerate all words
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_id = {w: i for i, w in enumerate(all_words)}
    
    # Build neighbor lists
    neighbors = []
    for w in all_words:
        nb_set = set()
        # All words within Hamming distance radius
        for u in all_words:
            dist = sum(1 for a, b in zip(w, u) if a != b)
            if dist <= radius:
                nb_set.add(word_to_id[u])
        neighbors.append(sorted(nb_set))
    
    # Greedy set cover
    uncovered = set(range(total_words))
    code = []
    
    while uncovered and time.time() < deadline:
        # Find word covering most uncovered
        best_word = -1
        best_count = 0
        for w_idx in range(total_words):
            if len(uncovered) == 0:
                break
            covered = len(uncovered & set(neighbors[w_idx]))
            if covered > best_count:
                best_count = covered
                best_word = w_idx
        
        if best_word < 0:
            break
        
        code.append(all_words[best_word])
        uncovered -= set(neighbors[best_word])
    
    return [list(w) for w in code]
