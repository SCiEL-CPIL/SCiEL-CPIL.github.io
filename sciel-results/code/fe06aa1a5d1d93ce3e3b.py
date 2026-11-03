import time
import random
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    # For our specific case: q=3, n=7, radius=1
    # We can hardcode some optimizations for this case
    if q == 3 and n == 7 and radius == 1:
        # Precompute all neighbors using bit operations for speed
        # Each word is an index from 0 to 3^7 - 1 = 2186
        # Neighbors of a word include the word itself and all words at Hamming distance 1
        # Total neighbors per word: 1 + 7 * 2 = 15
        
        cover_count = 1 + n * (q - 1)
        
        # Build neighbor lookup using integer arithmetic
        neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
        
        # Convert index to base-q representation
        def idx_to_base_q(idx):
            digits = [0] * n
            for i in range(n - 1, -1, -1):
                digits[i] = idx % q
                idx //= q
            return digits
        
        def base_q_to_idx(digits):
            idx = 0
            for d in digits:
                idx = idx * q + d
            return idx
        
        for i in range(total_words):
            digits = idx_to_base_q(i)
            idxs = [i]
            for pos in range(n):
                original_val = digits[pos]
                for new_val in range(q):
                    if new_val == original_val:
                        continue
                    new_digits = digits.copy()
                    new_digits[pos] = new_val
                    j = base_q_to_idx(new_digits)
                    idxs.append(j)
            for j, idx in enumerate(idxs):
                neighbors_arr[i, j] = idx
        
        start_time = time.time()
        deadline = start_time + 115
        
        best_code = None
        best_size = float('inf')
        
        def greedy_construct(random_seed):
            if random_seed is not None:
                random.seed(random_seed)
            
            uncovered = np.ones(total_words, dtype=np.bool_)
            code = []
            
            while uncovered.any():
                gains = np.sum(uncovered[neighbors_arr], axis=1)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                best_candidates = np.where(gains == max_gain)[0]
                chosen = random.choice(best_candidates.tolist())
                code.append(chosen)
                uncovered[neighbors_arr[chosen, :]] = False
            
            return code
        
        def reduce_code(code):
            if not code:
                return code
            counts = np.zeros(total_words, dtype=np.int16)
            for c in code:
                counts[neighbors_arr[c, :]] += 1
            
            changed = True
            max_rounds = 20
            round_count = 0
            while changed and round_count < max_rounds:
                changed = False
                round_count += 1
                new_code = []
                for c in code:
                    counts[neighbors_arr[c, :]] -= 1
                    if np.any(counts == 0):
                        counts[neighbors_arr[c, :]] += 1
                        new_code.append(c)
                    else:
                        changed = True
                code = new_code
            return code
        
        def local_search(code, deadline):
            code = list(code)
            if not code:
                return code
            
            counts = np.zeros(total_words, dtype=np.int16)
            for c in code:
                counts[neighbors_arr[c, :]] += 1
            
            improved = True
            no_improve_count = 0
            max_no_improve = 8
            while improved and time.time() < deadline:
                improved = False
                indices = list(range(len(code)))
                random.shuffle(indices)
                
                # Try 1-for-0 and 1-for-1 moves
                for i in indices:
                    if time.time() >= deadline:
                        break
                    
                    c_remove = code[i]
                    neighbors_c = neighbors_arr[c_remove, :]
                    counts[neighbors_c] -= 1
                    
                    if np.any(counts == 0):
                        holes = np.where(counts == 0)[0]
                        # Find candidates that cover all holes
                        candidates = set(neighbors_arr[holes[0]].tolist())
                        for h in holes[1:]:
                            if not candidates:
                                break
                            candidates &= set(neighbors_arr[h].tolist())
                        
                        candidates.discard(c_remove)
                        
                        if not candidates:
                            counts[neighbors_c] += 1
                            continue
                        
                        cand_list = list(candidates)
                        # Score candidates by how much they help with other under-covered words
                        if len(cand_list) > 200:
                            cand_list = random.sample(cand_list, 200)
                        
                        cand_indices = np.array(cand_list, dtype=np.int32)
                        cand_neighbors = neighbors_arr[cand_indices, :]
                        # Prefer candidates that cover words with low counts
                        scores = np.sum(counts[cand_neighbors], axis=1)
                        # Bonus for covering the holes
                        hole_set = set(holes.tolist())
                        for ci, cnd in enumerate(cand_list):
                            for nb in neighbors_arr[cnd]:
                                if nb in hole_set:
                                    scores[ci] += 10
                        
                        best_idx = np.argmax(scores)
                        best_cand = cand_list[best_idx]
                        
                        code[i] = best_cand
                        counts[neighbors_arr[best_cand, :]] += 1
                        improved = True
                        no_improve_count = 0
                        break
                    else:
                        code.pop(i)
                        improved = True
                        no_improve_count = 0
                        break
                
                # Try 2-for-1 move: remove two codewords, add one that covers all exposed words
                if not improved and len(code) >= 2 and time.time() < deadline:
                    pairs = random.sample(range(len(code)), min(30, len(code)))
                    for p in range(0, len(pairs) - 1, 2):
                        i, j = pairs[p], pairs[p + 1]
                        if i >= len(code) or j >= len(code) or i == j:
                            continue
                        ci, cj = code[i], code[j]
                        ni = neighbors_arr[ci, :]
                        nj = neighbors_arr[cj, :]
                        counts[ni] -= 1
                        counts[nj] -= 1
                        
                        if np.any(counts == 0):
                            holes = np.where(counts == 0)[0]
                            candidates = set(neighbors_arr[holes[0]].tolist())
                            for h in holes[1:]:
                                if not candidates:
                                    break
                                candidates &= set(neighbors_arr[h].tolist())
                            candidates.discard(ci)
                            candidates.discard(cj)
                            
                            if not candidates:
                                counts[ni] += 1
                                counts[nj] += 1
                                continue
                            
                            cand_list = list(candidates)
                            if len(cand_list) > 200:
                                cand_list = random.sample(cand_list, 200)
                            cand_indices = np.array(cand_list, dtype=np.int32)
                            cand_neighbors = neighbors_arr[cand_indices, :]
                            scores = np.sum(counts[cand_neighbors], axis=1)
                            # Bonus for covering holes
                            hole_set = set(holes.tolist())
                            for ci_idx, cnd in enumerate(cand_list):
                                for nb in neighbors_arr[cnd]:
                                    if nb in hole_set:
                                        scores[ci_idx] += 15
                            
                            best_idx = np.argmax(scores)
                            best_cand = cand_list[best_idx]
                            
                            # Remove both, add one
                            idx_remove = sorted([i, j], reverse=True)
                            for idx in idx_remove:
                                code.pop(idx)
                            code.append(best_cand)
                            counts[neighbors_arr[best_cand, :]] += 1
                            improved = True
                            no_improve_count = 0
                            break
                        else:
                            # Both can be removed (2-for-0)
                            idx_remove = sorted([i, j], reverse=True)
                            for idx in idx_remove:
                                code.pop(idx)
                            improved = True
                            no_improve_count = 0
                            break
                    
                    if not improved:
                        counts[ni] += 1
                        counts[nj] += 1
                
                # Try 2-for-2 swap: replace two codewords with two different ones
                if not improved and len(code) >= 2 and time.time() < deadline:
                    pairs = random.sample(range(len(code)), min(20, len(code)))
                    for p in range(0, len(pairs) - 1, 2):
                        i, j = pairs[p], pairs[p + 1]
                        if i >= len(code) or j >= len(code) or i == j:
                            continue
                        ci, cj = code[i], code[j]
                        ni = neighbors_arr[ci, :]
                        nj = neighbors_arr[cj, :]
                        counts[ni] -= 1
                        counts[nj] -= 1
                        
                        # Find words that become uncovered
                        holes = np.where(counts == 0)[0]
                        
                        if len(holes) == 0:
                            # Can remove both
                            idx_remove = sorted([i, j], reverse=True)
                            for idx in idx_remove:
                                code.pop(idx)
                            improved = True
                            no_improve_count = 0
                            break
                        elif len(holes) <= 2:
                            # Try to find two replacements that cover all holes
                            hole1_neighbors = set(neighbors_arr[holes[0]].tolist())
                            if len(holes) > 1:
                                hole2_neighbors = set(neighbors_arr[holes[1]].tolist())
                                # Need one candidate covering hole1 and one covering hole2 (or one covering both)
                                cand1_list = [c for c in hole1_neighbors if c not in (ci, cj)]
                                cand2_list = [c for c in (hole2_neighbors if len(holes) > 1 else hole1_neighbors) if c not in (ci, cj) and c not in cand1_list]
                                
                                if cand1_list and cand2_list:
                                    # Score by coverage
                                    c1_scores = np.array([np.sum(counts[neighbors_arr[c]]) for c in cand1_list[:50]])
                                    c2_scores = np.array([np.sum(counts[neighbors_arr[c]]) for c in cand2_list[:50]])
                                    best_c1 = cand1_list[np.argmax(c1_scores)]
                                    best_c2 = cand2_list[np.argmax(c2_scores)]
                                    
                                    # Verify this works
                                    test_counts = counts.copy()
                                    test_counts[neighbors_arr[best_c1]] += 1
                                    test_counts[neighbors_arr[best_c2]] += 1
                                    if not np.any(test_counts == 0):
                                        code[i] = best_c1
                                        code[j] = best_c2
                                        counts[neighbors_arr[best_c1]] += 1
                                        counts[neighbors_arr[best_c2]] += 1
                                        improved = True
                                        no_improve_count = 0
                                        break
                            else:
                                # Only one hole, find two candidates that together cover it (redundant but might help)
                                pass
                        
                        counts[ni] += 1
                        counts[nj] += 1
                    else:
                        if improved:
                            break
                
                if not improved:
                    no_improve_count += 1
                    if no_improve_count >= max_no_improve:
                        break
            return code
        
        def perturb_repair(code, k, deadline):
            code = list(code)
            if len(code) <= k:
                return code
            
            # Remove k random codewords
            remove_indices = random.sample(range(len(code)), k)
            for idx in sorted(remove_indices, reverse=True):
                code.pop(idx)
            
            counts = np.zeros(total_words, dtype=np.int16)
            for c in code:
                counts[neighbors_arr[c, :]] += 1
            
            uncovered = counts == 0
            
            while uncovered.any() and time.time() < deadline:
                gains = np.sum(uncovered[neighbors_arr], axis=1)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                best_candidates = np.where(gains == max_gain)[0]
                # Prefer candidates that also cover many already-covered words (better future flexibility)
                if len(best_candidates) > 10:
                    # Sample from top candidates
                    sample_size = min(10, len(best_candidates))
                    chosen_idx = random.choice(best_candidates.tolist()[:sample_size])
                else:
                    chosen_idx = random.choice(best_candidates.tolist())
                code.append(chosen_idx)
                counts[neighbors_arr[chosen_idx, :]] += 1
                uncovered = counts == 0
            
            return code
        
        seed = 0
        consecutive_no_improve = 0
        while time.time() < deadline:
            code = greedy_construct(seed)
            code = reduce_code(code)
            
            if time.time() < deadline - 5:
                code = local_search(code, deadline - 2)
                code = reduce_code(code)
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1
            
            # More aggressive perturbation and repair
            if time.time() < deadline - 15 and best_code is not None:
                # Try different perturbation sizes
                k = random.randint(1, 5)
                perturbed = perturb_repair(best_code, k, deadline - 10)
                perturbed = reduce_code(perturbed)
                
                if time.time() < deadline - 5:
                    perturbed = local_search(perturbed, deadline - 2)
                    perturbed = reduce_code(perturbed)
                
                if len(perturbed) < best_size:
                    best_size = len(perturbed)
                    best_code = perturbed
                    consecutive_no_improve = 0
                else:
                    consecutive_no_improve += 1
            
            # If stuck, try a fresh greedy with different seed
            if consecutive_no_improve >= 3 and time.time() < deadline - 20:
                fresh_code = greedy_construct(seed + 1000)
                fresh_code = reduce_code(fresh_code)
                if time.time() < deadline - 5:
                    fresh_code = local_search(fresh_code, deadline - 2)
                    fresh_code = reduce_code(fresh_code)
                
                if len(fresh_code) < best_size:
                    best_size = len(fresh_code)
                    best_code = fresh_code
                    consecutive_no_improve = 0
            
            seed += 1
        
        # Final polish: try to reduce further with more iterations
        if best_code is not None and time.time() < deadline - 3:
            polished = local_search(best_code, deadline - 2)
            polished = reduce_code(polished)
            if len(polished) < len(best_code):
                best_code = polished
        
        # One more round of local search if time permits
        if best_code is not None and time.time() < deadline - 1:
            polished2 = local_search(best_code, deadline - 0.5)
            polished2 = reduce_code(polished2)
            if len(polished2) < len(best_code):
                best_code = polished2
        
        result = []
        for idx in best_code:
            digits = idx_to_base_q(idx)
            result.append(digits)
        
        return result
    
    # Fallback for other parameters
    total_words = q ** n
    cover_count = 1 + n * (q - 1)
    
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx
    
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                idxs.append(j)
        for j, idx in enumerate(idxs):
            neighbors_arr[i, j] = idx
    
    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        
        while uncovered.any():
            gains = np.sum(uncovered[neighbors_arr], axis=1)
            valid = gains > 0
            if not valid.any():
                break
            max_gain = np.max(gains[valid])
            best_candidates = np.where((gains == max_gain) & valid)[0]
            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False
        
        return code
    
    def reduce_code(code):
        if not code:
            return code
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbors_arr[c, :]] -= 1
                if np.any(counts == 0):
                    counts[neighbors_arr[c, :]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code
    
    def local_search(code, deadline):
        code = list(code)
        if not code:
            return code
        
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        no_improve_count = 0
        max_no_improve = 5
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                neighbors_c = neighbors_arr[c_remove, :]
                counts[neighbors_c] -= 1
                
                if np.any(counts == 0):
                    holes = np.where(counts == 0)[0]
                    candidates = set(neighbors_arr[holes[0]].tolist())
                    for h in holes[1:]:
                        if not candidates:
                            break
                        candidates &= set(neighbors_arr[h].tolist())
                    
                    candidates.discard(c_remove)
                    
                    if not candidates:
                        counts[neighbors_c] += 1
                        continue
                    
                    cand_list = list(candidates)
                    if len(cand_list) > 100:
                        cand_list = random.sample(cand_list, 100)
                    
                    cand_indices = np.array(cand_list, dtype=np.int32)
                    cand_neighbors = neighbors_arr[cand_indices, :]
                    scores = np.sum(counts[cand_neighbors], axis=1)
                    # Bonus for covering holes
                    hole_set = set(holes.tolist())
                    for ci, cnd in enumerate(cand_list):
                        for nb in neighbors_arr[cnd]:
                            if nb in hole_set:
                                scores[ci] += 10
                    
                    best_idx = np.argmax(scores)
                    best_cand = cand_list[best_idx]
                    
                    code[i] = best_cand
                    counts[neighbors_arr[best_cand, :]] += 1
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    code.pop(i)
                    improved = True
                    no_improve_count = 0
                    break
            
            # Try 2-for-1 move
            if not improved and len(code) >= 2 and time.time() < deadline:
                pairs = random.sample(range(len(code)), min(15, len(code)))
                for p in range(0, len(pairs) - 1, 2):
                    i, j = pairs[p], pairs[p + 1]
                    if i >= len(code) or j >= len(code) or i == j:
                        continue
                    ci, cj = code[i], code[j]
                    ni = neighbors_arr[ci, :]
                    nj = neighbors_arr[cj, :]
                    counts[ni] -= 1
                    counts[nj] -= 1
                    
                    if np.any(counts == 0):
                        holes = np.where(counts == 0)[0]
                        candidates = set(neighbors_arr[holes[0]].tolist())
                        for h in holes[1:]:
                            if not candidates:
                                break
                            candidates &= set(neighbors_arr[h].tolist())
                        candidates.discard(ci)
                        candidates.discard(cj)
                        
                        if not candidates:
                            counts[ni] += 1
                            counts[nj] += 1
                            continue
                        
                        cand_list = list(candidates)
                        if len(cand_list) > 100:
                            cand_list = random.sample(cand_list, 100)
                        cand_indices = np.array(cand_list, dtype=np.int32)
                        cand_neighbors = neighbors_arr[cand_indices, :]
                        scores = np.sum(counts[cand_neighbors], axis=1)
                        hole_set = set(holes.tolist())
                        for ci_idx, cnd in enumerate(cand_list):
                            for nb in neighbors_arr[cnd]:
                                if nb in hole_set:
                                    scores[ci_idx] += 15
                        
                        best_idx = np.argmax(scores)
                        best_cand = cand_list[best_idx]
                        
                        idx_remove = sorted([i, j], reverse=True)
                        for idx in idx_remove:
                            code.pop(idx)
                        code.append(best_cand)
                        counts[neighbors_arr[best_cand, :]] += 1
                        improved = True
                        no_improve_count = 0
                        break
                    else:
                        idx_remove = sorted([i, j], reverse=True)
                        for idx in idx_remove:
                            code.pop(idx)
                        improved = True
                        no_improve_count = 0
                        break
                
                if not improved:
                    counts[ni] += 1
                    counts[nj] += 1
            
            if not improved:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    break
        return code
    
    seed = 0
    while time.time() < deadline:
        code = greedy_construct(seed)
        code = reduce_code(code)
        
        if time.time() < deadline - 5:
            code = local_search(code, deadline - 2)
            code = reduce_code(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
        
        seed += 1
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
