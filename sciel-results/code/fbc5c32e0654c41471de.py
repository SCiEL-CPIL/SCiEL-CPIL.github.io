import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    total_space = q ** n
    cover_count = 1 + n * (q - 1)
    
    # Precompute index conversions
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
    
    # Precompute neighbors array (compact representation)
    neighbors_arr = np.zeros((total_space, cover_count), dtype=np.int32)
    for i in range(total_space):
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
    
    # Build full boolean cover matrix for efficient operations
    # cover_matrix[i, j] = True if word j is within radius of word i
    cover_matrix = np.zeros((total_space, total_space), dtype=np.bool_)
    for i in range(total_space):
        cover_matrix[i, neighbors_arr[i, :]] = True
    
    # Precompute int8 version for dot products
    cover_matrix_int8 = cover_matrix.astype(np.int8)
    
    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_space, dtype=np.int8)
        code = []
        
        while np.any(uncovered):
            gains = cover_matrix_int8.dot(uncovered)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            # Speedup: instead of scanning all 2187 candidates, find an uncovered
            # word and only consider codewords within its ball (at most 1+7*2=15).
            # Among those, pick the one with the highest gain (tie-break randomly).
            first_uncovered = int(np.argmax(uncovered))
            local_cands = neighbors_arr[first_uncovered]
            local_gains = gains[local_cands]
            local_max = np.max(local_gains)
            ties = local_cands[local_gains == local_max]
            chosen = int(ties[random.randint(0, len(ties) - 1)])
            code.append(chosen)
            uncovered[cover_matrix[chosen]] = 0
        
        return code
    
    def reduce_code(code):
        if not code:
            return code
        
        counts = np.zeros(total_space, dtype=np.int32)
        for c in code:
            counts += cover_matrix_int8[c]
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts -= cover_matrix_int8[c]
                if np.any(counts == 0):
                    counts += cover_matrix_int8[c]
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code
    
    def reduce_code_multi_order(code, deadline):
        best = code
        for _ in range(4):
            if time.time() >= deadline:
                break
            shuffled = list(code)
            random.shuffle(shuffled)
            reduced = reduce_code(shuffled)
            if len(reduced) < len(best):
                best = reduced
        return best
    
    def local_search(code, deadline):
        code = list(code)
        if not code:
            return code
        
        counts = np.zeros(total_space, dtype=np.int32)
        for c in code:
            counts += cover_matrix_int8[c]
        
        improved = True
        no_improve_count = 0
        max_no_improve = 8
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                mask = cover_matrix_int8[c_remove]
                counts -= mask
                
                if np.any(counts == 0):
                    holes = np.where(counts == 0)[0]
                    # Find candidates that cover all holes
                    # A candidate c covers hole h if cover_matrix[c, h] is True
                    # i.e., h is in neighbors_arr[c], i.e., c is in neighbors_arr[h]
                    candidates = set(neighbors_arr[holes[0]].tolist())
                    for h in holes[1:]:
                        if not candidates:
                            break
                        candidates &= set(neighbors_arr[h].tolist())
                    
                    candidates.discard(c_remove)
                    
                    if not candidates:
                        counts += mask
                        continue
                    
                    cand_list = list(candidates)
                    if len(cand_list) > 80:
                        cand_list = random.sample(cand_list, 80)
                    
                    cand_indices = np.array(cand_list, dtype=np.int32)
                    # Score: how many words each candidate covers that are currently covered exactly once
                    # This helps find replacements that maintain coverage well
                    cand_masks = cover_matrix_int8[cand_indices, :]
                    # Count how many of the candidate's covered words have count==1 (would become count==2)
                    # vs count==0 (would become count==1, fixing holes)
                    # Actually, we want to maximize the number of words that go from 0 to 1 (fixing holes)
                    # and minimize disruption. Let's use: score = sum of counts over candidate's neighborhood
                    # Higher means the candidate covers words that are well-covered (good for robustness)
                    scores = cand_masks.dot(counts)
                    
                    best_idx = np.argmax(scores)
                    best_cand = cand_list[best_idx]
                    
                    code[i] = best_cand
                    counts += cover_matrix_int8[best_cand]
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    code.pop(i)
                    improved = True
                    no_improve_count = 0
                    break
            
            # Try 2-for-1 moves
            if not improved and len(code) >= 2 and time.time() < deadline:
                num_pairs = min(30, len(code) - 1)
                if num_pairs >= 2:
                    pairs = random.sample(range(len(code)), num_pairs)
                    for p in range(0, len(pairs) - 1, 2):
                        if time.time() >= deadline:
                            break
                        i, j = pairs[p], pairs[p + 1]
                        if i >= len(code) or j >= len(code) or i == j:
                            continue
                        ci, cj = code[i], code[j]
                        mi = cover_matrix_int8[ci]
                        mj = cover_matrix_int8[cj]
                        counts -= mi
                        counts -= mj
                        
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
                                counts += mi
                                counts += mj
                                continue
                            
                            cand_list = list(candidates)
                            if len(cand_list) > 100:
                                cand_list = random.sample(cand_list, 100)
                            cand_indices = np.array(cand_list, dtype=np.int32)
                            cand_masks = cover_matrix_int8[cand_indices, :]
                            scores = cand_masks.dot(counts)
                            
                            best_idx = np.argmax(scores)
                            best_cand = cand_list[best_idx]
                            
                            idx_remove = sorted([i, j], reverse=True)
                            for idx in idx_remove:
                                code.pop(idx)
                            code.append(best_cand)
                            counts += cover_matrix_int8[best_cand]
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
                        no_improve_count += 1
                        if no_improve_count >= max_no_improve:
                            break
        return code
    
    def perturb_repair(code, k, deadline):
        code = list(code)
        if len(code) <= k:
            return code
        
        remove_indices = random.sample(range(len(code)), k)
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
        
        counts = np.zeros(total_space, dtype=np.int32)
        for c in code:
            counts += cover_matrix_int8[c]
        
        uncovered = (counts == 0).astype(np.int8)
        
        while np.any(uncovered) and time.time() < deadline:
            gains = cover_matrix_int8.dot(uncovered)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            best_candidates = np.where(gains == max_gain)[0]
            chosen = int(best_candidates[random.randint(0, len(best_candidates) - 1)])
            code.append(chosen)
            counts += cover_matrix_int8[chosen]
            uncovered = (counts == 0).astype(np.int8)
        
        return code
    
    def final_polish(code, deadline):
        """High-precision polish: repeated reduction, targeted local search, and exact
        single-replacement local search that tries every candidate for every position."""
        code = list(code)
        if not code:
            return code
        
        # Multiple reduction passes with different orderings
        for _ in range(6):
            if time.time() >= deadline:
                break
            shuffled = list(code)
            random.shuffle(shuffled)
            reduced = reduce_code(shuffled)
            if len(reduced) < len(code):
                code = reduced
        
        # Targeted local search with tighter stopping
        if time.time() < deadline - 4:
            code = local_search(code, deadline - 3)
            code = reduce_code(code)
        
        # One more round of multi-order reduction
        if time.time() < deadline - 3:
            code = reduce_code_multi_order(code, deadline - 2)
        
        # Exact single-replacement local search: try every candidate for every
        # position. This is O(|code| * total_space) per pass but with |code|~186
        # and total_space=2187, one pass is ~400K ops which is fast.
        if time.time() < deadline - 1:
            counts = np.zeros(total_space, dtype=np.int32)
            for c in code:
                counts += cover_matrix_int8[c]
            
            improved = True
            while improved and time.time() < deadline - 0.5:
                improved = False
                for i in range(len(code)):
                    if time.time() >= deadline - 0.5:
                        break
                    c_remove = code[i]
                    counts -= cover_matrix_int8[c_remove]
                    
                    if np.any(counts == 0):
                        holes = np.where(counts == 0)[0]
                        # Find candidates that cover all holes
                        candidates = set(neighbors_arr[holes[0]].tolist())
                        for h in holes[1:]:
                            if not candidates:
                                break
                            candidates &= set(neighbors_arr[h].tolist())
                        candidates.discard(c_remove)
                        
                        if candidates:
                            # Pick the candidate that maximizes coverage of
                            # currently-well-covered words (robustness)
                            cand_list = list(candidates)
                            cand_indices = np.array(cand_list, dtype=np.int32)
                            cand_masks = cover_matrix_int8[cand_indices, :]
                            scores = cand_masks.dot(counts)
                            best_idx = int(np.argmax(scores))
                            best_cand = cand_list[best_idx]
                            code[i] = best_cand
                            counts += cover_matrix_int8[best_cand]
                            improved = True
                    else:
                        code.pop(i)
                        improved = True
                        break
        
        # Final reduction
        if time.time() < deadline:
            code = reduce_code(code)
        
        return code
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    consecutive_no_improve = 0
    
    while time.time() < deadline - 12:
        code = greedy_construct(seed)
        code = reduce_code(code)
        
        if time.time() < deadline - 8:
            code = local_search(code, deadline - 5)
            code = reduce_code(code)
        
        if time.time() < deadline - 10:
            code = reduce_code_multi_order(code, deadline - 7)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
        
        # Perturbation-based diversification
        if time.time() < deadline - 15 and best_code is not None and consecutive_no_improve >= 2:
            for k in [1, 2, 3, 4]:
                if time.time() >= deadline - 12:
                    break
                perturbed = perturb_repair(best_code, k, deadline - 10)
                perturbed = reduce_code(perturbed)
                
                if time.time() < deadline - 8:
                    perturbed = local_search(perturbed, deadline - 5)
                    perturbed = reduce_code(perturbed)
                
                if time.time() < deadline - 10:
                    perturbed = reduce_code_multi_order(perturbed, deadline - 7)
                
                if len(perturbed) < best_size:
                    best_size = len(perturbed)
                    best_code = perturbed
                    consecutive_no_improve = 0
                    break
                else:
                    consecutive_no_improve += 1
        
        seed += 1
    
    # Final polish phase
    if best_code is not None and time.time() < deadline - 3:
        polished = final_polish(best_code, deadline - 1)
        if len(polished) < len(best_code):
            best_code = polished
        
        # Try one more local search pass
        if time.time() < deadline - 2:
            polished2 = local_search(best_code, deadline - 1)
            polished2 = reduce_code(polished2)
            if len(polished2) < len(best_code):
                best_code = polished2
        
        # Final reduction
        if time.time() < deadline - 1:
            best_code = reduce_code(best_code)
    
    result = []
    for idx in best_code:
        digits = idx_to_base_q(idx)
        result.append(digits)
    
    return result
