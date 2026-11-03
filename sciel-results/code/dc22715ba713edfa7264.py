import time
import random
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    
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

    # Precompute neighbors
    cover_count = 1 + n * (q - 1)
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    
    # Vectorized generation of neighbors
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

    # Precompute reverse mapping: for each word, which codewords cover it
    # Since the covering graph is symmetric (if c covers w, then w is a neighbor of c,
    # and c is a neighbor of w), reverse_cover[word] is just neighbors_arr[word]
    # But let's be explicit for clarity and potential future use
    # Actually, for radius 1, the set of codewords that cover word w is exactly
    # the set of words at distance <= 1 from w, which is neighbors_arr[w].
    # So reverse_cover_arr = neighbors_arr in this case.
    # However, the list of "which codewords cover word w" is the same as
    # "which words are within distance 1 of w".
    # So we can just use neighbors_arr for both directions.

    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        
        while uncovered.any():
            # Vectorized gain computation
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
                    # Identify holes
                    holes = np.where(counts == 0)[0]
                    # Find candidates covering all holes
                    # Use the fact that candidates must be in the intersection of
                    # neighbors of all holes
                    candidates = set(neighbors_arr[holes[0]].tolist())
                    for h in holes[1:]:
                        if not candidates:
                            break
                        candidates &= set(neighbors_arr[h].tolist())
                    
                    candidates.discard(c_remove)
                    
                    if not candidates:
                        counts[neighbors_c] += 1
                        continue
                    
                    # Score candidates: Maximize redundancy overlap
                    best_cand = None
                    best_score = -1
                    cand_list = list(candidates)
                    
                    if len(cand_list) > 50:
                        cand_list = random.sample(cand_list, 50)
                        
                    for cand in cand_list:
                        cand_neigh = neighbors_arr[cand, :]
                        # Score: how many of the currently uncovered (holes) are covered
                        # Actually, we want to maximize the number of words with count >= 1
                        # that are covered by this candidate, to maximize redundancy
                        # But a simpler heuristic: minimize the number of words that become
                        # "critical" (count == 1) after adding this candidate
                        # Or maximize the sum of counts over the candidate's neighbors
                        score = np.sum(counts[cand_neigh])
                        if score > best_score:
                            best_score = score
                            best_cand = cand
                            
                    if best_cand is not None:
                        code[i] = best_cand
                        counts[neighbors_arr[best_cand, :]] += 1
                        improved = True
                        break
                    else:
                        counts[neighbors_c] += 1
                else:
                    # Can remove
                    code.pop(i)
                    improved = True
                    break
                    
            # 2-1 local search move: try to replace 2 codewords with 1
            if time.time() < deadline:
                if len(code) >= 2:
                    # Sample pairs
                    pair_count = min(100, (len(code) * (len(code) - 1)) // 2)
                    tried_pairs = set()
                    found_replacement = False
                    
                    for _ in range(pair_count):
                        if time.time() >= deadline:
                            break
                        i1, i2 = random.sample(range(len(code)), 2)
                        if (i1, i2) in tried_pairs or (i2, i1) in tried_pairs:
                            continue
                        tried_pairs.add((i1, i2))
                        
                        c1 = code[i1]
                        c2 = code[i2]
                        
                        # Temporarily remove both
                        counts[neighbors_arr[c1, :]] -= 1
                        counts[neighbors_arr[c2, :]] -= 1
                        
                        holes = np.where(counts == 0)[0]
                        
                        if len(holes) == 0:
                            # Can remove both, add nothing? That would be a 2-0 move
                            # But we want to maintain coverage, so this means both are redundant
                            # Restore and try single removal instead
                            counts[neighbors_arr[c1, :]] += 1
                            counts[neighbors_arr[c2, :]] += 1
                            continue
                        
                        # Find candidates covering all holes
                        candidates = set(neighbors_arr[holes[0]].tolist())
                        for h in holes[1:]:
                            if not candidates:
                                break
                            candidates &= set(neighbors_arr[h].tolist())
                        
                        candidates.discard(c1)
                        candidates.discard(c2)
                        
                        if not candidates:
                            counts[neighbors_arr[c1, :]] += 1
                            counts[neighbors_arr[c2, :]] += 1
                            continue
                        
                        # Choose the best candidate
                        best_cand = None
                        best_score = -1
                        cand_list = list(candidates)
                        if len(cand_list) > 30:
                            cand_list = random.sample(cand_list, 30)
                        
                        for cand in cand_list:
                            cand_neigh = neighbors_arr[cand, :]
                            score = np.sum(counts[cand_neigh])
                            if score > best_score:
                                best_score = score
                                best_cand = cand
                        
                        if best_cand is not None:
                            # Replace c1 and c2 with best_cand
                            code.pop(i2)
                            code[i1] = best_cand
                            counts[neighbors_arr[c1, :]] += 1
                            counts[neighbors_arr[c2, :]] += 1
                            counts[neighbors_arr[best_cand, :]] += 1
                            improved = True
                            found_replacement = True
                            break
                        else:
                            counts[neighbors_arr[c1, :]] += 1
                            counts[neighbors_arr[c2, :]] += 1
                    
        return code

    def perturb_repair(code, k, deadline):
        code = list(code)
        if len(code) <= k:
            return code
        
        remove_indices = random.sample(range(len(code)), k)
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
        
        # Repair
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
            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            counts[neighbors_arr[chosen, :]] += 1
            uncovered = counts == 0
            
        return code

    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        # 1. Greedy
        code = greedy_construct(seed)
        code = reduce_code(code)
        
        # 2. Local Search
        if time.time() < deadline - 5:
            code = local_search(code, deadline - 2)
            code = reduce_code(code)
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        # 3. Perturbation Strategy
        if time.time() < deadline - 15 and best_code is not None:
            k = random.randint(2, 5)
            perturbed = perturb_repair(best_code, k, deadline - 10)
            perturbed = reduce_code(perturbed)
            
            if time.time() < deadline - 5:
                perturbed = local_search(perturbed, deadline - 2)
                perturbed = reduce_code(perturbed)
                
            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_code = perturbed
        
        seed += 1
        
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
