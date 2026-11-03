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

    # Precompute neighbor masks and lists
    neighbor_masks = [0] * total_words
    neighbors_list = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        w = idx_to_tuple(i)
        mask = 1 << i
        neighbors_list[i].append(i)
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                mask |= (1 << j)
                neighbors_list[i].append(j)
        neighbor_masks[i] = mask

    full_mask = (1 << total_words) - 1

    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered_mask = full_mask
        code = []
        
        while uncovered_mask:
            max_gain = 0
            best_candidates = []
            
            for c in range(total_words):
                gain = (uncovered_mask & neighbor_masks[c]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [c]
                elif gain == max_gain and gain > 0:
                    best_candidates.append(c)
            
            if not best_candidates:
                break
                
            chosen = random.choice(best_candidates)
            code.append(chosen)
            uncovered_mask &= ~neighbor_masks[chosen]
            
        return code

    def reduce_code(code):
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            for nb in neighbors_list[c]:
                counts[nb] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                for nb in neighbors_list[c]:
                    counts[nb] -= 1
                if np.any(counts == 0):
                    for nb in neighbors_list[c]:
                        counts[nb] += 1
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
            for nb in neighbors_list[c]:
                counts[nb] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                # Remove c_remove
                for nb in neighbors_list[c_remove]:
                    counts[nb] -= 1
                
                # Check for holes among neighbors of c_remove
                holes = [h for h in neighbors_list[c_remove] if counts[h] == 0]
                
                if holes:
                    holes_mask = 0
                    for h in holes:
                        holes_mask |= (1 << h)
                    
                    # Find candidates covering all holes
                    candidates = []
                    for c in range(total_words):
                        if c == c_remove:
                            continue
                        if (holes_mask & ~neighbor_masks[c]) == 0:
                            candidates.append(c)
                            
                    if not candidates:
                        # Restore
                        for nb in neighbors_list[c_remove]:
                            counts[nb] += 1
                        continue
                        
                    # Score candidates
                    scored_cands = []
                    for c in candidates:
                        score = 0
                        for nb in neighbors_list[c]:
                            score += counts[nb]
                        scored_cands.append((score, c))
                    
                    scored_cands.sort(key=lambda x: x[0])
                    
                    top_k = min(5, len(scored_cands))
                    best_options = [x[1] for x in scored_cands[:top_k]]
                    best_cand = random.choice(best_options)
                    
                    # Apply swap
                    code[i] = best_cand
                    for nb in neighbors_list[best_cand]:
                        counts[nb] += 1
                    improved = True
                    break
                else:
                    # No holes, remove c_remove
                    code.pop(i)
                    improved = True
                    break
                    
        return code

    def perturb_repair(code, k, deadline):
        code = list(code)
        if len(code) <= k:
            return code
        
        indices = list(range(len(code)))
        random.shuffle(indices)
        remove_indices = indices[:k]
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
            
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            for nb in neighbors_list[c]:
                counts[nb] += 1
        
        while time.time() < deadline:
            uncovered_indices = np.where(counts == 0)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Greedy add
            um = 0
            for u in uncovered_indices:
                um |= (1 << u)
                
            max_gain = 0
            best_candidates = []
            for c in range(total_words):
                gain = (um & neighbor_masks[c]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [c]
                elif gain == max_gain and gain > 0:
                    best_candidates.append(c)
                    
            if not best_candidates:
                break
                
            chosen = random.choice(best_candidates)
            code.append(chosen)
            for nb in neighbors_list[chosen]:
                counts[nb] += 1
            
        return code

    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
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
