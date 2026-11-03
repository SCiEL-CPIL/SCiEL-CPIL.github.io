import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for 120s limit
    
    total_words = q ** n
    
    # Precompute neighbors
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

    neighbors = []
    cover_bitsets = []
    reverse_cover = [[] for _ in range(total_words)]
    
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
        
        neighbors.append(idxs)
        
        bs = 0
        for j in idxs:
            bs |= (1 << j)
            reverse_cover[j].append(i)
            
        cover_bitsets.append(bs)
    
    full_mask = (1 << total_words) - 1
    
    # Precompute neighbor arrays as numpy for faster counting
    neighbor_arrs = [np.array(nbr, dtype=np.int32) for nbr in neighbors]
    # Precompute reverse_cover as lists for fast intersection
    reverse_cover_sets = [set(rc) for rc in reverse_cover]
    
    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.uint8)
        code = []
        
        while uncovered.any():
            # Vectorized gain computation
            gains = np.array([np.sum(uncovered[neighbor_arrs[i]]) for i in range(total_words)], dtype=np.int32)
            
            best_gain = gains.max()
            if best_gain == 0:
                break
            
            best_candidates = np.where(gains == best_gain)[0]
            chosen = best_candidates[random.randint(0, len(best_candidates) - 1)]
            code.append(int(chosen))
            uncovered[neighbor_arrs[chosen]] = 0
        
        return code

    def reduce_code(code):
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            counts[neighbor_arrs[c]] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbor_arrs[c]] -= 1
                
                if np.any(counts[neighbor_arrs[c]] == 0):
                    counts[neighbor_arrs[c]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code

    def local_search(code, deadline):
        code = list(code)
        if not code:
            return code
            
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            counts[neighbor_arrs[c]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                nbrs_remove = neighbor_arrs[c_remove]
                
                # Try Removal
                counts[nbrs_remove] -= 1
                
                holes = nbrs_remove[counts[nbrs_remove] == 0]
                
                if len(holes) == 0:
                    code.pop(i)
                    improved = True
                    break
                else:
                    counts[nbrs_remove] += 1
                    
                    # Try Swap: find candidates that cover all holes
                    hole_list = holes.tolist()
                    # Start with smallest reverse_cover set
                    hole_list.sort(key=lambda x: len(reverse_cover_sets[x]))
                    candidates = set(reverse_cover_sets[hole_list[0]])
                    
                    for h in hole_list[1:]:
                        candidates &= reverse_cover_sets[h]
                        if not candidates:
                            break
                    
                    candidates.discard(c_remove)
                    
                    if candidates:
                        # Score: minimize waste = sum of counts on neighbors not in holes
                        hole_mask = np.zeros(total_words, dtype=np.uint8)
                        hole_mask[holes] = 1
                        
                        best_cand = None
                        best_score = float('inf')
                        
                        for cand in candidates:
                            nbrs_cand = neighbor_arrs[cand]
                            waste = np.sum(counts[nbrs_cand] * (1 - hole_mask[nbrs_cand]))
                            if waste < best_score:
                                best_score = waste
                                best_cand = cand
                        
                        if best_cand is not None:
                            counts[nbrs_remove] -= 1
                            counts[neighbor_arrs[best_cand]] += 1
                            code[i] = best_cand
                            improved = True
                            break
        return code

    def perturb_repair(code, k, deadline):
        code = list(code)
        if len(code) <= k:
            return code
        
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            counts[neighbor_arrs[c]] += 1
        
        critical_scores = []
        for i, c in enumerate(code):
            crit = np.sum(counts[neighbor_arrs[c]] == 1)
            critical_scores.append(crit)
        
        indices = list(range(len(code)))
        indices.sort(key=lambda x: critical_scores[x])
        
        remove_indices = indices[:k]
        
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
        
        covered = np.zeros(total_words, dtype=np.uint8)
        for c in code:
            covered[neighbor_arrs[c]] = 1
        uncovered = 1 - covered
        
        while uncovered.any() and time.time() < deadline:
            gains = np.array([np.sum(uncovered[neighbor_arrs[i]]) for i in range(total_words)], dtype=np.int32)
            
            best_gain = gains.max()
            if best_gain == 0:
                break
            
            best_candidates = np.where(gains == best_gain)[0]
            chosen = best_candidates[random.randint(0, len(best_candidates) - 1)]
            code.append(int(chosen))
            uncovered[neighbor_arrs[chosen]] = 0
            
        return code

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
    
    if best_code is None:
        best_code = greedy_construct(0)
        
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
