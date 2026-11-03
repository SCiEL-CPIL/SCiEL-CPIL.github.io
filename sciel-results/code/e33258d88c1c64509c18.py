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
    # For radius 1, neighbors are self + 1-bit changes
    cover_count = 1 + n * (q - 1)
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
        iterations = 0
        while improved and time.time() < deadline:
            improved = False
            iterations += 1
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
                    if len(cand_list) > 50:
                        cand_list = random.sample(cand_list, 50)
                    
                    # Vectorized score calculation
                    cand_indices = np.array(cand_list, dtype=np.int32)
                    cand_neighbors = neighbors_arr[cand_indices, :]
                    scores = np.sum(counts[cand_neighbors], axis=1)
                    
                    # Softmax sampling
                    # Temperature T
                    T = 5.0
                    scores_normalized = (scores - np.max(scores)) / T
                    exp_scores = np.exp(scores_normalized)
                    probs = exp_scores / np.sum(exp_scores)
                    
                    chosen_idx = random.choices(range(len(cand_list)), weights=probs)[0]
                    best_cand = cand_list[chosen_idx]
                    
                    code[i] = best_cand
                    counts[neighbors_arr[best_cand, :]] += 1
                    improved = True
                    break
                else:
                    code.pop(i)
                    improved = True
                    break
        return code

    def perturb_repair(code, k, deadline):
        code = list(code)
        if len(code) <= k:
            return code
        
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        code_arr = np.array(code, dtype=np.int32)
        all_neighbors = neighbors_arr[code_arr, :]
        scores = np.sum(counts[all_neighbors], axis=1)
        
        top_k_factor = 2
        num_to_consider = min(len(code), k * top_k_factor)
        top_indices = np.argsort(scores)[-num_to_consider:]
        
        top_indices_list = top_indices.tolist()
        random.shuffle(top_indices_list)
        remove_indices = top_indices_list[:k]
        
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
