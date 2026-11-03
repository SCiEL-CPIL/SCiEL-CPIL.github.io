import time
import random
import numpy as np
from itertools import product

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
    neighbor_indices = []
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
        neighbor_indices.append(idxs)
    
    cover_count = 1 + n * (q - 1)
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    for i in range(total_words):
        for j, idx in enumerate(neighbor_indices[i]):
            neighbors_arr[i, j] = idx

    # Precompute reverse mapping: for each word, which codewords cover it
    # reverse_cover[word] = list of codeword indices that cover this word
    reverse_cover = [[] for _ in range(total_words)]
    for i in range(total_words):
        for j in range(cover_count):
            w = neighbors_arr[i, j]
            reverse_cover[w].append(i)
    
    # Convert to numpy arrays for faster access
    reverse_cover_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    reverse_cover_counts = np.zeros(total_words, dtype=np.int32)
    for w in range(total_words):
        rc = reverse_cover[w]
        for j, ci in enumerate(rc):
            reverse_cover_arr[w, j] = ci
        reverse_cover_counts[w] = len(rc)

    def greedy_construct_vectorized(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        
        while uncovered.any():
            # Vectorized gain computation
            uncovered_expanded = uncovered[neighbors_arr]
            gains = uncovered_expanded.sum(axis=1)
            
            valid = gains > 0
            if not valid.any():
                break
            
            max_gain = gains[valid].max()
            best_candidates = np.where((gains == max_gain) & valid)[0]
            
            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False
        
        return code

    def reduce_code_fast(code):
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

    def local_search_improve(code, deadline):
        code = list(code)
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
                counts[neighbors_arr[c_remove, :]] -= 1
                
                if np.any(counts == 0):
                    counts[neighbors_arr[c_remove, :]] += 1
                    
                    critical_words = neighbors_arr[c_remove, :][counts[neighbors_arr[c_remove, :]] == 1]
                    
                    if len(critical_words) == 0:
                        continue
                    
                    # Find replacement using reverse_cover
                    # Candidates that cover all critical words
                    candidate_set = None
                    for cw in critical_words:
                        rc = reverse_cover_arr[cw, :reverse_cover_counts[cw]]
                        if candidate_set is None:
                            candidate_set = set(rc.tolist())
                        else:
                            candidate_set &= set(rc.tolist())
                    
                    if candidate_set is None or len(candidate_set) == 0:
                        continue
                    
                    candidate_set.discard(c_remove)
                    if len(candidate_set) == 0:
                        continue
                    
                    best_w = random.choice(list(candidate_set))
                    code[i] = best_w
                    counts[neighbors_arr[c_remove, :]] -= 1
                    counts[neighbors_arr[best_w, :]] += 1
                    improved = True
                    break
                else:
                    code.pop(i)
                    improved = True
                    break
        
        return code

    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        code = greedy_construct_vectorized(seed)
        code = reduce_code_fast(code)
        
        if len(code) < best_size or best_code is None:
            if time.time() < deadline - 10:
                code = local_search_improve(code, deadline - 5)
                code = reduce_code_fast(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
        
        seed += 1
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
