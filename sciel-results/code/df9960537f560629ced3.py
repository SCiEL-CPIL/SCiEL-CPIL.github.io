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
    
    # Precompute neighbor index arrays as numpy for fast operations
    neighbor_arrays = [np.array(neighbors[i], dtype=np.int64) for i in range(total_words)]
    neighbor_counts = np.array([len(neighbors[i]) for i in range(total_words)], dtype=np.int64)
    
    # Build a sparse representation: for each word, list of codewords that cover it
    # reverse_cover[j] = list of i such that j is in neighbors[i]
    
    # Precompute: for each codeword i, the set of words it covers as a frozenset for fast intersection
    cover_sets = [frozenset(neighbors[i]) for i in range(total_words)]
    
    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = set(range(total_words))
        code = []
        
        while uncovered:
            best_gain = 0
            best_candidates = []
            
            for i in range(total_words):
                gain = len(uncovered & cover_sets[i])
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain == 0:
                break
            
            chosen = random.choice(best_candidates)
            code.append(chosen)
            uncovered -= cover_sets[chosen]
        
        return code

    def reduce_code(code):
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            counts[neighbors[c]] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbors[c]] -= 1
                
                can_remove = np.all(counts[neighbors[c]] > 0)
                
                if can_remove:
                    changed = True
                else:
                    counts[neighbors[c]] += 1
                    new_code.append(c)
            code = new_code
        return code

    def local_search(code, deadline):
        code = list(code)
        if not code:
            return code
            
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            counts[neighbors[c]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                nbrs_remove = neighbors[c_remove]
                
                # Try Removal
                counts[nbrs_remove] -= 1
                
                holes = [w for w in nbrs_remove if counts[w] == 0]
                
                if not holes:
                    code.pop(i)
                    improved = True
                    break
                else:
                    counts[nbrs_remove] += 1
                    
                    # Try Swap: find codewords that cover ALL holes
                    holes_set = set(holes)
                    candidates = set()
                    for h in holes:
                        candidates.update(reverse_cover[h])
                    candidates = [c for c in candidates if all(h in cover_sets[c] for h in holes)]
                    candidates = [c for c in candidates if c != c_remove]
                    
                    if candidates:
                        # Score by waste: sum of counts on neighbors not in holes
                        hole_set = set(holes)
                        best_cand = None
                        best_score = float('inf')
                        
                        for cand in candidates:
                            waste = 0
                            for w in neighbors[cand]:
                                if w not in hole_set:
                                    waste += counts[w]
                            if waste < best_score:
                                best_score = waste
                                best_cand = cand
                        
                        if best_cand is not None:
                            counts[nbrs_remove] -= 1
                            counts[neighbors[best_cand]] += 1
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
            counts[neighbors[c]] += 1
        
        critical_scores = []
        for i, c in enumerate(code):
            crit = np.sum(counts[neighbors[c]] == 1)
            critical_scores.append(crit)
        
        indices = list(range(len(code)))
        indices.sort(key=lambda x: critical_scores[x])
        
        remove_indices = indices[:k]
        
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
        
        uncovered = set(range(total_words))
        for c in code:
            uncovered -= cover_sets[c]
        
        while uncovered and time.time() < deadline:
            best_gain = 0
            best_candidates = []
            
            for i in range(total_words):
                gain = len(uncovered & cover_sets[i])
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain == 0:
                break
            
            chosen = random.choice(best_candidates)
            code.append(chosen)
            uncovered -= cover_sets[chosen]
            
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
