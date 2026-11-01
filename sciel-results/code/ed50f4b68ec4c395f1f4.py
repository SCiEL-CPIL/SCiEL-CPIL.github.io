"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Optimized for q=3, n=6, radius=1 using targeted local search.
    """
    if q != 3 or n != 6 or radius != 1:
        return _generic_greedy(q, n, radius)

    total_words = 3 ** n
    all_words = list(itertools.product(range(3), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors indices
    neighbors = [[] for _ in range(total_words)]
    for i, w in enumerate(all_words):
        nb = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    new_w = list(w)
                    new_w[pos] = new_val
                    nb.append(word_to_idx[tuple(new_w)])
        neighbors[i] = nb

    # Initial Code: Tetracode based (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_code.append(tuple(h + list(tail)))
            
    code = set(initial_code)
    code_indices = [word_to_idx[c] for c in code]
    
    # Coverage counts
    cov_count = np.zeros(total_words, dtype=int)
    for idx in code_indices:
        cov_count[neighbors[idx]] += 1
        
    current_size = len(code)
    
    improved = True
    # Limit iterations to ensure time safety
    max_iters = 500
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # 1. Try Removals
        for idx in list(code_indices):
            cov_count[neighbors[idx]] -= 1
            if np.min(cov_count) > 0:
                code_indices.remove(idx)
                code.remove(all_words[idx])
                current_size -= 1
                improved = True
                break
            else:
                cov_count[neighbors[idx]] += 1
        
        # 2. Try Swaps
        if not improved:
            for idx_remove in list(code_indices):
                cov_count[neighbors[idx_remove]] -= 1
                deficits = np.where(cov_count == 0)[0]
                
                if len(deficits) == 0:
                    code_indices.remove(idx_remove)
                    code.remove(all_words[idx_remove])
                    current_size -= 1
                    improved = True
                    break
                else:
                    found_swap = False
                    for idx_add in range(total_words):
                        if idx_add in code:
                            continue
                        
                        is_valid = True
                        for d in deficits:
                            if d not in neighbors[idx_add]:
                                is_valid = False
                                break
                        
                        if is_valid:
                            code_indices.remove(idx_remove)
                            code.remove(all_words[idx_remove])
                            code_indices.append(idx_add)
                            code.add(all_words[idx_add])
                            cov_count[neighbors[idx_add]] += 1
                            improved = True
                            found_swap = True
                            break
                            
                    if found_swap:
                        break
                    else:
                        cov_count[neighbors[idx_remove]] += 1
                        
    return [list(all_words[idx]) for idx in code_indices]

def _generic_greedy(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    neighbors = []
    for w in all_words:
        nb = []
        for v in all_words:
            if sum(1 for i in range(n) if w[i] != v[i]) <= radius:
                nb.append(word_to_idx[v])
        neighbors.append(nb)
        
    uncovered = set(range(len(all_words)))
    code = []
    while uncovered:
        best_idx = -1
        max_cov = -1
        for i in range(len(all_words)):
            if i in code:
                continue
            cov = len(set(neighbors[i]) & uncovered)
            if cov > max_cov:
                max_cov = cov
                best_idx = i
        if best_idx == -1:
            break
        code.append(best_idx)
        uncovered -= set(neighbors[best_idx])
    return [list(all_words[idx]) for idx in code]

# EVOLVE-BLOCK-END
