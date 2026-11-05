import itertools
import time
import random
import numpy as np
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    block_t_indices = []
    for block in all_blocks:
        subs = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_t_indices.append(subs)
        
    t_to_blocks = [[] for _ in range(n_t)]
    for b_idx, subs in enumerate(block_t_indices):
        for s_idx in subs:
            t_to_blocks[s_idx].append(b_idx)

    k_choose_t = len(list(itertools.combinations(range(k), t)))
    row_indices = np.repeat(np.arange(n_blocks), k_choose_t)
    col_indices = np.array([idx for subs in block_t_indices for idx in subs])
    data = np.ones(len(col_indices), dtype=np.int8)
    cover_matrix = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_blocks, n_t))

    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        selected = []
        used = np.zeros(n_blocks, dtype=bool)
        
        while uncovered.any():
            if time.time() > deadline - 5.0:
                break
            
            gains = (cover_matrix @ uncovered.astype(np.int8)).astype(np.int32)
            gains[used] = -1
            
            max_gain = gains.max()
            if max_gain <= 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            if seed == 0:
                chosen = candidates[0]
            else:
                chosen = candidates[rng.randint(0, len(candidates) - 1)]
                
            selected.append(chosen)
            used[chosen] = True
            uncovered[block_t_indices[chosen]] = False
            
        return selected

    def local_search(selected, max_time=10.0):
        if not selected:
            return selected
            
        selected = list(selected)
        selected_set = set(selected)
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            cover_count[block_t_indices[b]] += 1
            
        improved = True
        start_ls = time.time()
        
        while improved and time.time() < start_ls + max_time:
            improved = False
            
            # 1. Try to remove redundant blocks
            for i in range(len(selected) - 1, -1, -1):
                b = selected[i]
                subs = block_t_indices[b]
                if np.min(cover_count[subs]) > 1:
                    cover_count[subs] -= 1
                    selected.pop(i)
                    selected_set.remove(b)
                    improved = True
                    break
                    
            if improved:
                continue
                
            # 2. Try swaps
            # Randomize order to avoid bias
            indices = list(range(len(selected)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > start_ls + max_time:
                    break
                    
                b = selected[i]
                subs = block_t_indices[b]
                essential = subs[cover_count[subs] == 1]
                
                if len(essential) == 0:
                    continue # Should not happen if removal worked
                
                first_ess = essential[0]
                candidates = t_to_blocks[first_ess]
                
                best_cand = -1
                best_risk = float('inf')
                
                # Sample candidates if too many
                if len(candidates) > 200:
                    candidates = random.sample(candidates, 200)
                    
                for cand in candidates:
                    if cand == b or cand in selected_set:
                        continue
                        
                    # Check if cand covers all essential
                    cand_subs = block_t_indices[cand]
                    cand_set = set(cand_subs)
                    if not all(e in cand_set for e in essential):
                        continue
                        
                    # Calculate risk: number of subsets in b \ cand that have count == 2
                    # These will become essential (count 1) after swap
                    # We want to minimize this
                    risk = 0
                    for s in subs:
                        if s not in cand_set:
                            if cover_count[s] == 2:
                                risk += 1
                                # Early exit if risk is already high
                                if risk >= best_risk:
                                    break
                                    
                    if risk < best_risk:
                        best_risk = risk
                        best_cand = cand
                        
                if best_cand != -1:
                    # Perform swap
                    old_subs = block_t_indices[b]
                    new_subs = block_t_indices[best_cand]
                    
                    cover_count[old_subs] -= 1
                    cover_count[new_subs] += 1
                    
                    selected_set.remove(b)
                    selected_set.add(best_cand)
                    selected[i] = best_cand
                    improved = True
                    break # Restart loop
                    
        return selected

    best_blocks = None
    best_len = float('inf')
    
    selected = greedy_cover(0)
    selected = local_search(selected, max_time=5.0)
    if selected:
        best_blocks = selected
        best_len = len(selected)
        
    n_restarts = 20
    for seed in range(1, n_restarts + 1):
        if time.time() > deadline - 10.0:
            break
            
        selected = greedy_cover(seed)
        selected = local_search(selected, max_time=5.0)
        if selected and len(selected) < best_len:
            best_blocks = selected
            best_len = len(selected)
            
    if best_blocks and time.time() < deadline - 5.0:
        best_blocks = local_search(best_blocks, max_time=20.0)
        
    result_blocks = [list(all_blocks[i]) for i in best_blocks]
    
    return result_blocks
