import itertools
import numpy as np
from math import comb
import time

def construct(v, k, t):
    start_time = time.time()
    # Use 110s to be safe against the 120s limit
    deadline = start_time + 110.0
    
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # block_triple_indices[i, j] is the index of the j-th triple in block i
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triple_to_idx[sub]
            
    def greedy_cover(random_seed):
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            gains = np.sum(uncovered_mask[block_triple_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            
            block_idx = block_triple_indices[best_idx]
            uncovered_mask[block_idx] = False
            remaining -= max_gain
            selected_blocks.append(best_idx)
            
        return selected_blocks

    def remove_redundants(selected_indices):
        current_selected = list(selected_indices)
        
        while True:
            cover_count = np.zeros(n_triples, dtype=np.int32)
            for b_idx in current_selected:
                for t_idx in block_triple_indices[b_idx]:
                    cover_count[t_idx] += 1
            
            to_remove = []
            for b_idx in current_selected:
                block_triples = block_triple_indices[b_idx]
                if np.min(cover_count[block_triples]) > 1:
                    to_remove.append(b_idx)
            
            if not to_remove:
                break
                
            remove_set = set(to_remove)
            current_selected = [b for b in current_selected if b not in remove_set]
            
        return current_selected

    def local_search(selected_indices, time_limit):
        current_selected = list(selected_indices)
        current_size = len(current_selected)
        selected_set = set(current_selected)
        
        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b_idx in current_selected:
            for t_idx in block_triple_indices[b_idx]:
                cover_count[t_idx] += 1
        
        critical_mask = (cover_count == 1)
        
        while time.time() < time_limit:
            critical_gains = np.sum(critical_mask[block_triple_indices], axis=1)
            
            # Mask selected blocks
            for b_idx in current_selected:
                critical_gains[b_idx] = -1
                
            # Find top 5 candidates
            candidates = []
            temp_gains = critical_gains.copy()
            for _ in range(5):
                max_idx = np.argmax(temp_gains)
                if temp_gains[max_idx] <= 0:
                    break
                if max_idx not in selected_set:
                    candidates.append(max_idx)
                temp_gains[max_idx] = -1
            
            if not candidates:
                break
                
            improved = False
            for b_new in candidates:
                temp_selected = current_selected + [b_new]
                new_selected = remove_redundants(temp_selected)
                
                if len(new_selected) < current_size:
                    current_selected = new_selected
                    current_size = len(current_selected)
                    selected_set = set(current_selected)
                    improved = True
                    
                    # Update cover_count
                    cover_count = np.zeros(n_triples, dtype=np.int32)
                    for b_idx in current_selected:
                        for t_idx in block_triple_indices[b_idx]:
                            cover_count[t_idx] += 1
                    critical_mask = (cover_count == 1)
                    break
            
            if not improved:
                break
                
        return current_selected

    best_blocks = None
    best_len = float('inf')
    
    restart_count = 0
    while time.time() < deadline:
        seed = restart_count
        selected_indices = greedy_cover(seed)
        selected_indices = remove_redundants(selected_indices)
        
        # Allow 2 seconds for local search per restart
        ls_deadline = min(deadline, time.time() + 2.0)
        selected_indices = local_search(selected_indices, ls_deadline)
        
        if len(selected_indices) < best_len:
            best_len = len(selected_indices)
            best_blocks = [list(all_blocks[i]) for i in selected_indices]
            
        restart_count += 1
        
    if best_blocks is None:
        selected_indices = greedy_cover(0)
        selected_indices = remove_redundants(selected_indices)
        best_blocks = [list(all_blocks[i]) for i in selected_indices]
        
    return best_blocks
