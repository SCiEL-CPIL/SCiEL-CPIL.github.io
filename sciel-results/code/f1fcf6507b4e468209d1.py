import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_tuple = [tuple(s) for s in t_subsets]
    t_index = {t: i for i, t in enumerate(t_subsets_tuple)}
    n_t_subsets = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k_subsets = len(k_subsets)
    
    k_to_t_indices = []
    for k_block in k_subsets:
        indices = []
        for t_sub in itertools.combinations(k_block, t):
            indices.append(t_index[t_sub])
        k_to_t_indices.append(indices)
        
    t_to_k_indices = [[] for _ in range(n_t_subsets)]
    for k_idx, indices in enumerate(k_to_t_indices):
        for t_idx in indices:
            t_to_k_indices[t_idx].append(k_idx)

    best_result = None
    best_size = float('inf')
    
    # Multi-start greedy
    num_runs = 10
    
    for run in range(num_runs):
        if time.time() > deadline - 5:
            break
            
        cover_count = [0] * n_t_subsets
        selected_k_indices = []
        
        while True:
            if time.time() > deadline:
                break
                
            min_count = min(cover_count)
            if min_count > 0:
                break
                
            # Find all t-subsets with min_count
            # To allow randomness in tie-breaking for the target,
            # we can pick a random one from the min set.
            min_indices = [i for i, c in enumerate(cover_count) if c == min_count]
            target_t_idx = random.choice(min_indices)
            
            candidates = t_to_k_indices[target_t_idx]
            
            best_k_idx = -1
            best_gain = -1
            best_candidates = []
            
            for k_idx in candidates:
                gain = 0
                for t_idx in k_to_t_indices[k_idx]:
                    if cover_count[t_idx] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_k_idx = k_idx
                    best_candidates = [k_idx]
                elif gain == best_gain:
                    best_candidates.append(k_idx)
                    
            if best_k_idx == -1:
                break
                
            # Random tie-breaking among best candidates
            chosen_k_idx = random.choice(best_candidates)
            
            selected_k_indices.append(chosen_k_idx)
            for t_idx in k_to_t_indices[chosen_k_idx]:
                cover_count[t_idx] += 1
                
        # Post-processing: Remove redundant blocks
        final_cover_count = [0] * n_t_subsets
        for k_idx in selected_k_indices:
            for t_idx in k_to_t_indices[k_idx]:
                final_cover_count[t_idx] += 1
                
        to_remove = set()
        for k_idx in selected_k_indices:
            redundant = True
            for t_idx in k_to_t_indices[k_idx]:
                if final_cover_count[t_idx] <= 1:
                    redundant = False
                    break
            if redundant:
                to_remove.add(k_idx)
                
        if to_remove:
            selected_k_indices = [k_idx for k_idx in selected_k_indices if k_idx not in to_remove]
            
        if len(selected_k_indices) < best_size:
            best_size = len(selected_k_indices)
            best_result = [list(k_subsets[k_idx]) for k_idx in selected_k_indices]
            
    if best_result is None:
        # Fallback to a single deterministic run if all failed (shouldn't happen)
        # Just return the first run's result if it exists
        if selected_k_indices:
             return [list(k_subsets[k_idx]) for k_idx in selected_k_indices]
        return []
        
    return best_result
