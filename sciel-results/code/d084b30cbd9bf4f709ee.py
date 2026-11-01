import itertools
import random
import time

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = [True] * num_t_subsets
        gain = [0] * num_k_subsets
        for b_idx in range(num_k_subsets):
            gain[b_idx] = len(block_t_indices[b_idx])
        
        selected = []
        while True:
            max_gain = max(gain)
            if max_gain == 0:
                break
            
            # Find all blocks with max gain
            candidates = [i for i, g in enumerate(gain) if g == max_gain]
            best_block = rng.choice(candidates)
            
            selected.append(best_block)
            
            for t_idx in block_t_indices[best_block]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx in t_subset_blocks[t_idx]:
                        gain[b_idx] -= 1
        return selected

    def local_search(selected):
        active = set(selected)
        
        # Calculate initial cover counts
        cover_count = [0] * num_t_subsets
        for b_idx in active:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
                
        # Iterative Deletion Pass
        changed = True
        while changed:
            changed = False
            # Shuffle to avoid bias
            for b_idx in list(active):
                # Check if redundant
                redundant = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_count[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    active.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True
        return list(active)

    best_solution = None
    best_size = float('inf')
    start_time = time.time()
    
    # Perform multiple restarts
    num_restarts = 100
    
    for i in range(num_restarts):
        if time.time() - start_time > 100:
            break
        sol = greedy_cover(i)
        sol = local_search(sol)
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol
            
    return [list(all_k_subsets[b]) for b in best_solution]
