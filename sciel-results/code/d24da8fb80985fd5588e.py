import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    k_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(k_blocks)
    
    block_masks = [0] * num_blocks
    block_t_subsets = [[] for _ in range(num_blocks)]
    
    for i, b in enumerate(k_blocks):
        mask = 0
        subs = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
            subs.append(idx)
        block_masks[i] = mask
        block_t_subsets[i] = subs
        
    # 2. Initial Greedy Construction
    uncovered_mask = (1 << num_t_subsets) - 1
    solution_indices = []
    
    while uncovered_mask:
        best_idx = -1
        best_count = 0
        # Find best block
        for i in range(num_blocks):
            count = (block_masks[i] & uncovered_mask).bit_count()
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1:
            break
            
        solution_indices.append(best_idx)
        uncovered_mask &= ~block_masks[best_idx]
        
        if time.time() > deadline:
            break
            
    # 3. Local Search (ILS)
    cover_counts = [0] * num_t_subsets
    for idx in solution_indices:
        for sub_idx in block_t_subsets[idx]:
            cover_counts[sub_idx] += 1
            
    best_solution = solution_indices.copy()
    best_len = len(solution_indices)
    
    def try_delete(sol, counts):
        removed_any = False
        i = 0
        while i < len(sol):
            idx = sol[i]
            can_remove = True
            for sub_idx in block_t_subsets[idx]:
                if counts[sub_idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                for sub_idx in block_t_subsets[idx]:
                    counts[sub_idx] -= 1
                sol.pop(i)
                removed_any = True
            else:
                i += 1
        return sol, counts, removed_any

    def repair(sol, counts):
        uncovered_mask = 0
        for i in range(num_t_subsets):
            if counts[i] == 0:
                uncovered_mask |= (1 << i)
                
        while uncovered_mask:
            best_idx = -1
            best_count = 0
            for i in range(num_blocks):
                count = (block_masks[i] & uncovered_mask).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            sol.append(best_idx)
            for sub_idx in block_t_subsets[best_idx]:
                counts[sub_idx] += 1
            uncovered_mask &= ~block_masks[best_idx]
            if time.time() > deadline:
                break
        return sol, counts

    while time.time() < deadline:
        # 1. Try to delete
        solution_indices, cover_counts, removed = try_delete(solution_indices, cover_counts)
        
        if removed:
            if len(solution_indices) < best_len:
                best_len = len(solution_indices)
                best_solution = solution_indices.copy()
        
        # 2. Perturb
        if len(solution_indices) > 0:
            num_remove = random.randint(1, min(3, len(solution_indices)))
            remove_indices = random.sample(range(len(solution_indices)), num_remove)
            for idx in sorted(remove_indices, reverse=True):
                block_idx = solution_indices[idx]
                for sub_idx in block_t_subsets[block_idx]:
                    cover_counts[sub_idx] -= 1
                solution_indices.pop(idx)
            
            # Repair
            solution_indices, cover_counts = repair(solution_indices, cover_counts)
            
            # Minimize again
            solution_indices, cover_counts, _ = try_delete(solution_indices, cover_counts)
            
            if len(solution_indices) < best_len:
                best_len = len(solution_indices)
                best_solution = solution_indices.copy()

    final_blocks = [list(k_blocks[idx]) for idx in best_solution]
    return final_blocks
