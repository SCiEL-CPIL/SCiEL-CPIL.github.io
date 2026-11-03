import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets.sort()
    t_sub_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t_subs = len(t_subsets)
    full_mask = (1 << num_t_subs) - 1
    
    k_subsets = list(itertools.combinations(range(v), k))
    k_subsets.sort()
    num_k_subs = len(k_subsets)
    
    # Precompute masks for each k-block
    block_masks = [0] * num_k_subs
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    best_count = float('inf')
    best_solution = []
    
    # 2. Search Loop
    while time.time() < deadline:
        uncovered_mask = full_mask
        selected_indices = []
        
        # Greedy Construction
        while uncovered_mask:
            best_gain = -1
            candidates = []
            
            # Find best blocks
            for i in range(num_k_subs):
                gain = (block_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if best_gain == 0:
                break
                
            # Random tie-breaking to explore different solutions
            chosen = random.choice(candidates)
            selected_indices.append(chosen)
            uncovered_mask &= ~block_masks[chosen]
            
        # Reduction Phase
        if not selected_indices:
            continue
            
        changed = True
        while changed:
            changed = False
            n = len(selected_indices)
            if n == 0:
                break
            
            # Compute prefix and suffix unions
            prefix = [0] * (n + 1)
            suffix = [0] * (n + 1)
            
            for i in range(n):
                prefix[i+1] = prefix[i] | block_masks[selected_indices[i]]
            for i in range(n-1, -1, -1):
                suffix[i] = suffix[i+1] | block_masks[selected_indices[i]]
                
            new_indices = []
            for i in range(n):
                others_mask = prefix[i] | suffix[i+1]
                # Check if selected_indices[i] is redundant
                if (block_masks[selected_indices[i]] & ~others_mask) == 0:
                    changed = True
                else:
                    new_indices.append(selected_indices[i])
            
            selected_indices = new_indices
            
        if len(selected_indices) < best_count:
            best_count = len(selected_indices)
            best_solution = [k_subsets[i] for i in selected_indices]
            
            # Early exit if we reach the known best
            if best_count <= 151:
                break

    return best_solution
