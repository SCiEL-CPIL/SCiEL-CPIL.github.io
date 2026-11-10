import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute t-subset indices for each block as a frozenset for fast intersection
    block_subsets = []
    for b in all_blocks:
        subs = frozenset(t_subset_to_idx[s] for s in itertools.combinations(b, t))
        block_subsets.append(subs)
        
    best_blocks = []
    best_count = float('inf')
    
    # Run multiple randomized greedy attempts
    # The number of attempts depends on speed.
    # One greedy run takes ~T seconds.
    # We can estimate T after first run.
    
    attempts = 0
    while time.time() < deadline - 5:
        uncovered = set(range(num_t_subsets))
        selected_block_indices = []
        
        while uncovered:
            best_gain = 0
            max_gain_blocks = []
            
            # Find blocks with max gain
            # Optimization: If we find a block with gain == len(uncovered), we can stop early?
            # Max possible gain is len(uncovered).
            # But unlikely.
            
            for b_idx in range(num_blocks):
                gain = len(uncovered.intersection(block_subsets[b_idx]))
                if gain > best_gain:
                    best_gain = gain
                    max_gain_blocks = [b_idx]
                elif gain == best_gain and gain > 0:
                    max_gain_blocks.append(b_idx)
                    
            if best_gain == 0:
                break
                
            # Random tie-breaking
            chosen_idx = random.choice(max_gain_blocks)
            selected_block_indices.append(chosen_idx)
            uncovered.difference_update(block_subsets[chosen_idx])
            
            if time.time() > deadline:
                break
                
        # Pruning
        if selected_block_indices:
            coverage_count = [0] * num_t_subsets
            for b_idx in selected_block_indices:
                for s_idx in block_subsets[b_idx]:
                    coverage_count[s_idx] += 1
                    
            changed = True
            while changed:
                changed = False
                new_selected = []
                for b_idx in selected_block_indices:
                    is_essential = False
                    for s_idx in block_subsets[b_idx]:
                        if coverage_count[s_idx] == 1:
                            is_essential = True
                            break
                    if is_essential:
                        new_selected.append(b_idx)
                    else:
                        for s_idx in block_subsets[b_idx]:
                            coverage_count[s_idx] -= 1
                        changed = True
                if changed:
                    selected_block_indices = new_selected
                    
            if len(selected_block_indices) < best_count:
                best_count = len(selected_block_indices)
                best_blocks = [all_blocks[i] for i in selected_block_indices]
                
        attempts += 1
        # If one attempt takes > 10s, we won't do many.
        # If one attempt takes < 1s, we can do many.
        if attempts == 1:
             # Estimate time
             pass

    return best_blocks
