import itertools
import random

def construct(v, k, t):
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map t-subset to index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Compute mask for each block
    block_masks = []
    for block in all_blocks:
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    # Full mask of all t-subsets
    full_mask = (1 << n_tsubs) - 1
    
    def run_greedy(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        remaining = set(range(n_blocks))
        
        while True:
            if uncovered == 0:
                break
                
            best_gain = -1
            candidates = []
            
            # Find max gain
            for b in remaining:
                gain = (block_masks[b] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [b]
                elif gain == best_gain:
                    candidates.append(b)
                    
            if best_gain <= 0:
                break
                
            # Pick random candidate
            best_block = rng.choice(candidates)
            selected.append(best_block)
            remaining.discard(best_block)
            uncovered &= ~block_masks[best_block]
            
        return selected

    def local_search(selected_indices, seed):
        rng = random.Random(seed + 100)
        selected_set = set(selected_indices)
        covered = 0
        for b in selected_set:
            covered |= block_masks[b]
            
        # Try to remove or swap
        # We need a list of available blocks not in selected_set
        # To speed up swap search, we might just check all blocks not in selected_set
        # This is O(N) per check. If we do this for each selected block (150) and each check is 50k, it's 7.5M ops. Fast.
        
        improved = True
        while improved:
            improved = False
            # Shuffle selected to randomize order
            sel_list = list(selected_set)
            rng.shuffle(sel_list)
            
            for b in sel_list:
                if b not in selected_set: continue # might have been removed
                
                # Check if b is redundant
                covered_without_b = covered & ~block_masks[b]
                if covered_without_b == covered:
                    # Redundant
                    selected_set.remove(b)
                    covered = covered_without_b
                    improved = True
                else:
                    # Try to find a swap
                    # We need a block b' not in selected_set such that block_masks[b'] covers (covered ^ covered_without_b)
                    lost = covered ^ covered_without_b
                    found_swap = False
                    for b_prime in range(n_blocks):
                        if b_prime in selected_set:
                            continue
                        if (lost & ~block_masks[b_prime]) == 0:
                            # b_prime covers all lost subsets
                            # Perform swap
                            selected_set.remove(b)
                            selected_set.add(b_prime)
                            covered = covered_without_b | block_masks[b_prime]
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        continue # Move to next b, as set changed
                    # Else no swap found, keep b

        return list(selected_set)

    best_solution = None
    best_len = float('inf')
    
    # Run multiple seeds
    for seed in range(10):
        greedy_sol = run_greedy(seed)
        # Local search
        ls_sol = local_search(greedy_sol, seed)
        
        if len(ls_sol) < best_len:
            best_len = len(ls_sol)
            best_solution = ls_sol
            
        # Early stop if we hit a very low bound?
        # Lower bound is 131. If we get 140, we can stop? 
        # But we want to minimize.
        
    # Convert indices to blocks
    return [list(all_blocks[b]) for b in best_solution]
