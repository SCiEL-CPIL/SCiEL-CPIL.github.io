import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute bitmask for each block and reverse mapping
    block_masks = [0] * num_blocks
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    block_t_subsets = [[] for _ in range(num_blocks)] # List of t-subset indices for each block
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        subs = list(itertools.combinations(block, t))
        for sub in subs:
            idx = t_subset_to_idx[sub]
            mask |= (1 << idx)
            t_to_blocks[idx].append(b_idx)
            block_t_subsets[b_idx].append(idx)
        block_masks[b_idx] = mask

    best_blocks = None
    best_count = float('inf')
    
    def prune_redundant(selected_indices):
        if not selected_indices:
            return []
        
        # Count coverage for each t-subset
        cover_counts = [0] * num_t_subsets
        for b_idx in selected_indices:
            for t_idx in block_t_subsets[b_idx]:
                cover_counts[t_idx] += 1
        
        improved = True
        current_blocks = list(selected_indices)
        
        while improved and time.time() < deadline:
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                # Check if redundant: all t-subsets in this block have count > 1
                redundant = True
                for t_idx in block_t_subsets[b_idx]:
                    if cover_counts[t_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove it
                    for t_idx in block_t_subsets[b_idx]:
                        cover_counts[t_idx] -= 1
                    improved = True
                else:
                    new_blocks.append(b_idx)
            current_blocks = new_blocks
            
        return current_blocks

    def local_search_improve(block_indices):
        current_blocks = list(block_indices)
        selected_set = set(current_blocks)
        
        # Maintain cover counts
        cover_counts = [0] * num_t_subsets
        for b_idx in current_blocks:
            for t_idx in block_t_subsets[b_idx]:
                cover_counts[t_idx] += 1
        
        improved = True
        last_change_time = time.time()
        
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Shuffle to find different swap opportunities
            random.shuffle(current_blocks)
            
            for b_old in current_blocks[:]:
                if time.time() > deadline - 2.0:
                    return current_blocks
                
                # Find unique t-subsets covered by b_old
                unique_mask = 0
                for t_idx in block_t_subsets[b_old]:
                    if cover_counts[t_idx] == 1:
                        unique_mask |= (1 << t_idx)
                
                if unique_mask == 0:
                    # This block is redundant, remove it
                    for t_idx in block_t_subsets[b_old]:
                        cover_counts[t_idx] -= 1
                    current_blocks.remove(b_old)
                    selected_set.remove(b_old)
                    improved = True
                    last_change_time = time.time()
                    continue
                
                # Try to find a block b_new that covers unique_mask
                # Optimization: Pick the "rarest" t-subset in unique_mask to guide search
                # We just pick the first set bit for simplicity, or one with fewest candidates
                lsb = unique_mask & -unique_mask
                target_t_idx = lsb.bit_length() - 1
                
                candidates = t_to_blocks[target_t_idx]
                
                # Shuffle candidates to avoid bias
                random.shuffle(candidates)
                
                found_swap = False
                for b_new in candidates:
                    if b_new in selected_set:
                        continue
                    
                    # Check if b_new covers ALL of unique_mask
                    if (block_masks[b_new] & unique_mask) == unique_mask:
                        # Valid swap!
                        # Perform swap
                        # Remove b_old
                        for t_idx in block_t_subsets[b_old]:
                            cover_counts[t_idx] -= 1
                        
                        # Add b_new
                        for t_idx in block_t_subsets[b_new]:
                            cover_counts[t_idx] += 1
                        
                        current_blocks.remove(b_old)
                        current_blocks.append(b_new)
                        selected_set.remove(b_old)
                        selected_set.add(b_new)
                        
                        improved = True
                        last_change_time = time.time()
                        found_swap = True
                        break
                
                # If no swap found, continue to next b_old
                # If swap found, we continue the loop, but counts changed.
        
        return current_blocks

    # Main Loop: Multiple Greedy Instances
    max_runs = 20 # Adjust based on speed
    
    for run_idx in range(max_runs):
        if time.time() > deadline - 30.0: # Leave 30s for local search on best
            break
            
        random.seed(run_idx)
        
        # --- Phase 1: Fast Greedy ---
        uncovered_mask = (1 << num_t_subsets) - 1
        selected_indices = []
        selected_set = set()
        
        # Initial gain: C(k, t)
        init_gain = len(block_t_subsets[0])
        gain = [init_gain] * num_blocks
        
        while uncovered_mask:
            if time.time() > deadline - 30.0:
                break
                
            # Find max gain
            max_gain = -1
            best_candidates = []
            
            # Linear scan
            for b_idx in range(num_blocks):
                if b_idx in selected_set:
                    continue
                g = gain[b_idx]
                # Add small noise for randomization
                g += random.random() * 0.01
                if g > max_gain:
                    max_gain = g
                    best_candidates = [b_idx]
                elif abs(g - max_gain) < 1e-9:
                    best_candidates.append(b_idx)
            
            if not best_candidates:
                break
                
            chosen_idx = random.choice(best_candidates)
            selected_indices.append(chosen_idx)
            selected_set.add(chosen_idx)
            
            # Update uncovered and gains
            newly_covered = uncovered_mask & block_masks[chosen_idx]
            uncovered_mask &= ~block_masks[chosen_idx]
            
            # Update gains for affected blocks
            m = newly_covered
            while m:
                lsb = m & -m
                t_idx = lsb.bit_length() - 1
                m ^= lsb
                for b_prime in t_to_blocks[t_idx]:
                    if b_prime not in selected_set:
                        gain[b_prime] -= 1
                        if gain[b_prime] < 0:
                            gain[b_prime] = 0
        
        if uncovered_mask != 0:
            continue
            
        # --- Phase 2: Prune ---
        selected_indices = prune_redundant(selected_indices)
        
        # --- Phase 3: Local Search (Swaps) ---
        # Only do local search if we have time
        if time.time() < deadline - 5.0:
            selected_indices = local_search_improve(selected_indices)
            
        if len(selected_indices) < best_count:
            best_count = len(selected_indices)
            best_blocks = [list(all_blocks[i]) for i in selected_indices]
            
            # If we found a very good solution, we might want to spend more time polishing it
            # But for now, we just keep the best.
            # To improve, we could restart local search on the best solution found so far
            # if we have time.
            
    # Final Polish on the best solution found
    if best_blocks is not None and time.time() < deadline - 2.0:
        # Convert back to indices for local search
        best_indices = [all_blocks.index(b) for b in best_blocks]
        # This index lookup is O(N^2) if not careful, but N is small.
        # Better: map block tuple to index
        block_to_idx = {b: i for i, b in enumerate(all_blocks)}
        best_indices = [block_to_idx[b] for b in best_blocks]
        
        polished = local_search_improve(best_indices)
        
        if len(polished) < best_count:
            best_count = len(polished)
            best_blocks = [list(all_blocks[i]) for i in polished]

    if best_blocks is None:
        return []
        
    return best_blocks
