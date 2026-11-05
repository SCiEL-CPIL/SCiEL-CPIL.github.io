import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    rng = random.Random(42)
    
    # 1. Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute coverage masks for all possible k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    block_masks = [0] * n_k_blocks
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    # 3. Greedy Construction (fast vectorized version)
    uncovered_mask = (1 << n_t_subs) - 1
    selected_block_indices = []
    
    # Convert masks to numpy array for vectorized operations
    mask_arr = np.array(block_masks, dtype=np.uint64) if n_t_subs <= 64 else np.array(block_masks, dtype=object)
    
    # For large bitmasks, we need a different approach. Let's use a coverage matrix.
    # coverage_matrix[i, j] = 1 if block i covers t-subset j
    coverage_matrix = np.zeros((n_k_blocks, n_t_subs), dtype=np.int32)
    for i, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            coverage_matrix[i, idx] = 1
    
    # Greedy: at each step, pick block that covers most uncovered t-subsets
    uncovered = np.ones(n_t_subs, dtype=np.int32)
    selected = []
    
    while uncovered.any():
        if time.time() > deadline - 30:
            break
        
        # Gain for each block = number of uncovered t-subsets it covers
        gains = coverage_matrix @ uncovered
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
        selected.append(best_idx)
        uncovered[coverage_matrix[best_idx] == 1] = 0
    
    # 4. Redundancy removal (vectorized)
    def remove_redundant(block_indices):
        if not block_indices:
            return block_indices
        
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for b in block_indices:
            cover_count += coverage_matrix[b]
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            new_blocks = []
            for b in block_indices:
                # Check if block b is the only cover for any t-subset
                block_subs = coverage_matrix[b] == 1
                if np.any(cover_count[block_subs] == 1):
                    new_blocks.append(b)
                else:
                    cover_count -= coverage_matrix[b]
                    improved = True
            block_indices = new_blocks
        return block_indices
    
    selected = remove_redundant(selected)
    
    best_count = len(selected)
    best_blocks = selected[:]
    
    # 5. Probabilistic Method with Alteration
    # Try multiple random initial selections with different probabilities
    # Then apply redundancy removal
    
    # Expected coverage: each t-subset is covered by a random block with prob p * (C(k-t, v-t) / C(k, v-t))
    # For v=19, k=7, t=4: C(3,15) / C(7,15) = 455 / 6435 ≈ 0.0707
    # So expected coverage per t-subset ≈ p * 0.0707
    # To cover all with high prob, need p * 0.0707 * n_t_subs ≈ ln(n_t_subs)
    # n_t_subs = C(19,4) = 3876, ln(3876) ≈ 8.26
    # p ≈ 8.26 / (0.0707 * 3876) ≈ 0.030
    
    prob_values = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
    n_trials_per_prob = 3
    
    for p in prob_values:
        for trial in range(n_trials_per_prob):
            if time.time() > deadline - 5:
                break
            
            # Randomly select blocks with probability p
            random_blocks = []
            for i in range(n_k_blocks):
                if rng.random() < p:
                    random_blocks.append(i)
            
            # Check if this random selection covers everything
            cover_count = np.zeros(n_t_subs, dtype=np.int32)
            for b in random_blocks:
                cover_count += coverage_matrix[b]
            
            if not cover_count.all():
                # Not covered, skip
                continue
            
            # Apply redundancy removal
            reduced = remove_redundant(random_blocks)
            
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_blocks = reduced[:]
    
    # 6. Local search: try swaps to improve further
    current_blocks = best_blocks[:]
    current_set = set(current_blocks)
    
    max_swap_rounds = 50
    for round_num in range(max_swap_rounds):
        if time.time() > deadline - 5:
            break
        
        n_swaps_per_round = 10
        for _ in range(n_swaps_per_round):
            if time.time() > deadline - 5:
                break
            
            if len(current_blocks) < 2:
                break
            
            # Pick a random block to remove
            remove_pos = rng.randrange(len(current_blocks))
            remove_block = current_blocks[remove_pos]
            
            # Pick a random block to add
            new_block = rng.randrange(n_k_blocks)
            if new_block in current_set:
                continue
            
            # Try the swap
            new_blocks = [b for i, b in enumerate(current_blocks) if i != remove_pos]
            new_blocks.append(new_block)
            
            # Check coverage
            cover_count = np.zeros(n_t_subs, dtype=np.int32)
            for b in new_blocks:
                cover_count += coverage_matrix[b]
            
            if not cover_count.all():
                continue
            
            # Apply redundancy removal
            reduced = remove_redundant(new_blocks)
            
            if len(reduced) < len(current_blocks):
                current_blocks = reduced
                current_set = set(current_blocks)
                if len(current_blocks) < best_count:
                    best_count = len(current_blocks)
                    best_blocks = current_blocks[:]
    
    # Final blocks
    blocks = [list(all_k_blocks[i]) for i in best_blocks]
    
    return blocks
