import itertools
import random
import numpy as np

def construct(v, k, t):
    # 1. Precomputation
    points = list(range(v))
    t_subs = list(itertools.combinations(points, t))
    n_t = len(t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # Precompute masks for all k-blocks
    # Note: 2^2024 is huge, but Python ints handle it.
    # However, generating 346k masks might take some time.
    # Let's generate them on the fly or cache them.
    
    block_combinations = list(itertools.combinations(points, k))
    n_blocks = len(block_combinations)
    
    # Map block tuple to index
    block_to_idx = {block: i for i, block in enumerate(block_combinations)}
    
    # Precompute masks for all blocks
    # This is the expensive part.
    # 346,104 blocks * 35 t-subsets per block = ~12M operations.
    # Should be fast enough (< 5s).
    block_masks = [0] * n_blocks
    for i, block in enumerate(block_combinations):
        mask = 0
        # Generate all t-subsets of this block
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    full_mask = (1 << n_t) - 1
    
    # 2. Initial Greedy Solution
    def greedy_cover():
        uncovered = full_mask
        selected_blocks = []
        remaining_blocks = list(range(n_blocks))
        
        while uncovered and remaining_blocks:
            best_block_idx = -1
            best_gain = -1
            # Optimization: Instead of checking all remaining blocks every time,
            # we can just check all. 346k is small.
            for b_idx in remaining_blocks:
                gain = bin(block_masks[b_idx] & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
                
            selected_blocks.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]
            remaining_blocks.remove(best_block_idx)
            
        return selected_blocks

    # Run greedy to get a valid starting point
    initial_blocks = greedy_cover()
    
    # 3. Simulated Annealing
    def popcount(x):
        return bin(x).count('1')
        
    def get_cost(blocks):
        # blocks is a list of indices
        if not blocks:
            return v * v * v # Huge penalty
        covered = 0
        for b in blocks:
            covered |= block_masks[b]
        uncovered = popcount(full_mask ^ covered)
        return len(blocks) + 100 * uncovered
        
    current_blocks = initial_blocks[:]
    current_cost = get_cost(current_blocks)
    best_blocks = current_blocks[:]
    best_cost = current_cost
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    max_iter = 50000
    rng = random.Random(42)
    
    # Precompute list of all block indices for random selection
    all_block_indices = list(range(n_blocks))
    
    for i in range(max_iter):
        if T < min_T:
            break
            
        # Pick a move: Remove or Add
        if rng.random() < 0.5 and current_blocks:
            # Remove
            idx_to_remove = rng.randrange(len(current_blocks))
            new_blocks = current_blocks[:]
            del new_blocks[idx_to_remove]
        else:
            # Add
            new_blocks = current_blocks[:]
            # Pick a random block not in current set? 
            # Just pick any, duplicates are handled by set logic? 
            # The problem says blocks must be distinct. 
            # So we should ensure we don't add a block already in the set.
            # For simplicity, we pick a random block. If it's already there, cost doesn't change (or does it? No, distinct blocks required).
            # If we add a duplicate, it's invalid. 
            # Let's pick a random block from all_blocks.
            # To avoid duplicates, we can check.
            # Given 346k blocks, collision is rare if we just pick random.
            # But if we pick one already in set, the set doesn't change, cost is same.
            # This is a "null move". Acceptable.
            b_idx = rng.choice(all_block_indices)
            if b_idx not in new_blocks:
                new_blocks.append(b_idx)
                
        new_cost = get_cost(new_blocks)
        delta = new_cost - current_cost
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_blocks = new_blocks
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_blocks = current_blocks[:]
                
        T *= cooling
        
    # Convert best block indices to lists of points
    result = [list(block_combinations[idx]) for idx in best_blocks]
    
    # Ensure distinctness and validity (sanity check)
    # The SA logic ensures distinctness by checking 'in' before adding, 
    # but the initial greedy ensures distinctness.
    # However, if we removed a block and added the same one, it's fine.
    # The 'in' check prevents duplicates.
    
    return result
