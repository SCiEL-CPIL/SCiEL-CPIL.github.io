import itertools
import time
import random
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precompute all t-subsets (triplets)
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to a bitmask of the triplets it covers
    block_masks = [0] * n_blocks
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        block_masks[b_idx] = mask
        
    # Full mask of all triplets
    all_mask = (1 << n_triplets) - 1

    def is_valid(blocks):
        covered = 0
        for b_idx in blocks:
            covered |= block_masks[b_idx]
        return covered == all_mask

    def greedy_set_cover(seed=0):
        rng = random.Random(seed)
        uncovered = all_mask
        solution_indices = []
        
        while uncovered:
            if time.time() > deadline - 10:
                break
                
            best_gain = -1
            best_candidates = []
            
            for b_idx in range(n_blocks):
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [b_idx]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(b_idx)
                    
            if not best_candidates:
                break
                
            chosen_idx = rng.choice(best_candidates)
            solution_indices.append(chosen_idx)
            uncovered &= ~block_masks[chosen_idx]
            
        return solution_indices

    def remove_redundant(blocks):
        if not blocks:
            return blocks
            
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b_idx in blocks:
            block = all_blocks[b_idx]
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                counts[tid] += 1
                
        new_blocks = []
        for b_idx in blocks:
            block = all_blocks[b_idx]
            redundant = True
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                if counts[tid] == 1:
                    redundant = False
                    break
            if not redundant:
                new_blocks.append(b_idx)
                
        return new_blocks

    def simulated_annealing(initial_blocks, seed=0):
        rng = random.Random(seed)
        current_blocks = list(initial_blocks)
        
        def get_cost(blocks):
            if not blocks:
                return 0, all_mask
            covered = 0
            for b_idx in blocks:
                covered |= block_masks[b_idx]
            uncovered = all_mask & ~covered
            uncovered_count = uncovered.bit_count()
            return len(blocks) + 1000 * uncovered_count, uncovered
        
        current_cost, current_uncovered = get_cost(current_blocks)
        
        T = 50.0
        T_min = 0.1
        cooling = 0.999
        
        while time.time() < deadline - 5:
            if T < T_min:
                T = T_min
                
            move_type = rng.random()
            new_blocks = current_blocks
            new_cost, new_uncovered = current_cost, current_uncovered
            
            if move_type < 0.7:
                # Remove
                if len(current_blocks) > 0:
                    idx_to_remove = rng.randrange(len(current_blocks))
                    new_blocks = current_blocks.copy()
                    new_blocks.pop(idx_to_remove)
                    new_cost, new_uncovered = get_cost(new_blocks)
            elif move_type < 0.9:
                # Replace
                if len(current_blocks) > 0:
                    idx_to_replace = rng.randrange(len(current_blocks))
                    new_block_idx = rng.randrange(n_blocks)
                    if new_block_idx not in current_blocks:
                        new_blocks = current_blocks.copy()
                        new_blocks[idx_to_replace] = new_block_idx
                        new_cost, new_uncovered = get_cost(new_blocks)
            else:
                # Add
                if current_uncovered:
                    best_gain = -1
                    best_idx = -1
                    for _ in range(50):
                        b_idx = rng.randrange(n_blocks)
                        gain = (block_masks[b_idx] & current_uncovered).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = b_idx
                    if best_idx != -1 and best_idx not in current_blocks:
                        new_blocks = current_blocks.copy()
                        new_blocks.append(best_idx)
                        new_cost, new_uncovered = get_cost(new_blocks)

            delta = new_cost - current_cost
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_blocks = new_blocks
                current_cost = new_cost
                current_uncovered = new_uncovered
                
            T *= cooling
            
        return current_blocks

    # 1. Get initial solution
    init_blocks = greedy_set_cover(0)
    init_blocks = remove_redundant(init_blocks)
    
    best_blocks = init_blocks
    best_count = len(init_blocks)
    
    # 2. Simulated Annealing
    num_sa_seeds = 10
    for seed in range(num_sa_seeds):
        if time.time() > deadline - 10:
            break
            
        sa_blocks = simulated_annealing(init_blocks, seed=seed)
        
        # Check validity
        if is_valid(sa_blocks):
            sa_blocks = remove_redundant(sa_blocks)
            if len(sa_blocks) < best_count:
                best_count = len(sa_blocks)
                best_blocks = sa_blocks
                init_blocks = sa_blocks

    return [list(all_blocks[i]) for i in best_blocks]

if __name__ == "__main__":
    pass
