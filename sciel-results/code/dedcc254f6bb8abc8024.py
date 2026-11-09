import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using randomized greedy set cover with
    softmax-based selection to escape local optima, followed by redundancy removal
    and local search refinement.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Precompute all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage bitmask for every possible k-block
    block_masks = []
    all_blocks = []
    
    # Generate all k-blocks and their masks
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    num_blocks = len(block_masks)
    
    # Helper for bit count
    def count_bits(x):
        if hasattr(x, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')
    
    # Efficient redundancy removal using zero_count tracking
    def remove_redundant(chosen_indices, time_budget):
        """Remove redundant blocks using efficient union tracking."""
        if len(chosen_indices) <= 1:
            return chosen_indices
        
        n = len(chosen_indices)
        masks = [block_masks[idx] for idx in chosen_indices]
        
        # Track how many times each t-subset is covered
        # Use a dictionary for sparse representation
        cover_count = {}
        for mask in masks:
            temp = mask
            while temp:
                # Extract lowest set bit
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                cover_count[idx] = cover_count.get(idx, 0) + 1
                temp &= temp - 1
        
        changed = True
        while changed and time.time() < time_budget:
            changed = False
            for i in range(len(chosen_indices)):
                mask_i = masks[i]
                temp = mask_i
                redundant = True
                while temp:
                    lsb = temp & (-temp)
                    idx = lsb.bit_length() - 1
                    if cover_count.get(idx, 0) <= 1:
                        redundant = False
                        break
                    temp &= temp - 1
                
                if redundant:
                    # Remove block i
                    for j in range(len(masks)):
                        if j != i:
                            # Decrement counts for blocks that share coverage with removed block
                            pass
                    # Simply rebuild cover_count
                    masks.pop(i)
                    chosen_indices.pop(i)
                    cover_count = {}
                    for mask in masks:
                        temp = mask
                        while temp:
                            lsb = temp & (-temp)
                            idx = lsb.bit_length() - 1
                            cover_count[idx] = cover_count.get(idx, 0) + 1
                            temp &= temp - 1
                    changed = True
                    break
        
        return chosen_indices
    
    # Randomized greedy with softmax selection
    def randomized_greedy(num_restarts, temperature_schedule):
        best_solution = None
        best_size = float('inf')
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            current_uncovered = (1 << num_t_subsets) - 1
            chosen = []
            current_union = 0
            step = 0
            
            while current_uncovered != 0:
                if time.time() > deadline - 3.0:
                    break
                
                # Compute gains for all blocks
                gains = []
                for i in range(num_blocks):
                    intersection = block_masks[i] & current_uncovered
                    gains.append(count_bits(intersection))
                
                max_gain = max(gains)
                if max_gain == 0:
                    break
                
                # Temperature for softmax (decays over time)
                temp_idx = min(step, len(temperature_schedule) - 1)
                temperature = temperature_schedule[temp_idx]
                
                if temperature < 0.01:
                    # Greedy: pick max gain
                    best_idx = gains.index(max_gain)
                else:
                    # Softmax selection: probability proportional to exp(gain / temperature)
                    # Only consider blocks with gain > 0
                    candidates = [(i, g) for i, g in enumerate(gains) if g > 0]
                    
                    if not candidates:
                        break
                    
                    # Compute softmax weights
                    scaled_gains = [g / temperature for _, g in candidates]
                    max_scaled = max(scaled_gains)
                    exp_weights = [np.exp(sg - max_scaled) for sg in scaled_gains]
                    total_weight = sum(exp_weights)
                    
                    # Sample
                    r = random.random() * total_weight
                    cumulative = 0
                    best_idx = candidates[-1][0]
                    for (idx, _), w in zip(candidates, exp_weights):
                        cumulative += w
                        if r <= cumulative:
                            best_idx = idx
                            break
                
                chosen.append(best_idx)
                current_uncovered &= ~block_masks[best_idx]
                current_union |= block_masks[best_idx]
                step += 1
            
            # Remove redundancies
            chosen = remove_redundant(chosen, deadline - 2.0)
            
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_solution = chosen[:]
        
        return best_solution
    
    # Try multiple randomized restarts with different temperature schedules
    best_solution = None
    best_size = float('inf')
    
    # Temperature schedules: start hot, cool down
    schedules = [
        [5.0, 3.0, 2.0, 1.0, 0.5, 0.1, 0.01],
        [3.0, 2.0, 1.5, 1.0, 0.5, 0.1, 0.01],
        [4.0, 2.5, 1.5, 0.8, 0.3, 0.1, 0.01],
        [2.0, 1.5, 1.0, 0.5, 0.2, 0.05, 0.01],
    ]
    
    num_restarts_per_schedule = 3
    
    for schedule in schedules:
        if time.time() > deadline - 10.0:
            break
        
        for _ in range(num_restarts_per_schedule):
            if time.time() > deadline - 5.0:
                break
            
            # Run randomized greedy with this schedule
            current_uncovered = (1 << num_t_subsets) - 1
            chosen = []
            current_union = 0
            step = 0
            
            while current_uncovered != 0:
                if time.time() > deadline - 3.0:
                    break
                
                # Compute gains for all blocks
                gains = []
                for i in range(num_blocks):
                    intersection = block_masks[i] & current_uncovered
                    gains.append(count_bits(intersection))
                
                max_gain = max(gains)
                if max_gain == 0:
                    break
                
                # Temperature for softmax
                temp_idx = min(step, len(schedule) - 1)
                temperature = schedule[temp_idx]
                
                if temperature < 0.01:
                    # Greedy: pick max gain with tie-breaking
                    best_gain = max_gain
                    best_idx = -1
                    best_tie = -1
                    for i, g in enumerate(gains):
                        if g == best_gain:
                            tie = count_bits(block_masks[i] & current_union)
                            if tie > best_tie:
                                best_tie = tie
                                best_idx = i
                else:
                    # Softmax selection
                    candidates = [(i, g) for i, g in enumerate(gains) if g > 0]
                    
                    if not candidates:
                        break
                    
                    # Compute softmax weights
                    scaled_gains = [g / temperature for _, g in candidates]
                    max_scaled = max(scaled_gains)
                    exp_weights = [np.exp(sg - max_scaled) for sg in scaled_gains]
                    total_weight = sum(exp_weights)
                    
                    # Sample
                    r = random.random() * total_weight
                    cumulative = 0
                    best_idx = candidates[-1][0]
                    for (idx, _), w in zip(candidates, exp_weights):
                        cumulative += w
                        if r <= cumulative:
                            best_idx = idx
                            break
                
                chosen.append(best_idx)
                current_uncovered &= ~block_masks[best_idx]
                current_union |= block_masks[best_idx]
                step += 1
            
            # Remove redundancies
            chosen = remove_redundant(chosen, deadline - 2.0)
            
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_solution = chosen[:]
    
    # If no solution found, fall back to pure greedy
    if best_solution is None:
        current_uncovered = (1 << num_t_subsets) - 1
        chosen = []
        current_union = 0
        
        while current_uncovered != 0:
            if time.time() > deadline - 3.0:
                break
            
            best_gain = -1
            best_idx = -1
            best_tie = -1
            
            for i in range(num_blocks):
                intersection = block_masks[i] & current_uncovered
                gain = count_bits(intersection)
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_tie = count_bits(block_masks[i] & current_union)
                elif gain == best_gain and gain > 0:
                    tie = count_bits(block_masks[i] & current_union)
                    if tie > best_tie:
                        best_tie = tie
                        best_idx = i
            
            if best_idx == -1:
                break
            
            chosen.append(best_idx)
            current_uncovered &= ~block_masks[best_idx]
            current_union |= block_masks[best_idx]
        
        best_solution = remove_redundant(chosen, deadline - 2.0)
    
    return [list(all_blocks[idx]) for idx in best_solution]
