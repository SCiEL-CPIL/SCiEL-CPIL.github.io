import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    
    points = list(range(v))
    
    # 1. Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    n_t = len(t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(t_subsets)}
    
    # 2. Precompute all k-blocks and their coverage masks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # block_masks[i] is an integer where bit j is 1 if block i covers t_subset j
    block_masks = [0] * n_blocks
    block_cover_indices = [[] for _ in range(n_blocks)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            block_cover_indices[i].append(idx)
        block_masks[i] = mask
    
    # Precompute the list of set bits for each mask to speed up iteration
    mask_bits = []
    for i in range(n_blocks):
        m = block_masks[i]
        bits = []
        while m:
            lsb = m & -m
            bits.append(lsb.bit_length() - 1)
            m ^= lsb
        mask_bits.append(bits)
    
    def time_left():
        return deadline - time.time()
    
    # 3. Greedy initialization with randomization
    def greedy_init(seed_val):
        random.seed(seed_val)
        uncovered_mask = (1 << n_t) - 1
        selected = []
        
        while uncovered_mask and time_left() > 5:
            best_block_idx = -1
            best_gain = 0
            candidates = []
            
            for i in range(n_blocks):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    candidates = [(i, gain)]
                elif gain == best_gain and gain > 0:
                    candidates.append((i, gain))
            
            if best_block_idx == -1:
                break
            
            # Randomization: with probability 0.3, pick a random candidate from top 10%
            if candidates and random.random() < 0.3:
                top_k = max(1, len(candidates) // 10)
                chosen_idx, _ = random.choice(candidates[:top_k])
            else:
                chosen_idx = best_block_idx
                
            selected.append(chosen_idx)
            uncovered_mask &= ~block_masks[chosen_idx]
        
        return selected, uncovered_mask
    
    # 4. Simulated Annealing
    def simulated_annealing(initial_selected, initial_uncovered, seed_val):
        random.seed(seed_val)
        
        # State
        selected_set = set(initial_selected)
        counts = [0] * n_t
        uncovered_mask = initial_uncovered
        
        # Initialize counts
        for b_idx in selected_set:
            for idx in mask_bits[b_idx]:
                counts[idx] += 1
        
        # Cost function: number of blocks + penalty for uncovered
        # We want to minimize blocks, but must cover all.
        # If uncovered_mask != 0, add large penalty.
        
        current_cost = len(selected_set) + 1000 * uncovered_mask.bit_count()
        best_selected = list(selected_set)
        best_cost = current_cost
        
        # SA Parameters
        T = 50.0  # Initial temperature
        T_min = 0.1
        cooling_rate = 0.9995
        max_iter = 0
        
        # Estimate max iterations based on time
        # Each iteration is roughly O(k*t) for updates, which is small.
        # Let's just run until time limit or T < T_min
        
        while T > T_min and time_left() > 2:
            # Generate move
            move_type = random.random()
            
            if move_type < 0.4:
                # Add a random block
                new_block = random.randint(0, n_blocks - 1)
                if new_block in selected_set:
                    continue
                
                # Calculate delta
                delta_len = 1
                delta_uncovered = 0
                for idx in mask_bits[new_block]:
                    if counts[idx] == 0:
                        delta_uncovered -= 1
                
                delta_cost = delta_len + 1000 * delta_uncovered
                
                if random.random() < np.exp(-delta_cost / T):
                    # Accept
                    selected_set.add(new_block)
                    for idx in mask_bits[new_block]:
                        if counts[idx] == 0:
                            uncovered_mask &= ~(1 << idx)
                        counts[idx] += 1
                    current_cost += delta_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = list(selected_set)
                        
            elif move_type < 0.8:
                # Remove a random block
                if not selected_set:
                    continue
                old_block = random.choice(list(selected_set))
                
                # Calculate delta
                delta_len = -1
                delta_uncovered = 0
                for idx in mask_bits[old_block]:
                    if counts[idx] == 1:
                        delta_uncovered += 1
                
                delta_cost = delta_len + 1000 * delta_uncovered
                
                if random.random() < np.exp(-delta_cost / T):
                    # Accept
                    selected_set.remove(old_block)
                    for idx in mask_bits[old_block]:
                        counts[idx] -= 1
                        if counts[idx] == 0:
                            uncovered_mask |= (1 << idx)
                    current_cost += delta_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = list(selected_set)
                        
            else:
                # Swap: remove one, add one
                if not selected_set:
                    continue
                old_block = random.choice(list(selected_set))
                new_block = random.randint(0, n_blocks - 1)
                if new_block in selected_set:
                    continue
                
                # Calculate delta
                delta_len = 0 # -1 + 1
                delta_uncovered = 0
                # Remove old
                for idx in mask_bits[old_block]:
                    if counts[idx] == 1:
                        delta_uncovered += 1
                # Add new
                for idx in mask_bits[new_block]:
                    if counts[idx] == 0:
                        delta_uncovered -= 1
                    elif counts[idx] == 1:
                        # It was uncovered by old, but covered by new?
                        # Wait, if counts[idx] == 1, it means old_block was the only one covering it.
                        # So removing old makes it uncovered (+1).
                        # Adding new makes it covered (-1).
                        # Net 0.
                        pass
                
                # Actually, let's just recalculate uncovered change carefully.
                # Uncovered set changes only if a bit goes 0->1 or 1->0 in counts.
                # Bit i is in uncovered_mask if counts[i] == 0.
                
                # Let's just do the move and check cost difference by bit_count?
                # No, that's slow.
                # Let's stick to the delta logic.
                
                # Remove old_block:
                # For each idx in old_block:
                #   if counts[idx] == 1: it becomes 0, so uncovered +1
                # Add new_block:
                # For each idx in new_block:
                #   if counts[idx] == 0 (after removal): it becomes 1, so uncovered -1
                #   if counts[idx] > 0: no change in uncovered status
                
                # This is tricky to do in one pass without state.
                # Let's just do the move, update state, and if rejected, revert.
                # Since k=8, t=5, each block covers C(8,5)=56 subsets.
                # Updating 56 counts is very fast.
                
                # Perform move
                selected_set.remove(old_block)
                for idx in mask_bits[old_block]:
                    counts[idx] -= 1
                    if counts[idx] == 0:
                        uncovered_mask |= (1 << idx)
                
                selected_set.add(new_block)
                for idx in mask_bits[new_block]:
                    if counts[idx] == 0:
                        uncovered_mask &= ~(1 << idx)
                    counts[idx] += 1
                
                new_cost = len(selected_set) + 1000 * uncovered_mask.bit_count()
                delta_cost = new_cost - current_cost
                
                if random.random() < np.exp(-delta_cost / T):
                    # Accept
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = list(selected_set)
                else:
                    # Revert
                    selected_set.remove(new_block)
                    for idx in mask_bits[new_block]:
                        counts[idx] -= 1
                        if counts[idx] == 0:
                            uncovered_mask |= (1 << idx)
                    
                    selected_set.add(old_block)
                    for idx in mask_bits[old_block]:
                        if counts[idx] == 0:
                            uncovered_mask &= ~(1 << idx)
                        counts[idx] += 1
            
            T *= cooling_rate
        
        return best_selected, best_cost
    
    # 5. Local Search: Redundancy removal
    def remove_redundant(selected_indices):
        if not selected_indices:
            return selected_indices
        
        counts = [0] * n_t
        for b_idx in selected_indices:
            for idx in mask_bits[b_idx]:
                counts[idx] += 1
        
        new_selected = []
        for b_idx in selected_indices:
            redundant = True
            for idx in mask_bits[b_idx]:
                if counts[idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                for idx in mask_bits[b_idx]:
                    counts[idx] -= 1
            else:
                new_selected.append(b_idx)
        
        return new_selected
    
    best_solution = None
    best_size = float('inf')
    
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time_left() < 15:
            break
        
        seed_val = 42 + restart
        
        # Greedy Init
        selected, uncovered = greedy_init(seed_val)
        
        if not selected:
            continue
        
        # Remove redundant blocks first
        selected = remove_redundant(selected)
        
        # Recalculate uncovered
        counts = [0] * n_t
        for b_idx in selected:
            for idx in mask_bits[b_idx]:
                counts[idx] += 1
        
        uncovered_mask = 0
        for i in range(n_t):
            if counts[i] == 0:
                uncovered_mask |= (1 << i)
        
        # Simulated Annealing
        sa_selected, sa_cost = simulated_annealing(selected, uncovered_mask, seed_val)
        
        # Remove redundant blocks from SA result
        sa_selected = remove_redundant(sa_selected)
        
        # Verify coverage
        if sa_selected:
            final_counts = [0] * n_t
            for b_idx in sa_selected:
                for idx in mask_bits[b_idx]:
                    final_counts[idx] += 1
            
            is_valid = all(c > 0 for c in final_counts)
            
            if is_valid:
                if len(sa_selected) < best_size:
                    best_size = len(sa_selected)
                    best_solution = sa_selected
    
    # Fallback if no valid solution found
    if best_solution is None:
        selected, uncovered = greedy_init(42)
        best_solution = selected
    
    # Convert indices back to blocks
    result_blocks = [list(all_blocks[i]) for i in best_solution]
    
    return result_blocks
