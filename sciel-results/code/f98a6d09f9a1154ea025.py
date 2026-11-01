import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)
            
    # Precompute degrees for weighting
    t_subset_degrees = [len(lst) for lst in t_subset_blocks]
    
    # Precompute for fast energy calculation
    block_size_t = np.array([len(block_t_indices[b]) for b in range(num_k_subsets)])

    # 2. Greedy Set Cover (Randomized)
    def greedy_set_cover():
        uncovered = [True] * num_t_subsets
        gain = [len(block_t_indices[b]) for b in range(num_k_subsets)]
        selected_blocks = []
        
        while True:
            max_gain = max(gain)
            if max_gain == 0:
                break
            
            # Random tie-breaking to allow diversity
            candidates = [i for i, g in enumerate(gain) if g == max_gain]
            best_block = random.choice(candidates)
            selected_blocks.append(best_block)
            
            for t_idx in block_t_indices[best_block]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx in t_subset_blocks[t_idx]:
                        gain[b_idx] -= 1
                        
        return selected_blocks

    # 3. Densest Subgraph Heuristic
    def densest_subgraph_init():
        """
        Models the problem as finding a maximum-density set of blocks.
        Uses iterative reweighting to penalize redundant coverage.
        """
        # Start with all blocks, then iteratively remove low-leverage blocks
        active = set(range(num_k_subsets))
        cover_count = [0] * num_t_subsets
        for b_idx in active:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
        
        # Iterative removal of low-leverage blocks
        for _ in range(20):  # Fixed number of iterations for speed
            # Compute leverage: unique coverage / total coverage
            min_leverage = float('inf')
            min_block = None
            
            for b_idx in list(active):
                unique = 0
                total = 0
                for t_idx in block_t_indices[b_idx]:
                    total += 1
                    if cover_count[t_idx] == 1:
                        unique += 1
                if total > 0:
                    leverage = unique / total
                else:
                    leverage = 0
                if leverage < min_leverage:
                    min_leverage = leverage
                    min_block = b_idx
            
            if min_block is None:
                break
                
            # Only remove if it doesn't create uncovered subsets
            can_remove = True
            for t_idx in block_t_indices[min_block]:
                if cover_count[t_idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                active.remove(min_block)
                for t_idx in block_t_indices[min_block]:
                    cover_count[t_idx] -= 1
            else:
                break
        
        return list(active), cover_count

    # 3. Minimalization (Deletion-based)
    def minimize(active_blocks_set, cover_counts):
        changed = True
        while changed:
            changed = False
            items = list(active_blocks_set)
            random.shuffle(items)
            for b_idx in items:
                if b_idx not in active_blocks_set:
                    continue
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_counts[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active_blocks_set.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_counts[t_idx] -= 1
                    changed = True
        return active_blocks_set, cover_counts

    # 4. 2-for-1 Swap: Replace two blocks with one
    def two_for_one_swap(active_blocks_set, cover_counts):
        """
        Try to replace two blocks with one block that covers all their unique coverage.
        Returns True if a swap was made.
        """
        items = list(active_blocks_set)
        if len(items) < 2:
            return False
        
        random.shuffle(items)
        
        # Sample pairs to limit computational cost
        max_pairs = min(50, len(items) * (len(items) - 1) // 2)
        pairs_sampled = 0
        
        for i in range(len(items)):
            if pairs_sampled >= max_pairs:
                break
            b1 = items[i]
            if b1 not in active_blocks_set:
                continue
            
            # Find subsets uniquely covered by b1
            unique_to_b1 = []
            for t_idx in block_t_indices[b1]:
                if cover_counts[t_idx] == 1:
                    unique_to_b1.append(t_idx)
            
            if not unique_to_b1:
                continue
            
            # Try to find a block that covers all unique_to_b1
            for b2 in items[i+1:]:
                if b2 not in active_blocks_set:
                    continue
                if b2 == b1:
                    continue
                
                # Check if b1 and b2 together can be replaced by a single block
                # We need a block that covers all unique_to_b1 AND all unique_to_b2
                unique_to_b2 = []
                for t_idx in block_t_indices[b2]:
                    if cover_counts[t_idx] == 1:
                        unique_to_b2.append(t_idx)
                
                # Find candidate replacement blocks
                # A block must cover all t_idx in unique_to_b1 and unique_to_b2
                # Use the t_subset_blocks index to find candidates
                candidates = None
                for t_idx in unique_to_b1[:3]:  # Limit to first 3 for speed
                    if candidates is None:
                        candidates = set(t_subset_blocks[t_idx])
                    else:
                        candidates &= set(t_subset_blocks[t_idx])
                
                if candidates is None:
                    candidates = set()
                
                for t_idx in unique_to_b2[:3]:
                    if candidates is None:
                        candidates = set(t_subset_blocks[t_idx])
                    else:
                        candidates &= set(t_subset_blocks[t_idx])
                
                if not candidates:
                    continue
                
                # Check each candidate
                for b_new in candidates:
                    if b_new == b1 or b_new == b2:
                        continue
                    
                    # Verify b_new covers all unique subsets of b1 and b2
                    valid = True
                    for t_idx in unique_to_b1:
                        if t_idx not in block_t_indices[b_new]:
                            valid = False
                            break
                    if not valid:
                        continue
                    for t_idx in unique_to_b2:
                        if t_idx not in block_t_indices[b_new]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Perform the swap
                    active_blocks_set.remove(b1)
                    active_blocks_set.remove(b2)
                    active_blocks_set.add(b_new)
                    
                    for t_idx in block_t_indices[b1]:
                        cover_counts[t_idx] -= 1
                    for t_idx in block_t_indices[b2]:
                        cover_counts[t_idx] -= 1
                    for t_idx in block_t_indices[b_new]:
                        cover_counts[t_idx] += 1
                    
                    return True
        
        return False

    # 4. Simulated Annealing Local Search
    def local_search_sa(active_blocks, cover_count, time_limit):
        start_time = time.time()
        
        # Energy: number of blocks + penalty for uncovered subsets
        # We want to minimize this.
        # State: set of active blocks, cover counts.
        
        def get_energy(blocks, counts):
            # Penalty for uncovered subsets is huge if we want to ensure validity
            # But we want to allow temporary invalid states to escape local minima?
            # The task requires a valid covering.
            # Standard SA for covering: 
            # If we allow invalid states, we must ensure we end up valid.
            # Let's stick to valid states but use swaps that maintain validity?
            # Or allow invalid states with a very high penalty, and only accept if it leads to a better valid state later?
            # Given the strict validity check, it's safer to stay in valid states or use a penalty that is so high we rarely accept invalid.
            # However, the prompt suggests "temporarily accept solutions with ... uncovered subsets".
            # Let's implement SA where moves can be Add, Remove, Swap.
            # If a move creates uncovered subsets, the penalty increases energy.
            # If a move reduces block count but creates uncovered, it might be accepted if T is high.
            
            uncovered_count = sum(1 for c in counts if c == 0)
            block_count = len(blocks)
            # Penalty weight for uncovered subsets
            penalty_weight = 1000 
            return block_count + penalty_weight * uncovered_count

        current_blocks = set(active_blocks)
        current_counts = list(cover_count)
        current_energy = get_energy(current_blocks, current_counts)
        
        best_blocks = set(current_blocks)
        best_counts = list(current_counts)
        best_energy = current_energy
        
        # Initial temperature
        T = 50.0
        cooling_rate = 0.995
        
        step_count = 0
        
        while time.time() - start_time < time_limit:
            step_count += 1
            # Generate a move
            move_type = random.random()
            
            if move_type < 0.35:
                # Swap: Remove one, Add one
                if not current_blocks:
                    continue
                B_old = random.choice(list(current_blocks))
                
                # Find candidates for B_new
                # Heuristic: Prefer blocks that cover many of the subsets currently covered by B_old
                # Or just random
                B_new = random.randint(0, num_k_subsets - 1)
                if B_new == B_old:
                    continue
                
                # Check if swap is valid (maintains coverage)
                # Calculate new counts
                # This is O(k) to update counts, but we need to check if any count drops to 0
                # Let's simulate
                temp_counts = current_counts.copy() # Expensive? num_t_subsets is ~ 3876 for v=19,t=4? C(19,4)=3876.
                # Actually, let's do a faster check.
                
                # Subsets covered by B_old but not B_new
                # If any of these have count 1, they become 0.
                # Let's just update a copy. 3876 ints is small.
                
                for t_idx in block_t_indices[B_old]:
                    temp_counts[t_idx] -= 1
                for t_idx in block_t_indices[B_new]:
                    temp_counts[t_idx] += 1
                
                new_blocks = set(current_blocks)
                new_blocks.remove(B_old)
                new_blocks.add(B_new)
                
                new_energy = get_energy(new_blocks, temp_counts)
                
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    # Accept
                    current_blocks = new_blocks
                    current_counts = temp_counts
                    current_energy = new_energy
                    
                    # Minimize after swap
                    current_blocks, current_counts = minimize(current_blocks, current_counts)
                    current_energy = get_energy(current_blocks, current_counts)
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = set(current_blocks)
                        best_counts = list(current_counts)
                        
            elif move_type < 0.5:
                # 2-for-1 Swap: Replace two blocks with one
                if len(current_blocks) >= 2:
                    if two_for_one_swap(current_blocks, current_counts):
                        current_energy = get_energy(current_blocks, current_counts)
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_blocks = set(current_blocks)
                            best_counts = list(current_counts)
                            
            elif move_type < 0.75:
                # Add a block
                B_new = random.randint(0, num_k_subsets - 1)
                if B_new in current_blocks:
                    continue
                
                new_blocks = set(current_blocks)
                new_blocks.add(B_new)
                
                temp_counts = current_counts.copy()
                for t_idx in block_t_indices[B_new]:
                    temp_counts[t_idx] += 1
                
                new_energy = get_energy(new_blocks, temp_counts)
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    current_blocks = new_blocks
                    current_counts = temp_counts
                    current_energy = new_energy
                    
                    # Minimize
                    current_blocks, current_counts = minimize(current_blocks, current_counts)
                    current_energy = get_energy(current_blocks, current_counts)
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = set(current_blocks)
                        best_counts = list(current_counts)
                        
            else:
                # Remove a block
                if not current_blocks:
                    continue
                B_old = random.choice(list(current_blocks))
                
                new_blocks = set(current_blocks)
                new_blocks.remove(B_old)
                
                temp_counts = current_counts.copy()
                for t_idx in block_t_indices[B_old]:
                    temp_counts[t_idx] -= 1
                
                new_energy = get_energy(new_blocks, temp_counts)
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    current_blocks = new_blocks
                    current_counts = temp_counts
                    current_energy = new_energy
                    
                    # Minimize
                    current_blocks, current_counts = minimize(current_blocks, current_counts)
                    current_energy = get_energy(current_blocks, current_counts)
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = set(current_blocks)
                        best_counts = list(current_counts)

            T *= cooling_rate
            if T < 0.01:
                break

        return best_blocks, best_counts

    # Main Loop
    best_blocks = None
    best_size = float('inf')
    total_time_limit = 115.0 # Safety margin for 150s limit
    start_total = time.time()
    
    # We will run multiple iterations of Greedy + SA
    # Each iteration should take a fraction of the time
    # Alternate between greedy and densest subgraph initialization
    
    iteration = 0
    
    while time.time() - start_total < total_time_limit:
        iteration += 1
        
        # Choose initialization strategy
        if iteration % 3 == 0:
            # Use densest subgraph heuristic
            selected, cover_count = densest_subgraph_init()
            active_blocks = set(selected)
        else:
            # Use greedy
            selected = greedy_set_cover()
            active_blocks = set(selected)
            cover_count = [0] * num_t_subsets
            for b_idx in active_blocks:
                for t_idx in block_t_indices[b_idx]:
                    cover_count[t_idx] += 1
        
        # 2. Minimize
        active_blocks, cover_count = minimize(active_blocks, cover_count)
        
        # 3. SA Local Search
        # Allocate remaining time / 5 for this iteration's SA
        remaining = total_time_limit - (time.time() - start_total)
        sa_time = max(0.5, remaining / 5.0)
        
        active_blocks, cover_count = local_search_sa(active_blocks, cover_count, sa_time)
        
        if len(active_blocks) < best_size:
            best_size = len(active_blocks)
            best_blocks = list(active_blocks)
            
            # Early stop if we hit the theoretical lower bound? 
            # Lower bound is 131. If we hit 131, we can stop.
            if best_size <= 131:
                break

    if best_blocks is None:
        # Fallback
        selected = greedy_set_cover()
        active_blocks = set(selected)
        cover_count = [0] * num_t_subsets
        for b_idx in active_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
        active_blocks, cover_count = minimize(active_blocks, cover_count)
        best_blocks = list(active_blocks)

    return [list(all_k_subsets[b_idx]) for b_idx in best_blocks]
