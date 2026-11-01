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
        
        while time.time() - start_time < time_limit:
            # Generate a move
            move_type = random.random()
            
            if move_type < 0.4:
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
                        
            elif move_type < 0.7:
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
    
    while time.time() - start_total < total_time_limit:
        # 1. Greedy
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
