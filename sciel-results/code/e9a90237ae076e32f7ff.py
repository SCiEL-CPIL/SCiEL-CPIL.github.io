import numpy as np
import itertools
import time
import random
import sys

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block_to_subsets and subset_to_blocks
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
    
    block_sizes = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
    
    # Convert to numpy arrays for faster operations
    # For each block, store its subsets as a numpy array
    block_subsets_np = [np.array(subs, dtype=np.int32) for subs in block_to_subsets]
    # For each subset, store its blocks as a numpy array
    subset_blocks_np = [np.array(blocks, dtype=np.int32) for blocks in subset_to_blocks]
    
    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = block_sizes.copy()
        
        selected_blocks = []
        
        while True:
            if not np.any(uncovered):
                break
                
            # Add noise to break ties and explore
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subsets_np[best_idx]
            # Find newly covered subsets
            newly_covered_mask = uncovered[b_subs]
            newly_covered = b_subs[newly_covered_mask]
            
            if len(newly_covered) == 0:
                continue
                
            for s in newly_covered:
                uncovered[s] = False
                for b in subset_blocks_np[s]:
                    gains[b] -= 1
                    
        return selected_blocks

    def run_greedy_partial(uncovered_mask, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return []
            
        gains = np.zeros(num_blocks, dtype=np.int32)
        for s_idx in uncovered_indices:
            for b_idx in subset_blocks_np[s_idx]:
                gains[b_idx] += 1
                
        selected_blocks = []
        current_uncovered = uncovered_mask.copy()
        
        while True:
            if not np.any(current_uncovered):
                break
                
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subsets_np[best_idx]
            newly_covered_mask = current_uncovered[b_subs]
            newly_covered = b_subs[newly_covered_mask]
            
            if len(newly_covered) == 0:
                continue
                
            for s in newly_covered:
                current_uncovered[s] = False
                for b in subset_blocks_np[s]:
                    gains[b] -= 1
                    
        return selected_blocks

    def minimize(blocks_indices, random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        
        if not current_blocks:
            return current_blocks
            
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_subsets_np[b_idx]:
                cover_count[s_idx] += 1
                
        # Repeatedly remove redundant blocks
        changed = True
        while changed:
            changed = False
            redundant_blocks = []
            
            for i, b_idx in enumerate(current_blocks):
                is_redundant = True
                for s_idx in block_subsets_np[b_idx]:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    redundant_blocks.append(i)
                    
            if not redundant_blocks:
                break
                
            # Remove all redundant blocks at once for efficiency
            for remove_pos in sorted(redundant_blocks, reverse=True):
                b_idx_to_remove = current_blocks[remove_pos]
                
                for s_idx in block_subsets_np[b_idx_to_remove]:
                    cover_count[s_idx] -= 1
                    
                current_blocks.pop(remove_pos)
                changed = True
            
        return current_blocks

    def simulated_annealing(blocks_indices, random_seed=None, max_time=20.0):
        """
        Simulated annealing on the block selection.
        State: set of blocks (represented as a list of block indices).
        Moves: 
          1. Remove a random block
          2. Add a random block
          3. Swap a random block with another random block
        Energy: number of uncovered subsets (should be 0 for valid solution) + penalty * number of blocks
        We want to minimize energy. Valid solutions have 0 uncovered.
        """
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        start_time = time.time()
        
        # Start with the given blocks
        current_blocks = list(blocks_indices)
        
        # Compute initial coverage
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_subsets_np[b_idx]:
                cover_count[s_idx] += 1
        
        current_uncovered = np.sum(cover_count == 0)
        current_energy = current_uncovered * 1000 + len(current_blocks)
        
        best_blocks = list(current_blocks)
        best_energy = current_energy
        
        # Simulated annealing parameters
        T = 10.0  # initial temperature
        cooling_rate = 0.995
        min_T = 0.1
        max_steps = 50000
        
        step = 0
        while T > min_T and step < max_steps:
            if time.time() - start_time > max_time:
                break
                
            # Choose a random move
            move_type = rng.choice(['remove', 'add', 'swap'])
            
            new_blocks = list(current_blocks)
            new_cover_count = cover_count.copy()
            
            if move_type == 'remove' and len(current_blocks) > 1:
                # Remove a random block
                idx = rng.randint(0, len(new_blocks) - 1)
                b_remove = new_blocks[idx]
                for s_idx in block_subsets_np[b_remove]:
                    new_cover_count[s_idx] -= 1
                new_blocks.pop(idx)
                
            elif move_type == 'add':
                # Add a random block
                new_b = rng.randint(0, num_blocks - 1)
                if new_b not in new_blocks:
                    for s_idx in block_subsets_np[new_b]:
                        new_cover_count[s_idx] += 1
                    new_blocks.append(new_b)
                else:
                    step += 1
                    T *= cooling_rate
                    continue
                    
            elif move_type == 'swap' and len(current_blocks) > 0:
                # Swap a random block with another
                idx = rng.randint(0, len(new_blocks) - 1)
                b_old = new_blocks[idx]
                new_b = rng.randint(0, num_blocks - 1)
                if new_b != b_old and new_b not in new_blocks:
                    for s_idx in block_subsets_np[b_old]:
                        new_cover_count[s_idx] -= 1
                    for s_idx in block_subsets_np[new_b]:
                        new_cover_count[s_idx] += 1
                    new_blocks[idx] = new_b
                else:
                    step += 1
                    T *= cooling_rate
                    continue
            
            new_uncovered = np.sum(new_cover_count == 0)
            new_energy = new_uncovered * 1000 + len(new_blocks)
            
            delta_e = new_energy - current_energy
            
            # Acceptance probability
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                current_blocks = new_blocks
                cover_count = new_cover_count
                current_uncovered = new_uncovered
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_blocks = list(current_blocks)
            
            step += 1
            T *= cooling_rate
        
        return best_blocks

    def local_search_swap(blocks_indices, random_seed=None, max_iters=50):
        """Try to swap blocks to reduce the total number."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        
        if len(current_blocks) < 2:
            return current_blocks
            
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_subsets_np[b_idx]:
                cover_count[s_idx] += 1
        
        best = list(current_blocks)
        best_len = len(current_blocks)
        
        for _ in range(max_iters):
            if not current_blocks:
                break
            
            # Try to find a block that can be swapped for a better one
            swap_idx = rng.randint(0, len(current_blocks) - 1)
            b_swap = current_blocks[swap_idx]
            
            # Try random replacements
            for _ in range(15):
                new_b = rng.randint(0, num_blocks - 1)
                if new_b == b_swap or new_b in current_blocks:
                    continue
                    
                # Check if swapping maintains coverage
                test_cover = cover_count.copy()
                for s_idx in block_subsets_np[b_swap]:
                    test_cover[s_idx] -= 1
                for s_idx in block_subsets_np[new_b]:
                    test_cover[s_idx] += 1
                    
                if np.all(test_cover > 0):
                    # Swap is valid
                    current_blocks[swap_idx] = new_b
                    cover_count = test_cover
                    
                    # Try to remove redundant blocks
                    redundant = []
                    for i, b_idx in enumerate(current_blocks):
                        is_redundant = True
                        for s_idx in block_subsets_np[b_idx]:
                            if cover_count[s_idx] == 1:
                                is_redundant = False
                                break
                        if is_redundant:
                            redundant.append(i)
                    for remove_pos in sorted(redundant, reverse=True):
                        b_to_remove = current_blocks[remove_pos]
                        for s_idx in block_subsets_np[b_to_remove]:
                            cover_count[s_idx] -= 1
                        current_blocks.pop(remove_pos)
                        
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best = list(current_blocks)
                    break
            
        return best

    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    trial = 0
    while time.time() < deadline:
        # Strategy 1: Full Greedy + Minimize
        greedy_res = run_greedy_full(trial)
        min_res = minimize(greedy_res, trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
        
        # Strategy 2: Local Search on Best
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 10:
            ls_res = local_search_swap(best_blocks, trial + 5000, max_iters=30)
            min_res_ls = minimize(ls_res, trial + 6000)
            
            if len(min_res_ls) < best_len:
                best_len = len(min_res_ls)
                best_blocks = min_res_ls
        
        # Strategy 3: Simulated Annealing on Best
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 15:
            sa_res = simulated_annealing(best_blocks, trial + 7000, max_time=15.0)
            min_res_sa = minimize(sa_res, trial + 8000)
            
            if len(min_res_sa) < best_len:
                best_len = len(min_res_sa)
                best_blocks = min_res_sa
        
        # Strategy 4: Perturb Best + Repair + Minimize
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 10:
            perturbed = list(best_blocks)
            num_remove = min(3, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    for s_idx in block_subsets_np[b_idx]:
                        cover_count_temp[s_idx] += 1
                        
                uncovered_mask = (cover_count_temp == 0).astype(bool)
                
                repair_res = run_greedy_partial(uncovered_mask, trial + 1000)
                
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                min_res2 = minimize(new_set, trial + 2000)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2

        trial += 1
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = minimize(greedy_res, 0)
        
    return [list(all_blocks[i]) for i in best_blocks]
