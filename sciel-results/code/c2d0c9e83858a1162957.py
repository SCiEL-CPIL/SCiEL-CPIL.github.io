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
        current_blocks = list(blocks_indices)
        
        if not current_blocks:
            return current_blocks
            
        # Build cover_count using numpy
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_subsets_np[b_idx]] += 1
        
        # Repeatedly remove redundant blocks using set-based approach
        changed = True
        while changed:
            changed = False
            # A block is redundant if all its subsets have cover_count > 1
            redundant_mask = np.ones(len(current_blocks), dtype=bool)
            for i, b_idx in enumerate(current_blocks):
                if not np.all(cover_count[block_subsets_np[b_idx]] > 1):
                    redundant_mask[i] = False
            
            if not np.any(redundant_mask):
                break
            
            # Remove all redundant blocks at once
            for i in np.where(redundant_mask)[0][::-1]:
                b_idx_to_remove = current_blocks[i]
                cover_count[block_subsets_np[b_idx_to_remove]] -= 1
                current_blocks.pop(i)
            changed = True
        
        return current_blocks

    def simulated_annealing(blocks_indices, random_seed=None, max_time=20.0):
        """
        Simulated annealing with smooth energy landscape.
        Uses soft coverage penalty via exp(-alpha * cover_count) as surrogate.
        Optimized with set membership and numpy vectorization.
        """
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        start_time = time.time()
        
        current_blocks = list(blocks_indices)
        current_set = set(current_blocks)
        
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_subsets_np[b_idx]] += 1
        
        alpha = 2.0
        def compute_energy(cover_cnt, num_sel):
            # Use log-sum-exp style: sum of exp(-alpha * c) is a smooth surrogate
            # for the count of uncovered subsets (where c=0).
            soft_uncovered = np.sum(np.exp(-alpha * cover_cnt))
            return soft_uncovered + 0.01 * num_sel
        
        current_energy = compute_energy(cover_count, len(current_blocks))
        
        best_blocks = list(current_blocks)
        best_energy = current_energy
        
        T = 5.0
        cooling_rate = 0.998
        min_T = 0.05
        max_steps = 100000
        
        step = 0
        while T > min_T and step < max_steps:
            if time.time() - start_time > max_time:
                break
                
            move_type = rng.random()
            
            new_blocks = list(current_blocks)
            new_cover_count = cover_count.copy()
            new_set = set(current_set)
            
            if move_type < 0.35 and len(current_blocks) > 1:
                idx = rng.randint(0, len(new_blocks) - 1)
                b_remove = new_blocks[idx]
                new_cover_count[block_subsets_np[b_remove]] -= 1
                new_blocks.pop(idx)
                new_set.remove(b_remove)
                
            elif move_type < 0.65:
                new_b = rng.randint(0, num_blocks - 1)
                if new_b not in new_set:
                    new_cover_count[block_subsets_np[new_b]] += 1
                    new_blocks.append(new_b)
                    new_set.add(new_b)
                else:
                    step += 1
                    T *= cooling_rate
                    continue
                    
            else:
                if len(current_blocks) == 0:
                    step += 1
                    T *= cooling_rate
                    continue
                idx = rng.randint(0, len(new_blocks) - 1)
                b_old = new_blocks[idx]
                new_b = rng.randint(0, num_blocks - 1)
                if new_b != b_old and new_b not in new_set:
                    new_cover_count[block_subsets_np[b_old]] -= 1
                    new_cover_count[block_subsets_np[new_b]] += 1
                    new_blocks[idx] = new_b
                    new_set.remove(b_old)
                    new_set.add(new_b)
                else:
                    step += 1
                    T *= cooling_rate
                    continue
            
            new_energy = compute_energy(new_cover_count, len(new_blocks))
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < np.exp(-max(0.0, delta_e) / T):
                current_blocks = new_blocks
                current_set = new_set
                cover_count = new_cover_count
                current_energy = new_energy
                
                if np.all(cover_count > 0) and len(current_blocks) < len(best_blocks):
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
            cover_count[block_subsets_np[b_idx]] += 1
        
        best = list(current_blocks)
        best_len = len(current_blocks)
        current_set = set(current_blocks)
        
        for _ in range(max_iters):
            if not current_blocks:
                break
            
            # Try to find a block that can be swapped for a better one
            swap_idx = rng.randint(0, len(current_blocks) - 1)
            b_swap = current_blocks[swap_idx]
            
            # Try random replacements
            for _ in range(20):
                new_b = rng.randint(0, num_blocks - 1)
                if new_b == b_swap or new_b in current_set:
                    continue
                    
                # Check if swapping maintains coverage
                test_cover = cover_count.copy()
                test_cover[block_subsets_np[b_swap]] -= 1
                test_cover[block_subsets_np[new_b]] += 1
                
                if np.all(test_cover > 0):
                    # Swap is valid
                    current_blocks[swap_idx] = new_b
                    current_set.remove(b_swap)
                    current_set.add(new_b)
                    cover_count = test_cover
                    
                    # Try to remove redundant blocks using set-based approach
                    changed = True
                    while changed:
                        changed = False
                        redundant_mask = np.ones(len(current_blocks), dtype=bool)
                        for i, b_idx in enumerate(current_blocks):
                            if not np.all(cover_count[block_subsets_np[b_idx]] > 1):
                                redundant_mask[i] = False
                        if not np.any(redundant_mask):
                            break
                        for i in np.where(redundant_mask)[0][::-1]:
                            b_to_remove = current_blocks[i]
                            cover_count[block_subsets_np[b_to_remove]] -= 1
                            current_blocks.pop(i)
                            current_set.remove(b_to_remove)
                        changed = True
                        
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best = list(current_blocks)
                    break
            
        return best

    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 112.0
    
    trial = 0
    while time.time() < deadline:
        # Strategy 1: Full Greedy + Minimize
        greedy_res = run_greedy_full(trial)
        min_res = minimize(greedy_res, trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
        
        # Strategy 2: Local Search on Best
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 15:
            ls_res = local_search_swap(best_blocks, trial + 5000, max_iters=50)
            min_res_ls = minimize(ls_res, trial + 6000)
            
            if len(min_res_ls) < best_len:
                best_len = len(min_res_ls)
                best_blocks = min_res_ls
        
        # Strategy 3: Simulated Annealing on Best
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 25:
            sa_res = simulated_annealing(best_blocks, trial + 7000, max_time=18.0)
            min_res_sa = minimize(sa_res, trial + 8000)
            
            if len(min_res_sa) < best_len:
                best_len = len(min_res_sa)
                best_blocks = min_res_sa
        
        # Strategy 4: Perturb Best + Repair + Minimize
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 15:
            perturbed = list(best_blocks)
            num_remove = min(4, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx not in blocks_to_remove:
                        cover_count_temp[block_subsets_np[b_idx]] += 1
                        
                uncovered_mask = (cover_count_temp == 0).astype(bool)
                
                repair_res = run_greedy_partial(uncovered_mask, trial + 1000)
                
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                min_res2 = minimize(new_set, trial + 2000)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2

        trial += 1
    
    # Final high-precision polish pass: aggressive minimize with multiple passes
    if best_blocks is not None:
        # Run minimize multiple times to ensure full convergence
        for _ in range(3):
            polished = minimize(best_blocks, trial + 9999)
            if len(polished) < len(best_blocks):
                best_blocks = polished
            else:
                break
        # One more local search on the polished result
        if time.time() < deadline - 5:
            final_ls = local_search_swap(best_blocks, trial + 11111, max_iters=30)
            final_min = minimize(final_ls, trial + 12222)
            if len(final_min) < len(best_blocks):
                best_blocks = final_min
    
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = minimize(greedy_res, 0)
        
    return [list(all_blocks[i]) for i in best_blocks]
