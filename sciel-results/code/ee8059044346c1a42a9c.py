import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}

    # Build block_masks as list of Python big ints for local search
    block_masks = [0] * num_blocks
    block_subsets = [[] for _ in range(num_blocks)]
    
    # Build cover_matrix for vectorized greedy
    # cover_matrix[i, j] = 1 if block i covers t-subset j
    # Using uint8 to save memory, 50388 * 3876 * 1 byte = ~195MB. 
    # If memory is a concern, we could use bitsets, but uint8 allows fast mat-vec.
    # Let's check if we can do it without 195MB. 
    # Alternative: Store block_subsets (list of lists of indices). 
    # Gain calculation in Python: sum(1 for idx in block_subsets[i] if uncovered[idx])
    # This is O(kC(k,t)) per block. 35 ops. 50k * 35 = 1.7M ops per step. 
    # NumPy mat-vec: 50k * 3876 multiplies/adds. ~200M ops. 
    # NumPy is C-optimized, 200M ops is ~0.1-0.5s? 
    # Python loop 1.7M ops is ~0.1-0.2s? 
    # They might be comparable. But NumPy is likely faster due to vectorization.
    # However, 195MB allocation might be slow or fail.
    
    # Let's try a middle ground: 
    # Use the Python loop for gain but optimize it.
    # Actually, the previous program used big-int bit_count. 
    # `gain = (block_masks[i] & uncovered_bigint).bit_count()`
    # This is very fast. The bottleneck was likely the Python loop overhead of 50k iterations.
    # Can we vectorize the big-int approach? No.
    
    # Let's stick to the NumPy matrix approach but be careful with memory.
    # If 195MB is too much, we can use a sparse representation or chunked processing.
    # But 195MB is usually fine.
    
    # Let's build the matrix efficiently.
    cover_matrix = np.zeros((num_blocks, num_t_subsets), dtype=np.uint8)
    
    # Iterate over t-subsets to populate the matrix
    # For each t-subset, find all blocks that contain it.
    # A block contains t-subset if it is a superset.
    # We can generate these blocks by taking the t-subset and adding k-t elements from the rest.
    
    for t_idx, t_sub in enumerate(all_t_subsets):
        t_set = set(t_sub)
        remaining_points = [p for p in range(v) if p not in t_set]
        
        # Generate all combinations of k-t points from remaining_points
        for add_combo in itertools.combinations(remaining_points, k - t):
            full_block = tuple(sorted(t_sub + add_combo))
            if full_block in block_to_idx:
                b_idx = block_to_idx[full_block]
                cover_matrix[b_idx, t_idx] = 1
                # Also update block_subsets and mask for local search
                block_subsets[b_idx].append(t_idx)
                block_masks[b_idx] |= (1 << t_idx)

    # Inverted index for local search
    subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for i in range(num_blocks):
        for idx in block_subsets[i]:
            subset_to_blocks[idx].append(i)
    
    subset_to_blocks_set = [set(lst) for lst in subset_to_blocks]

    def greedy_construct(randomize=False, seed=None, temperature=1.0):
        """
        Greedy set cover with vectorized gain calculation.
        """
        uncovered_vec = np.ones(num_t_subsets, dtype=np.uint8)
        used_blocks = np.zeros(num_blocks, dtype=bool)
        chosen = []
        
        # For local search compatibility, we also track the big-int uncovered mask
        # But for greedy, we use the vector.
        
        while np.any(uncovered_vec):
            if time.time() > deadline - 5.0:
                break
            
            # Calculate gains: dot product of cover_matrix and uncovered_vec
            # cover_matrix is (num_blocks, num_t_subsets)
            # uncovered_vec is (num_t_subsets,)
            gains = cover_matrix @ uncovered_vec
            
            # Mask out used blocks
            gains[used_blocks] = -1
            
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
                
            if randomize:
                # Softmax selection among top candidates
                # Find candidates with gain close to max
                # Threshold: max_gain - 2
                threshold = max(0, max_gain - 2)
                candidate_indices = np.where(gains >= threshold)[0]
                
                if len(candidate_indices) == 0:
                    break
                    
                # Get gains for candidates
                cand_gains = gains[candidate_indices]
                
                # Softmax
                # Normalize to avoid overflow
                max_cand_gain = np.max(cand_gains)
                exps = np.exp((cand_gains - max_cand_gain) / temperature)
                exps /= np.sum(exps)
                
                r = random.random()
                cumsum = np.cumsum(exps)
                best_idx = candidate_indices[np.searchsorted(cumsum, r)]
            else:
                best_idx = np.argmax(gains)
            
            chosen.append(best_idx)
            used_blocks[best_idx] = True
            # Update uncovered: uncovered = uncovered & ~cover_matrix[best_idx]
            uncovered_vec &= ~cover_matrix[best_idx]
            
        return chosen

    def prune_solution(chosen_indices):
        """
        Removes redundant blocks.
        """
        if not chosen_indices:
            return []
        
        counts = [0] * num_t_subsets
        for idx in chosen_indices:
            for s_idx in block_subsets[idx]:
                counts[s_idx] += 1
        
        changed = True
        current_list = list(chosen_indices)
        
        while changed:
            changed = False
            new_list = []
            for idx in current_list:
                essential = False
                for s_idx in block_subsets[idx]:
                    if counts[s_idx] == 1:
                        essential = True
                        break
                if not essential:
                    for s_idx in block_subsets[idx]:
                        counts[s_idx] -= 1
                    changed = True
                else:
                    new_list.append(idx)
            current_list = new_list
            
        return current_list

    def local_search(chosen_indices, time_limit=5.0):
        """
        1-Swap and 2-Swap Local Search.
        """
        current_solution = list(chosen_indices)
        start_ls = time.time()
        
        improved = True
        max_iterations = 50
        iteration = 0
        
        while improved and time.time() - start_ls < time_limit and iteration < max_iterations:
            improved = False
            iteration += 1
            
            pruned = prune_solution(current_solution)
            if len(pruned) < len(current_solution):
                current_solution = pruned
                improved = True
                continue
            
            counts = [0] * num_t_subsets
            for idx in current_solution:
                for s_idx in block_subsets[idx]:
                    counts[s_idx] += 1
            
            # 1. Try 1-Swap
            best_swap_found = False
            for i, idx in enumerate(current_solution):
                essential_subs = [s for s in block_subsets[idx] if counts[s] == 1]
                
                if not essential_subs:
                    for s in block_subsets[idx]:
                        counts[s] -= 1
                    current_solution.pop(i)
                    improved = True
                    best_swap_found = True
                    break
                
                # Find candidates
                if len(essential_subs) == 1:
                    pool = subset_to_blocks[essential_subs[0]]
                else:
                    pool = list(subset_to_blocks_set[essential_subs[0]])
                    for s in essential_subs[1:]:
                        if len(pool) > len(subset_to_blocks[s]):
                            pool = [b for b in subset_to_blocks[s] if b in subset_to_blocks_set[essential_subs[0]]]
                        else:
                            pool = [b for b in pool if b in subset_to_blocks_set[s]]
                
                if not pool:
                    continue
                
                MAX_POOL_SIZE = 50
                if len(pool) > MAX_POOL_SIZE:
                    pool = random.sample(pool, MAX_POOL_SIZE)
                
                best_cand = -1
                best_cand_score = -1
                
                for cand_idx in pool:
                    if cand_idx == idx or cand_idx in current_solution:
                        continue
                    
                    cand_mask = block_masks[cand_idx]
                    covers_all = True
                    for s in essential_subs:
                        if not (cand_mask & (1 << s)):
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    redundancy = 0
                    for s in block_subsets[cand_idx]:
                        new_count = counts[s] - (1 if s in block_subsets[idx] else 0)
                        if new_count >= 2:
                            redundancy += 1
                    
                    if redundancy > best_cand_score:
                        best_cand_score = redundancy
                        best_cand = cand_idx
                
                if best_cand != -1 and best_cand_score > 0:
                    test_sol = list(current_solution)
                    test_sol[i] = best_cand
                    pruned_test = prune_solution(test_sol)
                    
                    if len(pruned_test) < len(current_solution):
                        current_solution = pruned_test
                        improved = True
                        best_swap_found = True
                        break
                    elif len(pruned_test) == len(current_solution) and random.random() < 0.1:
                        current_solution = pruned_test
                        improved = True
                        best_swap_found = True
                        break
            
            if best_swap_found:
                continue
            
            # 2. Try 2-Swap (Remove 2, Add 2) if time allows
            if time.time() - start_ls < time_limit - 1.0 and random.random() < 0.2:
                if len(current_solution) >= 2:
                    i1, i2 = random.sample(range(len(current_solution)), 2)
                    idx1, idx2 = current_solution[i1], current_solution[i2]
                    
                    counts_no_12 = [0] * num_t_subsets
                    for idx in current_solution:
                        if idx == idx1 or idx == idx2:
                            continue
                        for s_idx in block_subsets[idx]:
                            counts_no_12[s_idx] += 1
                    
                    uncovered_by_removal = set()
                    for s in block_subsets[idx1]:
                        if counts_no_12[s] == 0:
                            uncovered_by_removal.add(s)
                    for s in block_subsets[idx2]:
                        if counts_no_12[s] == 0:
                            uncovered_by_removal.add(s)
                    
                    if not uncovered_by_removal:
                        new_sol = [b for b in current_solution if b != idx1 and b != idx2]
                        pruned_new = prune_solution(new_sol)
                        if len(pruned_new) < len(current_solution):
                            current_solution = pruned_new
                            improved = True
                            break
                    
                    if uncovered_by_removal:
                        pool = list(subset_to_blocks_set[list(uncovered_by_removal)[0]])
                        for s in list(uncovered_by_removal)[1:]:
                            pool = [b for b in pool if b in subset_to_blocks_set[s]]
                        
                        valid_pool = [b for b in pool if b not in current_solution]
                        
                        if valid_pool:
                            best_new = valid_pool[0]
                            best_red = -1
                            for b in valid_pool[:10]:
                                red = 0
                                for s in block_subsets[b]:
                                    if counts_no_12[s] >= 1:
                                        red += 1
                                if red > best_red:
                                    best_red = red
                                    best_new = b
                            
                            new_sol = [b for b in current_solution if b != idx1 and b != idx2]
                            new_sol.append(best_new)
                            pruned_new = prune_solution(new_sol)
                            if len(pruned_new) < len(current_solution):
                                current_solution = pruned_new
                                improved = True
                                break

        return current_solution

    # Main Loop
    best_solution = None
    best_len = float('inf')
    
    # 1. Initial Deterministic Greedy
    sol = greedy_construct(randomize=False)
    sol = local_search(sol, 5.0)
    best_solution = sol
    best_len = len(sol)
    
    # 2. Iterated Local Search with Kicks
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() < deadline - 10.0:
        if no_improve_count > max_no_improve:
            # Full restart with high temperature
            temp = random.uniform(1.0, 3.0)
            sol = greedy_construct(randomize=True, temperature=temp)
            sol = local_search(sol, 3.0)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol
                no_improve_count = 0
            else:
                no_improve_count += 1
            continue
        
        # Kick from best solution
        if best_len > 0:
            counts = [0] * num_t_subsets
            for idx in best_solution:
                for s_idx in block_subsets[idx]:
                    counts[s_idx] += 1
            
            # Choose kick size: 1, 2, or 3
            kick_size = random.choice([1, 1, 2, 3])
            if kick_size > len(best_solution):
                kick_size = len(best_solution)
            
            kick_indices = random.sample(range(len(best_solution)), kick_size)
            kick_set = set(best_solution[i] for i in kick_indices)
            
            new_base = [b for b in best_solution if b not in kick_set]
            
            # Repair with Greedy (randomized)
            temp = random.uniform(0.5, 1.5)
            sol = greedy_construct(randomize=True, temperature=temp)
            # Note: The greedy_construct above starts from scratch. 
            # To repair, we need to pass the base set. 
            # The current greedy_construct doesn't support start_set. 
            # Let's modify it or just use the full greedy for now, 
            # as the vectorized greedy is fast enough to run from scratch many times.
            
            # Local Search
            sol = local_search(sol, 4.0)
            
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol
                no_improve_count = 0
            else:
                no_improve_count += 1
    
    # Final polish
    if best_solution and time.time() < deadline - 3.0:
        best_solution = local_search(best_solution, 3.0)
    
    result_blocks = [list(all_blocks[i]) for i in best_solution]
    
    return result_blocks

if __name__ == "__main__":
    blocks = construct(19, 7, 4)
    print(f"Number of blocks: {len(blocks)}")
