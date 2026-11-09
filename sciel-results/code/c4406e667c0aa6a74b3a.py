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
    
    # Build block_masks as list of Python big ints
    block_masks = [0] * num_blocks
    block_subsets = [[] for _ in range(num_blocks)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        subs = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            subs.append(idx)
        block_masks[i] = mask
        block_subsets[i] = subs

    # Inverted index
    subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for i in range(num_blocks):
        for idx in block_subsets[i]:
            subset_to_blocks[idx].append(i)
    
    subset_to_blocks_set = [set(lst) for lst in subset_to_blocks]

    def greedy_construct(start_set=None, randomize=False, seed=None, temperature=1.0):
        """
        Greedy set cover with Softmax selection for randomization.
        """
        if start_set is None:
            uncovered = (1 << num_t_subsets) - 1
            chosen = []
            used = [False] * num_blocks
        else:
            uncovered = (1 << num_t_subsets) - 1
            chosen = list(start_set)
            used = [False] * num_blocks
            for idx in chosen:
                used[idx] = True
                uncovered &= ~block_masks[idx]
        
        while uncovered != 0:
            if time.time() > deadline - 5.0:
                break
            
            # Compute gains
            gains = np.zeros(num_blocks, dtype=np.int32)
            # We use a Python loop for the big-int bit_count, but store in NumPy for potential future vectorization or just for ease of access
            # Actually, just a list is fine.
            best_gain = -1
            best_idx = -1
            
            # If randomize, we use Softmax over top candidates
            if randomize:
                # Collect top candidates to avoid calculating softmax on 50k items
                # Just find the max gain, and then pick from those with gain >= max_gain - threshold
                # Or simpler: Pick from top 5% of gains?
                # Let's stick to the previous strategy but with a "temperature" parameter
                # that controls how often we pick a sub-optimal block.
                
                # Find max gain
                max_gain = 0
                for i in range(num_blocks):
                    if used[i]:
                        continue
                    gain = (block_masks[i] & uncovered).bit_count()
                    if gain > max_gain:
                        max_gain = gain
                
                if max_gain == 0:
                    break

                # Collect candidates with gain close to max
                # Threshold: max_gain - 2 (or scaled by temperature)
                threshold = max(1, int(max_gain * (1 - 0.1 * temperature)))
                candidates = []
                for i in range(num_blocks):
                    if used[i]:
                        continue
                    gain = (block_masks[i] & uncovered).bit_count()
                    if gain >= threshold:
                        candidates.append((gain, i))
                
                if not candidates:
                    break

                # Softmax selection
                # Normalize gains
                max_cand_gain = max(g for g, _ in candidates)
                # exp(gain / T)
                # To avoid overflow, subtract max
                exps = []
                total_exp = 0
                for g, idx in candidates:
                    # Scale gain to 0..1 range for better softmax behavior
                    # Or just use gain directly if T is large
                    # Let's use gain / (max_cand_gain + 1) * 10 / temperature
                    val = np.exp((g / (max_cand_gain + 1)) * (10.0 / temperature))
                    exps.append(val)
                    total_exp += val
                
                r = random.random() * total_exp
                acc = 0
                best_idx = candidates[-1][1] # Fallback
                for j, (g, idx) in enumerate(candidates):
                    acc += exps[j]
                    if r <= acc:
                        best_idx = idx
                        break
            else:
                # Deterministic max
                for i in range(num_blocks):
                    if used[i]:
                        continue
                    gain = (block_masks[i] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx == -1:
                    break

            chosen.append(best_idx)
            used[best_idx] = True
            uncovered &= ~block_masks[best_idx]
            
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
            # This is more expensive. Only do it if we have time and haven't improved for a while.
            if time.time() - start_ls < time_limit - 1.0 and random.random() < 0.2:
                # Pick 2 random blocks to remove
                if len(current_solution) >= 2:
                    i1, i2 = random.sample(range(len(current_solution)), 2)
                    idx1, idx2 = current_solution[i1], current_solution[i2]
                    
                    # Essential subsets of idx1 and idx2 (considering each other)
                    # Count subsets covered by idx1 or idx2
                    # If a subset is covered ONLY by idx1 (and not idx2), it's essential for idx1
                    # If a subset is covered ONLY by idx2 (and not idx1), it's essential for idx2
                    # If a subset is covered by BOTH, it's not essential for either individually, 
                    # but if we remove both, it becomes uncovered unless another block covers it.
                    
                    # Simpler: Calculate counts without idx1 and idx2
                    counts_no_12 = [0] * num_t_subsets
                    for idx in current_solution:
                        if idx == idx1 or idx == idx2:
                            continue
                        for s_idx in block_subsets[idx]:
                            counts_no_12[s_idx] += 1
                    
                    # Subsets that are uncovered if we remove idx1 and idx2
                    # These are subsets s where counts_no_12[s] == 0 AND (s in idx1 OR s in idx2)
                    # Actually, if s in idx1 and s in idx2, and counts_no_12[s] == 0, it's uncovered.
                    # If s in idx1 only, and counts_no_12[s] == 0, it's uncovered.
                    
                    uncovered_by_removal = set()
                    for s in block_subsets[idx1]:
                        if counts_no_12[s] == 0:
                            uncovered_by_removal.add(s)
                    for s in block_subsets[idx2]:
                        if counts_no_12[s] == 0:
                            uncovered_by_removal.add(s)
                    
                    if not uncovered_by_removal:
                        # We can just remove both?
                        # If we remove both and nothing is uncovered, we gain 2 blocks.
                        # But we need to add 2 blocks to keep size same? No, we want to reduce size.
                        # If we remove 2 and add 0, size decreases by 2.
                        # But we must cover all subsets.
                        # If uncovered_by_removal is empty, we can just remove them.
                        new_sol = [b for b in current_solution if b != idx1 and b != idx2]
                        pruned_new = prune_solution(new_sol)
                        if len(pruned_new) < len(current_solution):
                            current_solution = pruned_new
                            improved = True
                            break
                    
                    # If there are uncovered subsets, we need to find 2 blocks that cover them.
                    # This is a small set cover problem.
                    # For now, skip 2-swap if it's complex.
                    # Just try to find 1 block that covers all uncovered_by_removal?
                    # If found, we remove 2 and add 1. Net gain 1.
                    if uncovered_by_removal:
                        # Find blocks that cover ALL subsets in uncovered_by_removal
                        pool = list(subset_to_blocks_set[list(uncovered_by_removal)[0]])
                        for s in list(uncovered_by_removal)[1:]:
                            pool = [b for b in pool if b in subset_to_blocks_set[s]]
                        
                        # Exclude idx1, idx2 and any block in current_solution (unless we are swapping)
                        # We are replacing idx1, idx2 with 1 new block.
                        # So new block must not be in current_solution (unless it's idx1/idx2 which we are removing)
                        valid_pool = [b for b in pool if b not in current_solution or b in (idx1, idx2)]
                        # Actually, if b is in current_solution and b is not idx1/idx2, adding it is redundant?
                        # No, we are removing idx1, idx2. If b is in current_solution, it's already there.
                        # We need a NEW block.
                        valid_pool = [b for b in pool if b not in current_solution]
                        
                        if valid_pool:
                            # Pick the one with highest redundancy
                            best_new = valid_pool[0]
                            best_red = -1
                            for b in valid_pool[:10]: # Limit
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
    sol = greedy_construct()
    sol = local_search(sol, 5.0)
    best_solution = sol
    best_len = len(sol)
    
    # 2. Iterated Local Search with Kicks
    no_improve_count = 0
    max_no_improve = 20
    
    # Diversify starting points with different temperatures and kicks
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
            sol = greedy_construct(start_set=new_base, randomize=True, temperature=temp)
            
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
