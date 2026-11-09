import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage of each block as a list of t-subset indices
    # and as a bitmask for fast intersection
    block_subsets = [[] for _ in range(num_blocks)]
    block_masks = [0] * num_blocks
    
    for i, block in enumerate(all_blocks):
        mask = 0
        subs = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            subs.append(idx)
        block_masks[i] = mask
        block_subsets[i] = subs

    # Inverted index: for each t-subset, which blocks contain it
    # Using sets for fast intersection
    subset_to_blocks = [set() for _ in range(num_t_subsets)]
    for i in range(num_blocks):
        for idx in block_subsets[i]:
            subset_to_blocks[idx].add(i)

    # Full mask for all t-subsets
    full_mask = (1 << num_t_subsets) - 1

    def prune_solution(chosen_indices):
        """
        Removes redundant blocks. A block is redundant if all its t-subsets
        are covered by at least one other block in the chosen set.
        Returns a new list of indices.
        """
        if not chosen_indices:
            return []
        
        # Count coverage for each t-subset
        counts = [0] * num_t_subsets
        for idx in chosen_indices:
            for s_idx in block_subsets[idx]:
                counts[s_idx] += 1
        
        current_list = list(chosen_indices)
        changed = True
        
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
                    # Remove this block
                    for s_idx in block_subsets[idx]:
                        counts[s_idx] -= 1
                    changed = True
                else:
                    new_list.append(idx)
            current_list = new_list
            
        return current_list

    def local_search(chosen_indices, time_limit=5.0):
        """
        1-Swap Local Search.
        Tries to replace a block with another that covers its essential parts,
        potentially allowing further pruning.
        """
        current_solution = prune_solution(list(chosen_indices))
        start_ls = time.time()
        
        improved = True
        while improved and time.time() - start_ls < time_limit:
            improved = False
            
            # Compute counts
            counts = [0] * num_t_subsets
            for idx in current_solution:
                for s_idx in block_subsets[idx]:
                    counts[s_idx] += 1
            
            # Find a block to swap
            # We iterate through blocks. If we find a beneficial swap, we apply it and restart.
            # To avoid getting stuck, we can randomize the order or try multiple candidates.
            
            # Identify essential subsets for each block
            # We look for a block B in solution and a block C not in solution
            # such that C covers all essential subsets of B, and the swap reduces size.
            
            # Optimization: Only consider blocks that have essential subsets?
            # Actually, if a block has no essential subsets, it's redundant and should have been pruned.
            # So every block in current_solution has at least one essential subset.
            
            # To speed up, we can limit the number of swaps per iteration or check time more frequently.
            
            # Heuristic: Pick a random block to try to replace, or iterate deterministically?
            # Deterministic is safer for reproducibility, but random helps escape cycles.
            # Let's iterate through all blocks, but break early if found.
            
            for i, idx in enumerate(current_solution):
                # Find essential subsets for idx
                essential_subs = [s for s in block_subsets[idx] if counts[s] == 1]
                
                if not essential_subs:
                    # Should not happen after prune, but just in case
                    for s in block_subsets[idx]:
                        counts[s] -= 1
                    current_solution.pop(i)
                    improved = True
                    break
                
                # Find candidate blocks that cover ALL essential subsets
                # Intersection of subset_to_blocks for all essential_subs
                if len(essential_subs) == 1:
                    pool = subset_to_blocks[essential_subs[0]]
                else:
                    # Start with the smallest set to minimize intersection cost
                    min_set_idx = min(essential_subs, key=lambda s: len(subset_to_blocks[s]))
                    pool = set(subset_to_blocks[min_set_idx])
                    for s in essential_subs:
                        if s == min_set_idx:
                            continue
                        pool &= subset_to_blocks[s]
                        if not pool:
                            break
                
                if not pool:
                    continue
                
                # Filter out current block and blocks already in solution
                # Note: We might swap with a block already in solution? No, that's a no-op or swap.
                # We want C not in current_solution.
                # Also, C should not be idx.
                
                found_swap = False
                # Randomize candidate selection to avoid bias
                candidates = list(pool)
                random.shuffle(candidates)
                
                for cand_idx in candidates:
                    if cand_idx == idx:
                        continue
                    if cand_idx in current_solution:
                        continue
                    
                    # Verify cand_idx covers all essential_subs (should be true by construction of pool)
                    # But let's double check if pool logic was approximate (it's exact here)
                    
                    # Calculate new solution size if we swap idx with cand_idx
                    # New solution: current_solution with idx replaced by cand_idx
                    # Then prune.
                    
                    # To avoid expensive prune every time, we can estimate.
                    # But prune is O(N * K * T) roughly, which is small.
                    
                    test_sol = list(current_solution)
                    test_sol[i] = cand_idx
                    pruned_test = prune_solution(test_sol)
                    
                    if len(pruned_test) < len(current_solution):
                        current_solution = pruned_test
                        improved = True
                        found_swap = True
                        break
                    elif len(pruned_test) == len(current_solution):
                        # Neutral move? Accept with low probability to explore?
                        # Usually we only accept improvements to ensure progress.
                        # However, sometimes a neutral move leads to an improvement later.
                        # For now, stick to strict improvement to guarantee convergence.
                        pass
                
                if found_swap:
                    break
            
        return current_solution

    def greedy_construct(randomize=False):
        """
        Greedy set cover.
        """
        uncovered = full_mask
        chosen = []
        used = [False] * num_blocks
        
        # Available blocks
        # To speed up, we can just iterate through all blocks and check if they add value.
        # But as uncovered shrinks, many blocks become useless.
        
        while uncovered != 0:
            if time.time() > deadline - 5.0:
                break
            
            best_gain = -1
            best_idx = -1
            second_best_gain = -1
            second_best_idx = -1
            
            # If randomize, we can sample or add noise
            # For efficiency, we iterate through all blocks. 50k blocks * bit_count is fast enough?
            # 50k * (big int bit_count) is very fast in Python 3.10+.
            
            for i in range(num_blocks):
                if used[i]:
                    continue
                gain = (block_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    if randomize and best_gain > 0 and random.random() < 0.2:
                        # Keep some diversity? No, greedy is greedy.
                        # Randomization in greedy is usually done by picking top-k randomly.
                        pass
                    second_best_gain = best_gain
                    second_best_idx = best_idx
                    best_gain = gain
                    best_idx = i
                elif gain == best_gain and gain > 0:
                    if randomize:
                        if random.random() < 0.5:
                            second_best_gain = best_gain
                            second_best_idx = best_idx
                            best_gain = gain
                            best_idx = i
            
            if best_idx == -1:
                break
            
            # If randomize, occasionally pick second best
            if randomize and second_best_idx != -1 and random.random() < 0.1:
                best_idx = second_best_idx
            
            chosen.append(best_idx)
            used[best_idx] = True
            uncovered &= ~block_masks[best_idx]
            
        return chosen

    # Main Loop
    best_solution = None
    best_len = float('inf')
    
    # 1. Initial Deterministic Greedy
    sol = greedy_construct(randomize=False)
    sol = local_search(sol, 5.0)
    best_solution = sol
    best_len = len(sol)
    
    # 2. Iterated Local Search
    no_improve_count = 0
    max_no_improve = 10
    
    while time.time() < deadline - 10.0:
        if no_improve_count > max_no_improve:
            # Restart with random greedy
            sol = greedy_construct(randomize=True)
            sol = local_search(sol, 3.0)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol
                no_improve_count = 0
            else:
                no_improve_count += 1
            continue
        
        # Kick: Remove a random block from best_solution and repair
        if best_len > 0:
            # Choose a block to remove. 
            # Strategy: Remove a block that is "least essential" (covers few unique subsets)
            # Or just random.
            
            # Compute counts to find least essential
            counts = [0] * num_t_subsets
            for idx in best_solution:
                for s_idx in block_subsets[idx]:
                    counts[s_idx] += 1
            
            # Find block with minimum unique coverage
            min_unique = float('inf')
            kick_idx = 0
            for i, idx in enumerate(best_solution):
                unique_count = sum(1 for s in block_subsets[idx] if counts[s] == 1)
                if unique_count < min_unique:
                    min_unique = unique_count
                    kick_idx = i
            
            # Sometimes use random kick
            if random.random() < 0.5:
                kick_idx = random.randint(0, len(best_solution) - 1)
            
            new_base = [b for i, b in enumerate(best_solution) if i != kick_idx]
            
            # Repair with Greedy
            # We need to cover the subsets that were only covered by the removed block.
            # Greedy will handle this.
            sol = greedy_construct(randomize=True)
            
            # Better repair: Start from new_base and add blocks to cover missing parts
            # The greedy_construct above starts from scratch. 
            # Let's modify greedy to start from a set?
            # Actually, greedy_construct is simple. Let's just run it and take the best.
            # Or, we can implement a "repair" that starts with new_base.
            
            # Let's implement a quick repair:
            # uncovered = full_mask
            # for idx in new_base: uncovered &= ~block_masks[idx]
            # then greedy add.
            
            # I will write a helper for greedy repair starting from a set.
            def greedy_repair(start_set):
                uncovered = full_mask
                chosen = list(start_set)
                used = [False] * num_blocks
                for idx in chosen:
                    used[idx] = True
                    uncovered &= ~block_masks[idx]
                
                while uncovered != 0:
                    if time.time() > deadline - 5.0:
                        break
                    best_gain = -1
                    best_idx = -1
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

            sol = greedy_repair(new_base)
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
    
    # Convert to lists of points
    result_blocks = [list(all_blocks[i]) for i in best_solution]
    
    return result_blocks

if __name__ == "__main__":
    blocks = construct(19, 7, 4)
    print(f"Number of blocks: {len(blocks)}")
