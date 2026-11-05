import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def check_time(budget=5.0):
        return time.time() - start_time > deadline - budget

    # Precompute all t-subsets and index them
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map blocks to the indices of t-subsets they cover
    block_to_subsets = [[] for _ in range(num_blocks)]
    # Map t-subsets to the list of blocks that cover them
    subset_to_blocks = [[] for _ in range(num_tsubs)]
    
    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            s_idx = tsub_index[ts]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
            
    # Convert to numpy arrays for faster access where possible, 
    # but lists of lists are okay for the local search structure.
    
    def get_greedy_solution(seed=42, noise_level=0.0):
        rng = random.Random(seed)
        uncovered = np.ones(num_tsubs, dtype=bool)
        # Initialize gains: number of t-subsets covered by each block
        gains = np.array([len(bs) for bs in block_to_subsets], dtype=np.int32)
        selected = []
        
        # If noise is requested, we can simulate randomized greedy
        # For now, standard greedy with tie-breaking by index or random
        for _ in range(num_blocks):
            if not uncovered.any():
                break
            
            # Find best block
            # To add diversity, we can pick from top few if noise_level > 0
            if noise_level > 0:
                # Simple approach: add noise to gains
                noise = np.random.default_rng(rng.randint(0, 10**6)).uniform(0, noise_level, num_blocks)
                scores = gains + noise
                best_idx = np.argmax(scores)
            else:
                best_idx = np.argmax(gains)
                
            if gains[best_idx] == 0:
                break
                
            selected.append(best_idx)
            
            # Update uncovered and gains
            b_subs = block_to_subsets[best_idx]
            for s in b_subs:
                if uncovered[s]:
                    uncovered[s] = False
                    # Decrease gain for all blocks covering this subset
                    for b in subset_to_blocks[s]:
                        gains[b] -= 1
                        
        return selected

    def local_search_optimize(initial_indices, seed=42):
        """
        Performs deletion and swap local search to minimize the number of blocks.
        """
        rng = random.Random(seed)
        current_set = list(initial_indices)
        if not current_set:
            return []
            
        # Coverage counts: how many blocks in current_set cover each t-subset
        coverage_counts = np.zeros(num_tsubs, dtype=np.int32)
        for b in current_set:
            for s in block_to_subsets[b]:
                coverage_counts[s] += 1
                
        best_set = list(current_set)
        best_len = len(current_set)
        
        # Helper to check if a block is redundant (all its subsets are covered by others)
        def is_redundant(block_idx):
            for s in block_to_subsets[block_idx]:
                if coverage_counts[s] == 1:
                    return False
            return True

        # Helper to remove a block
        def remove_block(block_idx):
            nonlocal current_set
            # Find index in list
            try:
                pos = current_set.index(block_idx)
            except ValueError:
                return False
            
            for s in block_to_subsets[block_idx]:
                coverage_counts[s] -= 1
            current_set.pop(pos)
            return True

        # Helper to add a block
        def add_block(block_idx):
            nonlocal current_set
            current_set.append(block_idx)
            for s in block_to_subsets[block_idx]:
                coverage_counts[s] += 1

        # Helper to swap block old_idx with new_idx
        def swap_block(old_idx, new_idx):
            nonlocal current_set
            if old_idx == new_idx:
                return False
            try:
                pos = current_set.index(old_idx)
            except ValueError:
                return False
            
            # Remove old
            for s in block_to_subsets[old_idx]:
                coverage_counts[s] -= 1
            # Add new
            for s in block_to_subsets[new_idx]:
                coverage_counts[s] += 1
            current_set[pos] = new_idx
            return True

        max_iters = 500
        iters = 0
        
        while iters < max_iters and not check_time(5.0):
            iters += 1
            changed = False
            
            # 1. Deletion Pass
            # Iterate in random order to avoid bias
            indices = list(range(len(current_set)))
            rng.shuffle(indices)
            
            for i in indices:
                if not current_set:
                    break
                b = current_set[i]
                if is_redundant(b):
                    remove_block(b)
                    changed = True
                    # Note: We don't break immediately, we continue checking others?
                    # Actually, removing one block might make another redundant.
                    # But the loop index `i` refers to the old list.
                    # It's safer to restart the deletion pass or just continue and check validity.
                    # Since we check `is_redundant` dynamically, continuing is okay, 
                    # but `current_set[i]` might be wrong if we popped.
                    # Let's restart deletion pass on change for correctness and simplicity, 
                    # or just break and loop again.
                    break 

            if not current_set:
                break

            if len(current_set) < best_len:
                best_len = len(current_set)
                best_set = list(current_set)

            # 2. Swap Pass
            # Try to swap blocks to enable further deletions or just improve structure
            # We look for blocks that are "critical" (have unique subsets)
            # and try to replace them with a block that covers those unique subsets.
            
            indices = list(range(len(current_set)))
            rng.shuffle(indices)
            
            swaps_done = 0
            max_swaps = 20 # Limit swaps per iteration to allow time
            
            for i in indices:
                if swaps_done >= max_swaps:
                    break
                if not current_set:
                    break
                if check_time(2.0):
                    break
                    
                b = current_set[i]
                
                # Find unique subsets for b
                unique_subs = []
                for s in block_to_subsets[b]:
                    if coverage_counts[s] == 1:
                        unique_subs.append(s)
                        
                if not unique_subs:
                    # This block is redundant. 
                    # It should have been caught by deletion pass.
                    # If not, maybe due to ordering. Try to remove it.
                    if is_redundant(b):
                        remove_block(b)
                        changed = True
                        break # Restart
                    continue

                # We need to find a block b_new that covers all unique_subs
                # Intersection of subset_to_blocks[s] for s in unique_subs
                candidates = None
                for s in unique_subs:
                    if candidates is None:
                        candidates = set(subset_to_blocks[s])
                    else:
                        candidates.intersection_update(subset_to_blocks[s])
                        if not candidates:
                            break
                
                if not candidates:
                    continue
                    
                # Remove b from candidates if it's there (we want a *different* block)
                # Actually, if b is the only candidate, we can't swap.
                candidates.discard(b)
                
                if not candidates:
                    continue
                    
                # Pick a random candidate from the set
                # Prefer candidates not already in current_set?
                # If we pick one already in current_set, it's a no-op or duplicate removal?
                # We maintain a set of unique blocks.
                current_set_set = set(current_set)
                valid_candidates = [c for c in candidates if c not in current_set_set]
                
                if not valid_candidates:
                    # All candidates are already in the set.
                    # This means the unique subsets are covered by the union of other blocks in the set?
                    # No, `candidates` are blocks that cover ALL unique subsets.
                    # If a candidate `c` is in `current_set`, then `c` covers the unique subsets.
                    # But wait, if `c` is in `current_set`, then `coverage_counts[s]` for `s in unique_subs` 
                    # would be at least 2 (once from `b`, once from `c`).
                    # Contradiction: `unique_subs` are those with count 1.
                    # So `c` cannot be in `current_set` if it covers `s` and `b` covers `s` and count is 1.
                    # Unless `c == b`. But we discarded `b`.
                    # So `valid_candidates` should not be empty if `candidates` is not empty?
                    # Let's verify:
                    # If `c` is in `current_set` and `c != b`, then `c` covers `s`.
                    # `b` covers `s`.
                    # So `coverage_counts[s] >= 2`.
                    # But `s` is in `unique_subs`, so `coverage_counts[s] == 1`.
                    # Therefore, no other block in `current_set` covers `s`.
                    # So `c` cannot be in `current_set`.
                    # Thus `valid_candidates` is effectively `candidates`.
                    pass

                if valid_candidates:
                    # Pick one
                    new_b = rng.choice(valid_candidates)
                    
                    # Perform swap
                    # Check if this swap is beneficial?
                    # A swap itself doesn't reduce count.
                    # But it might make another block redundant.
                    # We perform the swap and let the next deletion pass handle the rest.
                    # To avoid thrashing, we only swap if we think it helps.
                    # Heuristic: Swap if the new block covers more "scarce" subsets?
                    # For now, just swap randomly to explore.
                    
                    swap_block(b, new_b)
                    swaps_done += 1
                    changed = True
                    # Don't break immediately, allow multiple swaps?
                    # But indices are based on old list. 
                    # `current_set` changed, so `current_set[i]` is now `new_b`.
                    # We should probably break to re-evaluate the whole set.
                    break

            if not changed:
                # No deletions and no swaps (or swaps didn't lead to immediate change in logic flow)
                # Try a more aggressive perturbation: random restart of local search?
                # Or just stop if we've been stuck for a while.
                # For now, we can try to inject a random block and delete 2?
                # Let's stick to simple local search for stability.
                # If no change, break to save time for other seeds.
                break

        return best_set

    # --- Main Execution ---
    
    best_blocks = []
    best_len = float('inf')
    
    # Run multiple instances with different seeds and noise
    # The greedy initialization benefits from noise to find diverse starting points
    seeds = [42, 123, 999, 31415, 27182]
    
    for i, seed in enumerate(seeds):
        if check_time(10.0):
            break
            
        # Vary noise level for diversity
        noise = 0.0 if i == 0 else 0.5
        
        initial = get_greedy_solution(seed=seed, noise_level=noise)
        
        if not initial:
            continue
            
        # Optimize
        optimized = local_search_optimize(initial, seed=seed)
        
        if len(optimized) < best_len:
            best_len = len(optimized)
            best_blocks = optimized
            
        # If we find a very good solution, we can spend more time polishing it
        # For now, we just iterate through seeds.
        
    # Final polish: Run local search again on the best found solution with different seeds
    if best_blocks:
        for seed in [100, 200, 300]:
            if check_time(5.0):
                break
            # Re-optimize the best solution
            # Note: local_search_optimize takes a list of indices.
            # We need to ensure we are passing the indices, not the blocks themselves.
            # `best_blocks` from previous step is a list of indices (from `local_search_optimize` return).
            # Wait, `get_greedy_solution` returns indices. `local_search_optimize` returns indices.
            # So `best_blocks` is a list of indices.
            
            # To allow exploration from the best solution, we can do a "kick"
            # or just run the local search again. Since it's deterministic given seed,
            # running with new seeds on the same start might not change much if it's a strict local search.
            # However, the random shuffles inside `local_search_optimize` make it stochastic.
            
            polished = local_search_optimize(best_blocks, seed=seed)
            if len(polished) < best_len:
                best_len = len(polished)
                best_blocks = polished

    # Convert indices back to blocks
    if not best_blocks:
        # Fallback: just return the greedy result if local search failed or found nothing
        # (Unlikely, greedy should work)
        initial = get_greedy_solution(seed=42)
        return [list(all_blocks[i]) for i in initial]
        
    return [list(all_blocks[i]) for i in best_blocks]
