import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115.0  # Safety margin before 120s limit

    # 1. Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # block_tsubs: list of arrays of t-subset indices for each block
    block_tsubs = [None] * n_blocks
    # tsub_blocks: list of lists of block indices for each t-subset
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = np.array(tsubs, dtype=np.int32)
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
            
    # Convert tsub_blocks to sets for fast intersection during swaps
    tsub_block_sets = [set(lst) for lst in tsub_blocks]
    
    # 2. Heuristic Greedy Initialization
    # We use a weighted greedy approach. Weights are initialized to 1.
    # To add diversity, we can randomize the initial weights slightly or the tie-breaking.
    
    def greedy_init(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        
        # Initial gain is just the number of t-subsets (since all are uncovered)
        # But we can weight by "rarity" or just use count.
        # Let's stick to count for speed, but randomize tie-breaking.
        gain = [len(block_tsubs[i]) for i in range(n_blocks)]
        
        selected = []
        
        # To speed up max finding, we could use a heap, but scanning 50k is fast.
        # However, updating gains is the bottleneck.
        # We update gains only for blocks that share t-subsets with the chosen block.
        
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
            
            # Find max gain
            # Using numpy argmax on a list converted to array is okay, but we maintain a list.
            # Let's just use max.
            max_gain = -1
            best_idx = -1
            # Optimization: We can keep a list of indices with current max gain?
            # Or just scan.
            for i in range(n_blocks):
                if gain[i] > max_gain:
                    max_gain = gain[i]
                    best_idx = i
                    # Randomize tie-breaking: if equal, pick with some probability?
                    # Or just pick the first. To add diversity, we can shuffle candidates.
                    # But finding candidates is O(N).
                    # Let's just pick the first max for now, seed affects initial order if we shuffled blocks?
                    # No, blocks are fixed.
                    # Let's add a small random noise to gain initially?
                    # Or just rely on multiple restarts.
                    
            if max_gain <= 0:
                break
                
            selected.append(best_idx)
            
            # Update gains
            # For each t-subset in best_idx that was uncovered:
            #   mark covered
            #   decrement gain for all blocks covering it
            ts_list = block_tsubs[best_idx]
            active_mask = uncovered[ts_list]
            if np.any(active_mask):
                active_ts_indices = ts_list[active_mask]
                uncovered[active_ts_indices] = False
                
                for ts_idx in active_ts_indices:
                    for bj in tsub_blocks[ts_idx]:
                        gain[bj] -= 1
                        
            # Prevent re-selection
            gain[best_idx] = -1
            
        return selected

    def local_search(block_indices, seed=0):
        """
        Local search: Remove redundant blocks, then try swaps.
        """
        rng = random.Random(seed)
        selected = set(block_indices)
        
        # Compute coverage counts
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for bi in selected:
            cover_count[block_tsubs[bi]] += 1
            
        best_selected = set(selected)
        best_count = len(selected)
        
        improved = True
        max_rounds = 100
        round_num = 0
        
        while improved and time.time() < deadline - 2 and round_num < max_rounds:
            improved = False
            round_num += 1
            
            # Phase 1: Remove redundant blocks
            to_remove = []
            for bi in selected:
                # Check if any t-subset has count 1
                if np.min(cover_count[block_tsubs[bi]]) > 1:
                    to_remove.append(bi)
                    
            if to_remove:
                for bi in to_remove:
                    selected.discard(bi)
                    cover_count[block_tsubs[bi]] -= 1
                improved = True
                # Update best
                if len(selected) < best_count:
                    best_count = len(selected)
                    best_selected = set(selected)
                continue # Restart loop
            
            # Phase 2: Try swaps
            # For each block, find unique t-subsets.
            # If none, it's redundant (handled above).
            # If some, find a replacement block that covers all unique t-subsets.
            
            current_blocks = list(selected)
            rng.shuffle(current_blocks)
            
            for bi in current_blocks:
                if time.time() > deadline - 2:
                    break
                if bi not in selected:
                    continue
                    
                ts_indices = block_tsubs[bi]
                unique_mask = cover_count[ts_indices] == 1
                unique_tsubs = ts_indices[unique_mask]
                
                if len(unique_tsubs) == 0:
                    # Should have been removed in Phase 1, but just in case
                    selected.discard(bi)
                    cover_count[ts_indices] -= 1
                    improved = True
                    break # Restart
                    
                # Find candidates that cover all unique_tsubs
                # Start with the rarest unique t-subset
                # Sort unique_tsubs by frequency of covering blocks
                # We precomputed tsub_block_sets.
                # min_ts = min(unique_tsubs, key=lambda x: len(tsub_block_sets[x]))
                
                # To speed up, just pick the first one? Or the one with fewest blocks?
                # Let's pick the one with fewest blocks to minimize candidate set.
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_block_sets[x]))
                candidates = set(tsub_block_sets[min_ts])
                
                for ts_idx in unique_tsubs:
                    if ts_idx == min_ts:
                        continue
                    candidates &= tsub_block_sets[ts_idx]
                    
                candidates.discard(bi)
                
                if not candidates:
                    continue
                
                # Try swapping
                # We want to minimize the "cost" of the swap?
                # Ideally, we want a block that covers many other uncovered things?
                # But we are maintaining coverage, so we just need any valid swap.
                # However, different swaps lead to different local minima.
                # Let's pick a random candidate from the list.
                
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                swapped = False
                # Limit attempts to avoid hanging on a single block
                for bj in cand_list[:20]: 
                    if bj in selected:
                        continue
                        
                    # Perform swap
                    selected.discard(bi)
                    cover_count[ts_indices] -= 1
                    
                    selected.add(bj)
                    cover_count[block_tsubs[bj]] += 1
                    
                    improved = True
                    swapped = True
                    
                    # Update best
                    if len(selected) < best_count:
                        best_count = len(selected)
                        best_selected = set(selected)
                        
                    break # Restart the while loop? Or continue to next block?
                    # Restarting is safer to re-evaluate redundancy.
                    # But we can also just continue.
                    # Let's restart the outer while loop to be safe.
                    # Actually, breaking out of the for loop and setting improved=True
                    # will trigger the while loop condition check.
                    # But we need to restart the Phase 1/2 logic.
                    # So break out of the for loop.
                
                if swapped:
                    break # Break out of the for loop, go back to while loop
            
            # If no swap found in this round, and no removals, we are stuck.
            # We can try a random perturbation?
            # For now, just stop.
            
        return [all_blocks[bi] for bi in best_selected]

    def iterated_local_search(seed=0):
        """
        ILS: Start with greedy, refine, perturb, refine again.
        """
        rng = random.Random(seed)
        
        # Initial solution
        init_blocks = greedy_init(seed)
        if not init_blocks:
            return None
            
        current_blocks = local_search(init_blocks, seed)
        current_count = len(current_blocks)
        
        best_blocks = current_blocks
        best_count = current_count
        
        # ILS Loop
        max_iters = 20
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Perturb: Remove a few blocks, add some random ones?
            # Or just start a new greedy with different seed?
            # Let's try a "shake": remove 5-10 blocks, then repair.
            
            # Convert to set
            curr_set = set(current_blocks)
            
            # Remove some blocks
            n_remove = rng.randint(2, 5)
            to_remove = rng.sample(list(curr_set), min(n_remove, len(curr_set)))
            for bi in to_remove:
                curr_set.discard(bi)
                
            # Repair: Add blocks to cover gaps
            # We need to check coverage.
            # This is expensive to recompute from scratch.
            # Let's just use greedy_init again but biased?
            # Or just do a fresh greedy_init with a new seed.
            # Given time constraints, fresh restarts might be better than complex repair.
            
            # Let's do a fresh greedy_init with a new seed derived from current seed + i
            new_seed = seed * 1000 + i
            new_init = greedy_init(new_seed)
            if new_init:
                new_blocks = local_search(new_init, new_seed)
                if len(new_blocks) < best_count:
                    best_count = len(new_blocks)
                    best_blocks = new_blocks
                    current_blocks = new_blocks
                # Else keep current best? Or accept worse to escape?
                # Simulated Annealing style: accept worse with probability exp(-delta/T)
                # But we don't have a temperature schedule easily here.
                # Let's just keep the best found so far.
            
        return best_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Multiple restarts
    num_restarts = 20
    for seed in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Use ILS for each seed
        blocks = iterated_local_search(seed)
        
        if blocks and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # Early termination
        if best_len <= 145:
            break

    if best_blocks is None:
        # Fallback: simple greedy
        init = greedy_init(0)
        best_blocks = [all_blocks[bi] for bi in init]

    return best_blocks
