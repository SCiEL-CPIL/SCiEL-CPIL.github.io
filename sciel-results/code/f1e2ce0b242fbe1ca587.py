import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, compute the list of t-subset indices it covers
    # Using lists of ints for fast iteration in Python loops
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = [t_sub_to_idx[sub] for sub in itertools.combinations(k_sub, t)]
        k_sub_t_idxs.append(t_idxs)
    
    # Build reverse mapping: for each t-subset, which k-subsets cover it
    # Stored as lists for easy set operations
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    def greedy_cover(seed):
        """
        Efficient greedy set cover using NumPy for gain updates.
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        
        # Initial gains: number of t-subsets covered by each block
        # All blocks cover C(k,t) subsets initially.
        gains = np.full(n_k, len(k_sub_t_idxs[0]), dtype=np.int32)
        
        selected = []
        
        for _ in range(n_k): # Safety bound
            if not uncovered.any():
                break
            
            # Find best block
            # We add noise to break ties and introduce randomness for different seeds
            # Noise magnitude is small relative to gain (which is ~1287 for C(8,5))
            # Actually, for C(19,8,5), C(8,5)=56.
            # So noise should be small, e.g., 0.1
            if seed > 0:
                # Vectorized noise generation is faster than loop
                noise = rng.random() * 0.1
                scores = gains.astype(np.float64) + noise
            else:
                scores = gains.astype(np.float64)
            
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected.append(best_idx)
            
            # Update gains
            # For each t-subset covered by best_idx, if it was uncovered,
            # decrement gain for all blocks covering that t-subset
            for t_idx in k_sub_t_idxs[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    # Vectorized decrement? No, t_sub_to_k_idxs is a list of lists.
                    # We have to loop.
                    for b_idx in t_sub_to_k_idxs[t_idx]:
                        gains[b_idx] -= 1
                        
        return selected

    def reduce_blocks(blocks_idx):
        """Remove redundant blocks via iterative pass."""
        if not blocks_idx:
            return blocks_idx

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        changed = True
        while changed:
            changed = False
            new_blocks = []
            # Shuffle to avoid bias in deletion order
            # Note: blocks_idx is a list of indices
            shuffled_blocks = list(blocks_idx)
            random.shuffle(shuffled_blocks)
            
            for b_idx in shuffled_blocks:
                is_essential = False
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
            blocks_idx = new_blocks

        return blocks_idx

    def local_search(blocks_idx, max_time):
        """
        Iterated Local Search: Deletion + Swap.
        State: List of block indices.
        """
        current_blocks = list(blocks_idx)
        if not current_blocks:
            return current_blocks

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_blocks:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1
        
        best_blocks = list(current_blocks)
        best_len = len(current_blocks)
        
        rng = random.Random(12345)
        
        # Track time
        while time.time() < max_time:
            # 1. Deletion Pass
            changed = True
            while changed:
                changed = False
                # Shuffle to avoid bias
                shuffled_indices = list(range(len(current_blocks)))
                rng.shuffle(shuffled_indices)
                
                for i in shuffled_indices:
                    if i >= len(current_blocks):
                        continue
                    b_idx = current_blocks[i]
                    is_essential = False
                    for t_idx in k_sub_t_idxs[b_idx]:
                        if cover_count[t_idx] == 1:
                            is_essential = True
                            break
                    if not is_essential:
                        # Remove
                        for t_idx in k_sub_t_idxs[b_idx]:
                            cover_count[t_idx] -= 1
                        current_blocks.pop(i)
                        changed = True
                        break # Restart loop as indices changed

            # Check if improved
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = list(current_blocks)

            # 2. Swap Pass
            # Try to swap blocks to improve structure
            # We iterate through a random subset of blocks to save time
            num_swaps_to_try = min(len(current_blocks), 50)
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            swaps_done = 0
            for i in indices[:num_swaps_to_try]:
                if time.time() >= max_time:
                    break
                if i >= len(current_blocks):
                    continue
                    
                old_b = current_blocks[i]
                
                # Find unique subsets covered by old_b
                unique_ts = []
                for t_idx in k_sub_t_idxs[old_b]:
                    if cover_count[t_idx] == 1:
                        unique_ts.append(t_idx)
                
                if not unique_ts:
                    # No unique subsets. We can swap with any other block.
                    # This allows "shuffling" the set without breaking coverage.
                    # Pick a random block from all_k_subs that is NOT in current_blocks.
                    # To ensure it's not in current_blocks, we can try a few times.
                    new_b = -1
                    for _ in range(10):
                        candidate = rng.randint(0, n_k - 1)
                        if candidate not in current_blocks:
                            new_b = candidate
                            break
                    if new_b == -1:
                        continue
                    
                    # Perform swap
                    # Remove old_b
                    for t_idx in k_sub_t_idxs[old_b]:
                        cover_count[t_idx] -= 1
                    # Add new_b
                    for t_idx in k_sub_t_idxs[new_b]:
                        cover_count[t_idx] += 1
                    
                    current_blocks[i] = new_b
                    swaps_done += 1
                    
                else:
                    # We have unique subsets. New block MUST cover all of them.
                    # Find candidates
                    candidates = set(t_sub_to_k_idxs[unique_ts[0]])
                    for t_idx in unique_ts[1:]:
                        candidates.intersection_update(t_sub_to_k_idxs[t_idx])
                    
                    # Remove old_b and current blocks from candidates
                    candidates.discard(old_b)
                    candidates -= set(current_blocks)
                    
                    if not candidates:
                        continue # No valid swap candidates
                    
                    # Pick a random candidate
                    new_b = rng.choice(list(candidates))
                    
                    # Perform swap
                    for t_idx in k_sub_t_idxs[old_b]:
                        cover_count[t_idx] -= 1
                    for t_idx in k_sub_t_idxs[new_b]:
                        cover_count[t_idx] += 1
                    
                    current_blocks[i] = new_b
                    swaps_done += 1
            
            # If no swaps were done and no deletions happened, we might be stuck.
            # Perturb: randomly remove a block and re-add a random one? 
            # Or just break out and restart greedy? 
            # For now, if swaps_done == 0 and not changed (in deletion), we are stuck.
            # Let's add a simple perturbation: swap a random block with a random valid one, 
            # even if it temporarily increases size or breaks coverage? No, we must maintain validity.
            
            # A better perturbation: If stuck, try a "kick" by swapping a block 
            # that has unique subsets with a candidate that covers them, 
            # but we already do that. 
            # If stuck, maybe we should restart the whole local search from a slightly different state?
            # To keep it simple, we just continue the loop. The randomization in indices helps.
            
            # If we've done a lot of work without improvement, we might want to break 
            # and try a new greedy start. But the main loop handles that.

        return best_blocks

    # Run initial greedy
    blocks_idx = greedy_cover(0)
    blocks_idx = reduce_blocks(blocks_idx)
    
    best_blocks = blocks_idx
    best_len = len(blocks_idx)
    
    # Try randomized greedy if time allows
    remaining_time = deadline - time.time()
    if remaining_time > 20.0:
        seeds = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]
        for seed in seeds:
            if time.time() >= deadline - 20.0: # Reserve 20s for final LS
                break
            b_idx = greedy_cover(seed)
            b_idx = reduce_blocks(b_idx)
            if b_idx and len(b_idx) < best_len:
                best_blocks = b_idx
                best_len = len(b_idx)

    # Run Local Search on the best solution found
    if best_blocks and time.time() < deadline - 5.0:
        ls_result = local_search(best_blocks, deadline - 5.0)
        if len(ls_result) < best_len:
            best_blocks = ls_result
            best_len = len(best_blocks)

    # Final reduction pass
    if best_blocks and time.time() < deadline - 2.0:
        best_blocks = reduce_blocks(best_blocks)

    # Convert to actual blocks
    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
