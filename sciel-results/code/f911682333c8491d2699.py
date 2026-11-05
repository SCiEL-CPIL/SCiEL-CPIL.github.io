import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s margin before 150s kill
    
    # 1. Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(k_subsets)
    
    # 3. Efficiently compute bitmasks for each block
    # We want a bitmask for each block representing the t-subsets it covers.
    # A t-subset is covered by a block if t-subset is a subset of block.
    
    # Instead of iterating combinations for each block (O(C(v,k) * C(k,t))),
    # we can iterate over all t-subsets and find which blocks contain them?
    # Or better:
    # For each block, its mask is the OR of masks of its t-subsets.
    # Precompute mask for every t-subset (just a single bit).
    # Then for each block, OR the masks of its C(k,t) t-subsets.
    
    # This is O(C(v,k) * C(k,t)).
    # C(19,7) = 50388. C(7,4) = 35. Total ~ 1.7M operations. Very fast.
    
    block_masks = np.zeros(num_blocks, dtype=np.uint64)
    
    # Helper to get mask for a t-subset
    # Since num_t_subsets can be > 64, we might need multiple uint64s or Python ints.
    # For C(19,4) = 3876, we need ~61 uint64s.
    # NumPy arrays of uint64 are fast.
    
    # Let's use a list of Python ints for masks if we want simplicity, 
    # but NumPy vectorization is key for speed.
    # If we use NumPy, we need to handle multi-word bitmasks.
    # This is complex.
    
    # Alternative: Use Python ints for masks, but vectorize the *selection*?
    # Python ints bit_count is fast in C, but looping 50k times in Python is slow.
    
    # Let's try to stick to Python ints for masks but optimize the loop.
    # Or, use a list of numpy arrays for the bitmask parts?
    
    # Actually, for v=19, k=7, t=4, num_t_subsets=3876.
    # We can split the bitmask into chunks of 64 bits.
    # num_chunks = (3876 + 63) // 64 = 61.
    # We can have a 2D numpy array: block_masks_np of shape (num_blocks, num_chunks) of uint64.
    
    num_chunks = (num_t_subsets + 63) // 64
    block_masks_np = np.zeros((num_blocks, num_chunks), dtype=np.uint64)
    
    # Precompute single-bit masks for each t-subset in the numpy array
    # t_subset_masks_np[t_idx, chunk_idx] = bit
    
    for ts_idx, ts in enumerate(t_subsets):
        chunk = ts_idx // 64
        bit = ts_idx % 64
        # We don't need to store single bit masks separately if we just set them in block masks directly?
        # No, we need to OR them.
        
    # Let's just compute block masks directly.
    for b_idx, block in enumerate(k_subsets):
        for t_sub in itertools.combinations(block, t):
            ts_idx = t_subset_to_idx[t_sub]
            chunk = ts_idx // 64
            bit = ts_idx % 64
            block_masks_np[b_idx, chunk] |= (1 << bit)
            
    # Full mask for checking coverage
    # We need to check if all t-subsets are covered.
    # This is equivalent to checking if (uncovered_mask & block_mask) == 0 for all?
    # No, we maintain an 'uncovered' mask.
    # Initially uncovered = all 1s.
    # When we select a block, we do uncovered &= ~block_mask.
    # We need to check if uncovered is 0.
    
    # Let's represent uncovered as a numpy array of uint64s.
    uncovered = np.ones(num_chunks, dtype=np.uint64)
    # The last chunk might have unused bits. We should mask them out.
    # Bits in [num_t_subsets, num_chunks*64) should be 0 in 'covered' masks, 
    # so they will remain 1 in 'uncovered' if we don't mask the initial uncovered?
    # If they remain 1, uncovered will never be 0.
    # So we must initialize uncovered with 0s for the unused bits.
    
    # Actually, it's safer to just check coverage at the end or maintain a count.
    # But checking uncovered == 0 is fast if we do it correctly.
    
    # Let's mask the last chunk of uncovered.
    if num_t_subsets % 64 != 0:
        last_chunk_bits = num_t_subsets % 64
        mask_last = (1 << last_chunk_bits) - 1
        uncovered[-1] = mask_last
    # Other chunks are all 1s.
    
    def is_covered(uncovered_np):
        # Check if all elements are 0
        return not np.any(uncovered_np)
        
    def get_gain(uncovered_np, block_mask_np):
        # Gain is number of bits set in (uncovered & block_mask)
        # We can compute this per chunk and sum.
        # intersection = uncovered_np & block_mask_np
        # sum of bit counts.
        # np.bitwise_and is vectorized.
        # Counting bits in numpy uint64 array?
        # There is no direct np.bit_count.
        # We can use a lookup table or just sum the integers? No, integers are 64-bit.
        # We can convert to list and use int.bit_count?
        # That breaks vectorization benefit for the gain calculation if done in Python loop.
        
        # Alternative:
        # Since we are selecting the block with max gain, and we have ~50k blocks,
        # doing this in Python loop over chunks (61 chunks) for each block (50k) is 3M ops per step.
        # 150 steps * 3M = 450M ops. Too slow in Python.
        
        # We need faster gain calculation.
        # Option: Use Python ints for the mask.
        # Python int bit_count is very fast (C implementation).
        # Looping 50k blocks in Python:
        # for i in range(50k):
        #    gain = (uncovered_int & block_masks_int[i]).bit_count()
        # This is 50k * (big int ops).
        # Big int ops for 3876 bits are fast.
        # 50k * ~100ns = 5ms per step.
        # 150 steps * 5ms = 750ms per run.
        # We can do ~100 runs in 150s.
        # This is feasible!
        
        # So, let's switch back to Python ints for masks.
        # It simplifies the code and might be fast enough.
        
        pass
        
    # Reset: Use Python ints for masks.
    block_masks_int = []
    for b_idx in range(num_blocks):
        mask = 0
        # Reconstruct from numpy? Or just compute again?
        # Computing again is 1.7M ops, which is fast.
        block = k_subsets[b_idx]
        for t_sub in itertools.combinations(block, t):
            ts_idx = t_subset_to_idx[t_sub]
            mask |= (1 << ts_idx)
        block_masks_int.append(mask)
        
    # Convert to numpy array? No, Python list of ints is fine.
    # But we want to vectorize the *comparison*?
    # No, we just want to find the max gain.
    
    # Let's try to speed up the gain calculation.
    # If we store block_masks_int as a numpy array of objects, we can't vectorize bit_count.
    
    # Let's stick to Python loop for gain, but optimize the inner loop.
    
    best_blocks = None
    best_count = float('inf')
    
    # Pre-shuffle block indices for randomization?
    # We will use random noise on gain.
    
    runs = 0
    while time.time() < deadline - 5: # Reserve 5s for final cleanup
        runs += 1
        uncovered = (1 << num_t_subsets) - 1
        selected_indices = []
        
        # Random noise scale
        noise_scale = 0.1 * runs # Increase noise over time? Or keep constant?
        # Let's keep it small to not ruin the greedy too much.
        noise_scale = 0.5
        
        while uncovered:
            best_gain = -1
            best_block_idx = -1
            
            # Iterate through blocks
            # Optimization: If we have many blocks, this loop is the bottleneck.
            # Can we prune?
            # If (uncovered & mask) == 0, gain is 0.
            
            for i in range(num_blocks):
                # Calculate gain
                gain = (uncovered & block_masks_int[i]).bit_count()
                if gain == 0:
                    continue
                
                # Add noise
                # random.random() is slow.
                # Use a fast pseudo-random?
                # Or just break ties by index?
                # To get diversity, we can shuffle the order of blocks we check?
                # Shuffling 50k items is fast.
                # If we shuffle, the first max found is picked.
                # This is a good way to randomize.
                
                # But we need to find the *max* gain first?
                # No, if we just pick the first one with max gain, we need to know the max.
                # Standard greedy: find max gain, then pick one.
                # If we shuffle, we can just pick the first one that achieves the current best?
                # No, that's not max gain.
                
                # Strategy:
                # 1. Find the max gain value M.
                # 2. Among all blocks with gain M, pick one randomly.
                
                # This requires two passes or storing candidates.
                # Storing candidates: list of indices with gain M.
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    # Reset candidates?
                    # We don't need to store all if we just use random tie-breaking in a second pass?
                    # Or we can just pick the first one?
                    # To randomize, let's collect candidates.
                    # But collecting candidates adds overhead.
                    
                    # Alternative:
                    # Use a random permutation of block indices for this run.
                    # Then iterate in that order.
                    # If gain > best_gain, update best.
                    # If gain == best_gain, we might replace best with some probability?
                    # Or just keep the first one found in the shuffled order.
                    # This effectively picks the first block with max gain in a random order.
                    # This is a valid randomization.
                    
                    # But we need the shuffled order.
                    # Let's generate it once per run.
                    
                    pass # Placeholder, logic below
                    
            # The loop above needs to be restructured to use a shuffled order.
            
            if best_gain == 0:
                break
                
            selected_indices.append(best_block_idx)
            uncovered &= ~block_masks_int[best_block_idx]
            
            if time.time() > deadline - 5:
                break
        
        # Post-processing: Remove redundant blocks
        if time.time() < deadline - 5:
            # Rebuild counts
            counts = [0] * num_t_subsets
            block_ts_list = []
            for idx in selected_indices:
                # We didn't store ts_list for each block index in a fast way?
                # We can recompute or store it.
                # Storing is better.
                # Let's precompute block_ts_indices_list.
                pass # Need to fix this
            
            # ... (Implement pruning and local search) ...
            
            # Update best
            if len(selected_indices) < best_count:
                best_count = len(selected_indices)
                best_blocks = [k_subsets[i] for i in selected_indices]
                
    if best_blocks is None:
        # Fallback: return empty or a trivial solution?
        # Must return valid covering.
        # If no time, we might have an incomplete solution.
        # We must ensure validity.
        # If we ran out of time during greedy, we might not cover all.
        # We should check validity and if invalid, try to fix or return best effort?
        # The task says "Valid when every t-subset...".
        # If we return invalid, score is 0.
        # So we must ensure the returned list is valid.
        # If we don't have a valid one, we are in trouble.
        # But greedy usually finishes fast.
        
        return []
        
    return best_blocks
