import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit

    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Build coverage bitmasks for each block
    # Each block covers C(k,t) t-subsets. We pack these into a bitmask.
    # num_t_subsets = 2024 for v=24, k=7, t=3.
    # We need a bitmask of length num_t_subsets.
    # np.packbits works on bytes. 2024 bits = 253 bytes.
    # Let's use a list of numpy arrays of uint8 for bitmasks.
    
    block_bitmasks = []
    # Reverse index: t_subset_idx -> list of block indices
    t_to_blocks = [[] for _ in range(num_t_subsets)]

    # Optimization: Precompute the list of t-subset indices for each block
    # This is the bottleneck. 346k blocks * 35 subsets.
    # Let's try to vectorize this construction as much as possible or just do it fast in Python.
    
    # Actually, building the reverse index is crucial for the "focused greedy" strategy.
    # But for the vectorized greedy, we might not need the reverse index if we scan all blocks.
    # Scanning all 346k blocks with NumPy is fast.
    # Let's precompute the bitmasks.
    
    # To speed up precomputation, we can generate combinations of indices.
    # For each block (tuple of points), we need the indices of its t-subsets.
    
    # Let's store block_covered_indices as a list of lists of ints first, then convert to bitmask.
    # This might be memory heavy but 346k * 35 * 8 bytes ~ 100MB. Acceptable.
    
    # Even better: Just build the bitmasks directly.
    # For each block, initialize a zero array of uint8 of size ceil(num_t_subsets/8).
    
    mask_size = (num_t_subsets + 7) // 8
    block_bitmasks = [None] * num_blocks
    
    for i, b in enumerate(all_blocks):
        mask = np.zeros(mask_size, dtype=np.uint8)
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            byte_idx = idx // 8
            bit_idx = idx % 8
            mask[byte_idx] |= (1 << bit_idx)
        block_bitmasks[i] = mask
        
        # Build reverse index
        # This is expensive in Python. Let's see if we can skip it for the vectorized scan.
        # If we scan all blocks, we don't need reverse index.
        # Gain = popcount(mask & ~uncovered_mask)
        # We can do this for all blocks at once using NumPy?
        # Uncovered mask is a single vector. Block masks are a list of vectors.
        # We can stack them into a 2D array? 346k * 253 bytes ~ 87 MB.
        # Yes, this is feasible.
        
    # Stack all bitmasks into a 2D numpy array for vectorized operations
    # This will use ~87MB RAM.
    all_masks = np.array(block_bitmasks, dtype=np.uint8) # shape (num_blocks, mask_size)
    
    # Initial uncovered mask: all 1s
    # We need to be careful with the last byte if num_t_subsets is not a multiple of 8.
    # Let's create a valid mask that has 1s only for valid t-subset indices.
    valid_mask = np.zeros(mask_size, dtype=np.uint8)
    for i in range(num_t_subsets):
        valid_mask[i // 8] |= (1 << (i % 8))
        
    uncovered_mask = valid_mask.copy()
    
    # Greedy strategy:
    # At each step, calculate gain for all blocks:
    # gain_i = popcount(all_masks[i] & uncovered_mask)
    # Pick block with max gain.
    # Update uncovered_mask &= ~all_masks[best_idx]
    
    # To calculate popcount for all blocks at once:
    # We can use np.bitwise_and(all_masks, uncovered_mask[None, :], axis=1)
    # Then sum bits.
    # np.unpackbits gives bits in little-endian order (bit 0 is least significant).
    # We need to be careful with the order.
    # np.unpackbits(arr, count=8*mask_size, axis=1)
    # This will give an array of shape (num_blocks, 8*mask_size).
    # We sum over axis=1 to get popcount.
    
    # This operation is:
    # 1. Bitwise AND: 346k * 253 bytes ~ 87 MB operation. Fast in C.
    # 2. Unpack: 346k * 2024 bits ~ 700 MB operation. Might be slower.
    # 3. Sum: 346k * 2024 ints.
    
    # Alternative: Use np.unpackbits on the result of AND.
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple random restarts?
    # The vectorized greedy is deterministic if we always pick max gain.
    # To diversify, we can add random noise to gains or pick randomly from top N.
    
    def run_vectorized_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        unc = valid_mask.copy()
        selected = []
        
        while True:
            if time.time() > deadline:
                break
            
            # Check if uncovered is empty
            # If all bits in unc are 0, we are done.
            # We can check if unc.any() is False.
            if not unc.any():
                break
            
            # Calculate gains for all blocks
            # intersection = all_masks & unc[None, :]
            # bits = np.unpackbits(intersection, axis=1, count=8*mask_size)
            # gains = bits.sum(axis=1)
            
            # Optimization: np.unpackbits is expensive.
            # Can we use a lookup table for popcount of bytes?
            # Yes!
            # Create a lookup table for 0..255 -> popcount
            popcount_lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
            
            # intersection = np.bitwise_and(all_masks, unc[None, :])
            # gains = popcount_lut[intersection].sum(axis=1)
            
            intersection = np.bitwise_and(all_masks, unc[None, :])
            gains = popcount_lut[intersection].sum(axis=1)
            
            # Find max gain
            max_gain = gains.max()
            if max_gain == 0:
                break # Should not happen if unc is not empty and valid input
            
            # Get all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one to diversify
            best_idx = candidates[random.randint(0, len(candidates)-1)]
            
            selected.append(best_idx)
            
            # Update uncovered mask
            unc = np.bitwise_and(unc, np.bitwise_not(all_masks[best_idx]))
            
        return selected

    def try_remove_blocks(blocks):
        if not blocks:
            return blocks
        
        # Use bitmasks for redundancy check
        # cover_count is not needed if we use bitmasks?
        # A block is redundant if (OR of other blocks' masks) covers (block's mask)
        # i.e., block_mask & ~OR_others == 0
        
        # This is O(N^2 * mask_size) if done naively.
        # Better: Maintain a "covered by others" mask?
        # Or just do a simple pass:
        # For each block i, check if all bits in block_mask_i are covered by others.
        
        # Let's use the cover_count approach with bitmasks?
        # No, cover_count is per t-subset.
        # Let's stick to the bitmask OR approach for small N.
        # N is ~70-80.
        
        blocks = list(blocks)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks
                
                b_idx = blocks[i]
                b_mask = all_masks[b_idx]
                
                # Check if b_mask is covered by others
                # OR all other masks
                others_or = np.zeros(mask_size, dtype=np.uint8)
                for j in range(len(blocks)):
                    if j != i:
                        others_or = np.bitwise_or(others_or, all_masks[blocks[j]])
                
                # If b_mask & ~others_or is 0, it's redundant
                if not np.bitwise_and(b_mask, np.bitwise_not(others_or)).any():
                    blocks.pop(i)
                    changed = True
                    break # Restart loop
        
        return blocks

    # Main loop
    seeds = [42, 123, 999, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        greedy_res = run_vectorized_greedy(seed)
        if not greedy_res:
            continue
            
        processed_res = try_remove_blocks(greedy_res)
        
        if len(processed_res) < best_len:
            best_len = len(processed_res)
            best_blocks = processed_res
            
        if best_len <= 69:
            break

    # If time remains, try more random seeds
    if time.time() < deadline:
        while time.time() < deadline - 5.0:
            seed = random.randint(0, 1000000)
            greedy_res = run_vectorized_greedy(seed)
            if not greedy_res:
                continue
                
            processed_res = try_remove_blocks(greedy_res)
            
            if len(processed_res) < best_len:
                best_len = len(processed_res)
                best_blocks = processed_res
                
            if best_len <= 69:
                break

    if best_blocks is None:
        return []
        
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    
    for b in final_blocks:
        b.sort()
        
    return final_blocks
