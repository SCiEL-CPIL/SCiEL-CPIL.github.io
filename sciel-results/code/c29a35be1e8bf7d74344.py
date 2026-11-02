<<<<<<< SEARCH
import time
import random
import itertools
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    if v == 24 and k == 7 and t == 3:
        return construct_24_7_3(deadline)
    else:
        return construct_generic(v, k, t, deadline)

def construct_24_7_3(deadline):
    v, k, t = 24, 7, 3
    points = range(v)
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Precompute blocks and their masks
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Masks for triples covered by each block
    # Using Python ints for bitmasks
    block_masks = [0] * num_blocks
    block_triple_lists = [[] for _ in range(num_blocks)]
    
    # Precomputation
    for i, block in enumerate(all_blocks):
        mask = 0
        t_list = []
        # Optimization: precompute triples for this block?
        # itertools.combinations(block, t)
        for triple in itertools.combinations(block, t):
            idx = tsub_to_idx[triple]
            mask |= (1 << idx)
            t_list.append(idx)
        block_masks[i] = mask
        block_triple_lists[i] = t_list
        
    if time.time() > deadline - 30:
        # If precomp took too long (unlikely), fallback to simple greedy
        pass

    def greedy_construct(randomize=False):
        uncovered_mask = (1 << num_tsubs) - 1
        blocks = []
        
        while uncovered_mask:
            best_idx = -1
            best_cov = -1
            
            # Scan all blocks
            for i in range(num_blocks):
                m = block_masks[i]
                # Count overlapping bits
                # In Python 3.10+, int.bit_count() is available
                cov = (m & uncovered_mask).bit_count()
                
                # Randomize tie-breaking if needed
                if randomize:
                    cov += random.random() * 0.1 # Small noise
                
                if cov > best_cov:
                    best_cov = cov
                    best_idx = i
            
            if best_idx == -1:
                break
                
            blocks.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
            
            if time.time() > deadline - 10:
                break
                
        return blocks

    def refine(blocks):
        # blocks is list of indices
        if not blocks:
            return []
            
        # Count coverage
        counts = [0] * num_tsubs
        for b_idx in blocks:
            for t_idx in block_triple_lists[b_idx]:
                counts[t_idx] += 1
                
        changed = True
        while changed:
            changed = False
            # Iterate backwards to remove
            for i in range(len(blocks) - 1, -1, -1):
                b_idx = blocks[i]
                redundant = True
                for t_idx in block_triple_lists[b_idx]:
                    if counts[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block
                    for t_idx in block_triple_lists[b_idx]:
                        counts[t_idx] -= 1
                    blocks.pop(i)
                    changed = True
                    
                    if time.time() > deadline - 5:
                        return blocks
        return blocks

    # Run 1: Deterministic
    blocks_idx = greedy_construct(randomize=False)
    blocks_idx = refine(blocks_idx)
    
    best_blocks = [list(all_blocks[i]) for i in blocks_idx]
    
    # Run 2: Randomized if time permits
    if time.time() < deadline - 15:
        for _ in range(3): # A few restarts
            if time.time() > deadline - 5:
                break
            blocks_idx_r = greedy_construct(randomize=True)
            blocks_idx_r = refine(blocks_idx_r)
            if len(blocks_idx_r) < len(blocks_idx):
                blocks_idx = blocks_idx_r
                best_blocks = [list(all_blocks[i]) for i in blocks_idx]
                
    return best_blocks
=======
import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    if v == 24 and k == 7 and t == 3:
        return construct_24_7_3(deadline)
    else:
        return construct_generic(v, k, t, deadline)

def construct_24_7_3(deadline):
    v, k, t = 24, 7, 3
    points = range(v)
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Precompute blocks and their masks
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Masks for triples covered by each block
    # Using Python ints for bitmasks
    block_masks = [0] * num_blocks
    block_triple_lists = [[] for _ in range(num_blocks)]
    
    # Precomputation
    for i, block in enumerate(all_blocks):
        mask = 0
        t_list = []
        for triple in itertools.combinations(block, t):
            idx = tsub_to_idx[triple]
            mask |= (1 << idx)
            t_list.append(idx)
        block_masks[i] = mask
        block_triple_lists[i] = t_list

    # Convert to numpy uint64 array for vectorized operations
    # Each mask fits in a Python int; for 2024 triples we need multiple uint64 words
    # But we can use numpy's uint64 with a different approach:
    # Actually, let's use a list of uint64 chunks. 2024 bits = 32 uint64 words.
    # Alternative: use numpy's object array of Python ints and do vectorized ops.
    # Better: split into chunks of 64 bits and use uint64 arrays.
    
    num_words = (num_tsubs + 63) // 64  # 2024/64 = 31.625 -> 32
    # masks_np shape: (num_blocks, num_words) uint64
    masks_np = np.zeros((num_blocks, num_words), dtype=np.uint64)
    for i in range(num_blocks):
        m = block_masks[i]
        for w in range(num_words):
            masks_np[i, w] = (m >> (64 * w)) & 0xFFFFFFFFFFFFFFFF
    
    # Precompute for each block: which words it touches (non-zero)
    # For counting overlaps: sum of popcount of (masks_np & uncovered_np)
    # numpy doesn't have native popcount, but we can use lookup or bit tricks.
    # Alternative: use a lookup table for 8-bit popcount.
    
    # Build a lookup table for popcount of bytes
    popcount_table = np.array([bin(x).count('1') for x in range(256)], dtype=np.uint8)
    
    def popcount_u64(arr):
        """Compute popcount for a numpy array of uint64."""
        # Split into bytes
        result = np.zeros(arr.shape, dtype=np.int32)
        for shift in range(0, 64, 8):
            byte = (arr >> shift) & 0xFF
            result += popcount_table[byte.astype(np.int32)]
        return result
    
    def greedy_construct_vectorized(randomize=False, seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        # uncovered as numpy uint64 array
        uncovered_np = np.zeros(num_words, dtype=np.uint64)
        for w in range(num_words):
            uncovered_np[w] = (0xFFFFFFFFFFFFFFFF << (64 * w)) & 0xFFFFFFFFFFFFFFFF
        # Clear bits beyond num_tsubs in the last word
        if num_tsubs % 64 != 0:
            valid_bits = num_tsubs % 64
            uncovered_np[num_words - 1] &= (1 << valid_bits) - 1
        
        blocks = []
        selected_set = set()
        
        while True:
            # Check if uncovered is all zeros
            if not np.any(uncovered_np):
                break
            
            # Compute coverage: for each block, count bits in (masks_np & uncovered_np)
            # Vectorized: AND then popcount
            overlap = masks_np & uncovered_np[np.newaxis, :]
            # Popcount per block
            # This is the expensive part - do it efficiently
            cov_counts = np.zeros(num_blocks, dtype=np.int32)
            for w in range(num_words):
                byte_arr = overlap[:, w]
                for shift in range(0, 64, 8):
                    byte = (byte_arr >> shift) & 0xFF
                    cov_counts += popcount_table[byte.astype(np.int32)]
            
            # Filter to blocks not already selected
            for i in range(num_blocks):
                if i in selected_set:
                    cov_counts[i] = -1
            
            if randomize:
                # Add small random noise to break ties
                noise = rng.rand(num_blocks).astype(np.float64) * 0.5
                cov_float = cov_counts.astype(np.float64)
                cov_float[cov_counts < 0] = -1e9
                cov_float += noise
                best_idx = int(np.argmax(cov_float))
            else:
                best_idx = int(np.argmax(cov_counts))
            
            if cov_counts[best_idx] <= 0:
                break
            
            blocks.append(best_idx)
            selected_set.add(best_idx)
            uncovered_np &= ~masks_np[best_idx]
            
            if time.time() > deadline - 10:
                break
        
        return blocks

    def refine(blocks):
        # blocks is list of indices
        if not blocks:
            return []
            
        # Count coverage
        counts = [0] * num_tsubs
        for b_idx in blocks:
            for t_idx in block_triple_lists[b_idx]:
                counts[t_idx] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                b_idx = blocks[i]
                redundant = True
                for t_idx in block_triple_lists[b_idx]:
                    if counts[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for t_idx in block_triple_lists[b_idx]:
                        counts[t_idx] -= 1
                    blocks.pop(i)
                    changed = True
                    
                    if time.time() > deadline - 5:
                        return blocks
        return blocks

    def local_search_swap(blocks):
        """Try to improve by swapping blocks."""
        if not blocks:
            return blocks
        
        # Build coverage counts
        counts = [0] * num_tsubs
        for b_idx in blocks:
            for t_idx in block_triple_lists[b_idx]:
                counts[t_idx] += 1
        
        # For each block, find triples it uniquely covers
        # Try replacing a block with one that covers more uncovered-elsewhere triples
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                b_idx = blocks[i]
                # Triples covered only by this block
                unique_triples = [t_idx for t_idx in block_triple_lists[b_idx] if counts[t_idx] == 1]
                if not unique_triples:
                    continue
                
                # Find best replacement
                best_repl = -1
                best_gain = -1
                for j in range(num_blocks):
                    if j == b_idx or j in blocks:
                        continue
                    # Gain: how many of unique_triples does block j cover?
                    # Plus: how many new triples does j cover that aren't covered by others?
                    # Simplified: just count overlap with unique_triples
                    gain = 0
                    for t_idx in unique_triples:
                        if t_idx in block_triple_lists[j]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_repl = j
                
                if best_repl != -1 and best_gain > 0:
                    # Do the swap
                    for t_idx in block_triple_lists[b_idx]:
                        counts[t_idx] -= 1
                    for t_idx in block_triple_lists[best_repl]:
                        counts[t_idx] += 1
                    blocks[i] = best_repl
                    improved = True
        
        return blocks

    # Run 1: Deterministic
    blocks_idx = greedy_construct_vectorized(randomize=False)
    blocks_idx = refine(blocks_idx)
    
    best_blocks = [list(all_blocks[i]) for i in blocks_idx]
    best_len = len(blocks_idx)
    
    # Run multiple randomized restarts
    if time.time() < deadline - 15:
        rng_seed = random.randint(0, 2**31 - 1)
        num_restarts = 30
        for r in range(num_restarts):
            if time.time() > deadline - 8:
                break
            seed = rng_seed + r * 7919
            blocks_idx_r = greedy_construct_vectorized(randomize=True, seed=seed)
            blocks_idx_r = refine(blocks_idx_r)
            if len(blocks_idx_r) < best_len:
                best_len = len(blocks_idx_r)
                blocks_idx = blocks_idx_r
                best_blocks = [list(all_blocks[i]) for i in blocks_idx]
    
    # Local search on best solution if time permits
    if time.time() < deadline - 10 and best_blocks:
        # Convert back to indices for local search
        block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
        best_idx_list = [block_to_idx[tuple(b)] for b in best_blocks]
        improved_idx = local_search_swap(best_idx_list)
        if len(improved_idx) < len(best_idx_list):
            best_blocks = [list(all_blocks[i]) for i in improved_idx]
    
    return best_blocks
>>>>>>> REPLACE

Now let me also improve the generic fallback slightly:

