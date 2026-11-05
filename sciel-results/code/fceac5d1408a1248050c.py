import itertools
import random
import time
import heapq
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Pre-compute all t-subsets (triples)
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    
    # Map each triple to an index for bitset operations
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    
    # Pre-compute the bitset of triples covered by each potential block
    # C(24, 7) = 346,104 blocks. C(7, 3) = 35 triples per block.
    # We can represent the set of covered triples as a Python integer (bitmask)
    # or a numpy array of uint64. Python integers are fast for bitwise ops.
    
    all_blocks = []
    block_masks = []
    
    # Optimization: We don't need to store all blocks if we can generate them on the fly?
    # But for global greedy, we need to evaluate gains for many blocks.
    # Storing masks is memory efficient (346k * ~100 bytes = ~35MB).
    
    for block in itertools.combinations(range(v), k):
        mask = 0
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            mask |= (1 << idx)
        all_blocks.append(block)
        block_masks.append(mask)
        
    num_blocks = len(all_blocks)
    
    # Global Greedy Construction
    def global_greedy():
        covered_mask = 0
        selected_blocks = []
        # We need to efficiently find the block with max marginal gain.
        # Marginal gain of block B = popcount(mask[B] & ~covered_mask)
        
        # Since num_blocks is large (~350k), iterating through all of them at each step
        # is O(num_blocks * num_selected). 
        # num_selected is likely ~70-80.
        # 350k * 80 = 28M operations. Each op is bitwise AND + popcount.
        # Python big int bitwise ops are fast, but 28M might be too slow in pure Python loop.
        
        # Strategy: 
        # 1. Use numpy to vectorize the gain calculation?
        #    Store masks as numpy array of uint64? 
        #    Num triples C(24,3) = 2024. This exceeds 64 bits.
        #    So we need multiple uint64s or use Python ints.
        #    If we use Python ints, we can't easily vectorize with numpy.
        
        # Alternative: 
        # Maintain a list of (gain, block_idx). 
        # At each step, we need to update gains for all blocks? 
        # That's O(num_blocks) per step.
        # Total O(num_blocks * num_selected).
        
        # Let's try a different approach:
        # Instead of pre-computing all masks, we can use a "lazy" heap or just re-evaluate.
        # But 350k blocks is a lot.
        
        # Let's check if we can reduce the candidate pool.
        # Symmetry breaking? 
        # In a covering design, points are symmetric. 
        # We can fix the first block to be (0, 1, ..., k-1).
        # Then fix the next block to include 0,1,2 and the smallest possible new point?
        # This drastically reduces search space if we do backtracking, but for greedy?
        
        # Let's stick to Global Greedy but optimize the gain calculation.
        # We can store the masks in a list of Python ints.
        # To find max gain:
        # for i in range(num_blocks):
        #    gain = (block_masks[i] & ~covered_mask).bit_count()
        # This is a tight loop. 350k iterations.
        # Python loop overhead is significant.
        
        # Can we use numpy?
        # If we represent the mask as a list of uint64 chunks?
        # 2024 bits / 64 = 32 chunks.
        # Array shape: (346104, 32) of uint64.
        # Memory: 346104 * 32 * 8 = ~89 MB. Acceptable.
        
        # Let's pre-compute numpy array of masks.
        # This setup is done once.
        
        pass # Placeholder, logic moved to main setup

    # Re-do setup with numpy
    # Number of 64-bit words needed for 2024 bits
    num_words = (num_triples + 63) // 64
    
    masks_np = np.zeros((num_blocks, num_words), dtype=np.uint64)
    
    for i, block in enumerate(all_blocks):
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            word_idx = idx // 64
            bit_idx = idx % 64
            masks_np[i, word_idx] |= (1 << bit_idx)
            
    # Now, global greedy
    def run_greedy():
        covered = np.zeros(num_words, dtype=np.uint64)
        selected = []
        
        # To speed up, we can maintain a list of active blocks?
        # Or just iterate all.
        
        # Optimization: 
        # We can sort blocks by some heuristic initially?
        # Or just pick the best one each time.
        
        # Let's try to optimize the gain calculation using numpy.
        # gain = sum(popcount(masks_np[i] & ~covered))
        # We can compute this for all blocks at once?
        # not_covered = ~covered
        # intersected = masks_np & not_covered
        # This creates a (num_blocks, num_words) array. 
        # Then we need popcount per row.
        # Numpy doesn't have a direct popcount for arrays of uint64.
        # We can use a lookup table for 8-bit chunks?
        # Or use np.unpackbits?
        # np.unpackbits(intersected.ravel(), axis=0) -> (8*64*num_words, num_blocks)
        # This is huge: 8 * 64 * 32 * 346104 ~ 560 million elements. Too slow/memory heavy.
        
        # Alternative: 
        # Since we only pick one block at a time, we can just loop in Python.
        # But 350k * 70 = 24.5M iterations.
        # Each iteration: bitwise AND of 32 uint64s + popcount.
        # In Python, this is slow.
        
        # Let's try a hybrid:
        # Use Python integers for masks.
        # Pre-compute list of masks as Python ints.
        
        masks_py = []
        for i in range(num_blocks):
            mask = 0
            for w in range(num_words):
                mask |= (int(masks_np[i, w]) << (w * 64))
            masks_py.append(mask)
            
        covered_int = 0
        
        # To speed up the search for max gain, we can use a heap?
        # But gains change as covered_int changes.
        
        # Let's just do the linear scan in Python.
        # Is it fast enough?
        # 24.5M iterations.
        # Each: (mask & ~covered).bit_count()
        # Python big int bit_count is implemented in C and is fast.
        # The loop overhead is the main cost.
        # Let's hope it fits in 100s.
        
        # Optimization: 
        # We can prune blocks that are subsets of already selected blocks? 
        # No, blocks are all same size.
        
        # Let's try to reduce the number of blocks considered.
        # Symmetry: 
        # The problem is symmetric under S_24.
        # We can fix the first few blocks.
        # Block 1: {0, 1, 2, 3, 4, 5, 6}
        # Block 2: Must cover triples not covered by B1.
        # Actually, global greedy doesn't easily allow fixing blocks unless we restrict the pool.
        
        # Let's just run the global greedy on all blocks and see.
        
        # To make it faster, we can use a "candidate pool" that shrinks?
        # No, any block can be useful.
        
        # Let's implement the Python int version.
        
        while True:
            if time.time() > deadline - 5:
                break
                
            best_gain = -1
            best_idx = -1
            
            # We can skip blocks that have 0 gain?
            # We need to find the one with max gain.
            
            # To speed up, we can randomize the order or use a heuristic?
            # For now, linear scan.
            
            # Optimization: 
            # If we find a gain of C(k,t) = 35, it's perfect. We can break early?
            # Max possible gain is 35.
            
            for i in range(num_blocks):
                # Quick check: if mask is subset of covered, gain is 0
                # (masks_py[i] | covered_int) == covered_int
                # But we need max gain.
                
                gain = (masks_py[i] & ~covered_int).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    if best_gain == 35: # Max possible
                        break
                        
            if best_idx == -1 or best_gain == 0:
                break # No more coverage possible? Should not happen if valid.
                
            selected.append(all_blocks[best_idx])
            covered_int |= masks_py[best_idx]
            
            # Check if all covered
            if (covered_int | (~masks_py[0] & ((1 << num_triples) - 1))) == ((1 << num_triples) - 1):
                 # This check is wrong. 
                 # Check if covered_int has all bits set.
                 pass
            
            # Simpler check:
            # if covered_int.bit_count() == num_triples:
            #    break
            # But bit_count on huge int is fast.
            if covered_int.bit_count() == num_triples:
                break
                
        return selected

    # Local Search: Swap-Add
    def local_search(blocks):
        # Try to swap one block for another, or add a block and remove two?
        # Simple: Try to replace a block B with a block B' such that 
        # coverage is maintained and we don't increase size (so size stays same, but maybe allows removal later?)
        # Actually, standard local search for set cover:
        # 1. Try to remove a block (if redundant).
        # 2. Try to swap a block B for B' where B' covers everything B covers (plus maybe more) 
        #    and allows removal of another block?
        
        # Let's stick to the "remove redundant" pass first, which is fast.
        # Then try "swap": Replace block i with block j if:
        # - Block j is not in the set
        # - All triples in block i are covered by (Set - {i} + {j})
        # This doesn't reduce size, but might lead to a state where a block becomes redundant.
        
        # Better: "Add-Swap-Remove"
        # 1. Add a new block j.
        # 2. Remove two blocks i, k that are now redundant.
        # Net change: -1 block.
        
        # This is expensive to search.
        
        # Let's implement a simpler "Greedy Removal" first, which is O(N^2 * 35).
        # Then a limited "Swap" search.
        
        def build_counts(blocks):
            counts = {}
            for b in blocks:
                for tri in itertools.combinations(b, t):
                    counts[tri] = counts.get(tri, 0) + 1
            return counts

        def optimize(blocks):
            counts = build_counts(blocks)
            active = [True] * len(blocks)
            changed = True
            while changed:
                changed = False
                indices = list(range(len(blocks)))
                random.shuffle(indices)
                for i in indices:
                    if not active[i]:
                        continue
                    b = blocks[i]
                    can_remove = True
                    for tri in itertools.combinations(b, t):
                        if counts[tri] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        active[i] = False
                        changed = True
                        for tri in itertools.combinations(b, t):
                            counts[tri] -= 1
            return [blocks[i] for i in range(len(blocks)) if active[i]]

        return optimize(blocks)

    # Main Loop
    best_blocks = None
    best_len = float('inf')
    
    # Run global greedy once (it's deterministic if we don't randomize tie-breaking, 
    # but we can randomize tie-breaking to get different solutions)
    
    # To allow randomization in global greedy, we need to modify the selection.
    # If multiple blocks have the same max gain, pick one randomly.
    
    def run_greedy_randomized():
        covered_int = 0
        selected = []
        
        while True:
            if time.time() > deadline - 5:
                break
                
            best_gain = -1
            best_indices = []
            
            for i in range(num_blocks):
                gain = (masks_py[i] & ~covered_int).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                    if best_gain == 35:
                        break
                elif gain == best_gain:
                    best_indices.append(i)
                    
            if best_idx == -1 or best_gain == 0:
                break
                
            # Pick randomly from best_indices
            idx = random.choice(best_indices)
            selected.append(all_blocks[idx])
            covered_int |= masks_py[idx]
            
            if covered_int.bit_count() == num_triples:
                break
                
        return selected

    # The linear scan is likely too slow for multiple restarts within 110s.
    # 350k blocks * 80 steps = 28M iterations.
    # In Python, 28M simple iterations might take ~5-10 seconds.
    # So we can do ~10-20 restarts.
    
    # Let's optimize the gain calculation.
    # We can pre-compute the list of blocks sorted by some static property?
    # No, gain is dynamic.
    
    # Let's try to speed up the loop using a list of masks and local variables.
    
    def fast_greedy():
        covered_int = 0
        selected = []
        masks = masks_py
        blocks = all_blocks
        n = num_blocks
        max_gain_possible = 35
        full_mask = (1 << num_triples) - 1
        
        while covered_int != full_mask:
            if time.time() > deadline - 2:
                break
                
            best_gain = -1
            best_indices = []
            
            # Use a for loop
            for i in range(n):
                gain = (masks[i] & ~covered_int).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                    if best_gain == max_gain_possible:
                        break
                elif gain == best_gain:
                    best_indices.append(i)
                    
            if best_gain == 0:
                break # Stuck?
                
            idx = random.choice(best_indices)
            selected.append(blocks[idx])
            covered_int |= masks[idx]
            
        return selected

    # Run multiple restarts
    while time.time() < deadline:
        cand = fast_greedy()
        if not cand:
            continue
            
        opt_cand = local_search(cand)
        
        if len(opt_cand) < best_len:
            best_len = len(opt_cand)
            best_blocks = opt_cand
            
        # If we hit the lower bound, stop
        if best_len <= 69:
            break
            
    return best_blocks
