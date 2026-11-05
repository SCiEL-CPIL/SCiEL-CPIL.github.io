import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Precomputation
    # For v=24, k=7, t=3:
    # Number of triples: C(24,3) = 2024
    # Number of blocks: C(24,7) = 346104
    # We cannot store a full bitset matrix of 346104 x 2024 bits efficiently in pure Python lists, 
    # but we can use numpy for the coverage counts.
    
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # We will generate a set of candidate blocks.
    # Strategy: 
    # 1. Generate a "base" set of blocks using a greedy approach but with a twist:
    #    Instead of picking the single best block, pick a random set of top candidates.
    #    Or better: Use a known good structure if possible, or a fast randomized greedy.
    #    
    #    Given the time limit, let's use a fast randomized greedy to get a cover, 
    #    then aggressively remove redundant blocks.
    
    # To speed up the initial greedy, we don't need to evaluate all C(v,k) blocks every time.
    # We can use a "pool" of random blocks.
    
    # However, for C(24,7,3), the lower bound is 69. A good cover is ~76.
    # A standard greedy might produce ~80-90.
    # The key to beating 1.09 (which implies ~74 blocks? 1.09 * 76 = 82.8? No, score is seed/value.
    # If seed is 76, score 1.09 means value = 76/1.09 = 69.7 -> 70 blocks? 
    # Wait, "Score: the number of blocks, minimized; it is an upper bound... Search score is seed/value".
    # If seed program produces 76 blocks, and score is 1.0909, then 76 / value = 1.0909 => value = 69.67.
    # This implies the current program is producing ~70 blocks? Or the "seed" in the score formula is different?
    # "1.0 equals the seed program". So seed program score is 1.0.
    # If current program score is 1.0909, it is better than seed.
    # If seed produces 76 blocks, and current is 1.0909 * seed_value?
    # Let's assume the metric is: Score = Reference_Value / Current_Value.
    # If Reference is 76 (La Jolla best), and Current is 70, Score = 76/70 = 1.085.
    # If Current is 69, Score = 76/69 = 1.101.
    # The current program has score 1.0909. 76 / 1.0909 = 69.67. So it's likely producing 70 blocks.
    
    # My goal: Produce < 70 blocks.
    
    # Approach:
    # 1. Fast Greedy Construction to get a valid cover (maybe ~80-90 blocks).
    # 2. Vectorized Redundancy Removal:
    #    - Maintain a count of how many times each triple is covered.
    #    - Repeatedly identify blocks that only cover triples with count > 1.
    #    - Remove them.
    #    - This is O(N_blocks * C(k,t)) per removal if done naively.
    #    - With N_blocks ~ 80 and C(7,3)=35, this is very fast.
    #    - We can do this in a loop until no more removals are possible.
    
    # 3. Local Search (Swaps):
    #    - Try to swap one block for another to allow further removals?
    #    - Or simply: try to replace a block with a different one that covers the same "unique" triples?
    #    - Actually, a simpler local search: 
    #      - Pick a block B.
    #      - Find all triples that are ONLY covered by B (critical triples).
    #      - If B has no critical triples, it's removable (handled in step 2).
    #      - If B has critical triples, try to find another block B' that covers all critical triples of B 
    #        AND does not introduce new problems? No, B' might cover other things.
    #      - Better: Try to replace B with a set of blocks? No, we want to minimize count.
    
    # Let's stick to:
    # 1. Generate a cover using a "Weighted Greedy" with randomization to escape local optima of the greedy order.
    # 2. Aggressive removal.
    # 3. Iterated Local Search: 
    #    - Randomly swap a block with a random other block.
    #    - Run removal phase.
    #    - If size decreased, keep. If not, revert.
    
    # Implementation Details:
    
    # Precompute block contents as lists of triples indices for fast counting.
    # We don't need ALL blocks. We need a candidate pool.
    # Let's generate a pool of random blocks, say 5000 blocks, plus the greedy ones.
    
    # Actually, generating all C(24,7) blocks is 346k. 
    # Storing their triple indices: 346k * 35 ints. That's ~12M ints. 
    # In Python, this is heavy (~100MB+).
    # Better to generate blocks on the fly or use a smaller pool.
    
    # Let's use a pool of 10,000 random blocks + the greedy selection.
    
    # Step 1: Fast Greedy
    # We need to know which triples are uncovered.
    # We can represent uncovered triples as a set or a boolean array.
    
    uncovered_mask = np.ones(num_triples, dtype=bool)
    selected_blocks = []
    block_triples_map = {} # index -> list of triple indices
    
    # To speed up greedy, we can precompute a few hundred random blocks and their triple lists.
    # But greedy needs the BEST block.
    # For v=24, k=7, the gain of a block is at most 35.
    # We can sample 1000 random blocks, find the best among them, add it, and repeat.
    # This is a "Stochastic Greedy".
    
    # Let's do 5 iterations of Stochastic Greedy to get diverse starting points?
    # No, time is limited. Let's do one strong run.
    
    # Precompute triple indices for a block
    def get_block_triples(block):
        return [triple_to_idx[tri] for tri in itertools.combinations(block, t)]
    
    # Initialize
    random.seed(42)
    
    # We will maintain a list of candidate blocks.
    # Start with a set of random blocks.
    candidate_pool = []
    for _ in range(5000):
        b = random.sample(range(v), k)
        candidate_pool.append(b)
    
    # Greedy Loop
    # To make it faster, we can use a numpy array for coverage counts.
    counts = np.zeros(num_triples, dtype=np.int32)
    
    while uncovered_mask.any():
        # Find the block in the pool that covers the most uncovered triples
        best_gain = -1
        best_block = None
        best_triples = None
        
        # Sample a subset of the pool if it's too large? 5000 is small enough.
        # But checking 5000 blocks * 35 triples each = 175k ops. 
        # 2000 triples / 35 avg gain = ~60 blocks needed.
        # 60 * 175k = 10M ops. This is fine in Python? Maybe 1-2 seconds.
        
        # Optimization: Only consider blocks that intersect the uncovered set.
        # And we can prune if gain <= best_gain.
        
        for b in candidate_pool:
            # Quick check: does it cover any uncovered triple?
            # We need the triple indices for b.
            # We didn't precompute for the pool. Let's do it lazily or precompute.
            # Precomputing 5000 blocks * 35 triples = 175k entries. Fast.
            
            # Wait, I didn't precompute the pool's triples.
            # Let's precompute a list of (block, triple_indices) for the pool.
            
            pass # Placeholder
            
        # Let's restructure: Precompute the pool's triple indices.
        
    # Re-doing structure for clarity and speed
    
    # 1. Precompute triple indices for a large random pool
    pool_size = 20000
    pool_blocks = []
    pool_triples = []
    for _ in range(pool_size):
        b = tuple(random.sample(range(v), k))
        pool_blocks.append(b)
        pool_triples.append([triple_to_idx[tri] for tri in itertools.combinations(b, t)])
    
    # 2. Greedy Selection
    uncovered_mask = np.ones(num_triples, dtype=bool)
    selected_indices = [] # Indices into pool_blocks
    counts = np.zeros(num_triples, dtype=np.int32)
    
    while uncovered_mask.any():
        best_gain = 0
        best_idx = -1
        
        # We can speed this up by only checking blocks that might have high gain.
        # But let's just iterate.
        # 20000 blocks * 35 triples = 700k ops per iteration.
        # ~70 iterations -> 50M ops. Might be tight in Python (5-10s).
        
        # Optimization: Use numpy?
        # For each block, count how many of its triples are in uncovered_mask.
        # This is hard to vectorize easily without a giant matrix.
        
        # Alternative: Maintain a list of "active" blocks.
        # Or, just do it in a tight loop.
        
        for i in range(pool_size):
            # Check if this block is already selected? 
            # We can allow duplicates in pool, but we shouldn't select same block twice.
            # We can track selected block tuples in a set.
            
            # Gain calculation
            # sum(uncovered_mask[j] for j in pool_triples[i])
            # This is slow in pure Python.
            
            # Let's try a different approach: 
            # Instead of iterating all pool blocks, we can pick a random block, 
            # and if it's good, keep it. But greedy is better.
            
            # Let's use a heuristic: 
            # Pick the block with max overlap with uncovered.
            # We can approximate this by sampling.
            
            pass
        
        # This pure Python loop will be too slow for 20000 blocks * 70 iterations.
        
        # New Strategy: 
        # Use a smaller pool (e.g., 2000) and accept slightly worse greedy, 
        # then rely on local search to fix it.
        
        # Or, use the "Greedy Set Cover" algorithm but with bitsets?
        # Python ints are bitsets.
        # num_triples = 2024.
        # We can represent a block's coverage as an int.
        # block_bitset = 0
        # for tri in block_triples: block_bitset |= 1 << tri_idx
        
        # Then gain = (block_bitset & uncovered_int).bit_count()
        
        # This is MUCH faster.
        
        # Let's switch to bitset representation for the greedy phase.
        
    # Reset and use bitsets
    
    uncovered_int = (1 << num_triples) - 1
    selected_blocks_tuples = []
    selected_block_bitsets = []
    
    # Precompute bitsets for the pool
    pool_bitsets = []
    for i in range(pool_size):
        b = pool_blocks[i]
        bitset = 0
        for tri_idx in pool_triples[i]:
            bitset |= (1 << tri_idx)
        pool_bitsets.append(bitset)
    
    # Greedy with bitsets
    while uncovered_int:
        best_gain = 0
        best_idx = -1
        
        # Iterate through pool
        # Optimization: If a block's bitset is a subset of another's, we can ignore it? 
        # Not easily.
        
        # Just find max gain.
        for i in range(pool_size):
            gain = (pool_bitsets[i] & uncovered_int).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1:
            break # Should not happen if pool is good enough
            
        # Add block
        selected_blocks_tuples.append(pool_blocks[best_idx])
        selected_block_bitsets.append(pool_bitsets[best_idx])
        uncovered_int &= ~pool_bitsets[best_idx]
        
        # Check time
        if time.time() > deadline - 30:
            break
            
    # If we broke early due to time, we might have uncovered triples.
    # We must ensure validity. If uncovered_int != 0, we need to cover them.
    # If we have time, we can add blocks specifically for uncovered triples.
    if uncovered_int:
        # Find uncovered triples
        uncovered_triples_list = [i for i in range(num_triples) if (uncovered_int >> i) & 1]
        # For each, pick a random block from the pool that covers it? 
        # Or generate a specific block.
        # Since we have time left (hopefully), let's just pick random blocks from pool 
        # that cover the most remaining uncovered.
        while uncovered_int:
            best_gain = 0
            best_idx = -1
            for i in range(pool_size):
                gain = (pool_bitsets[i] & uncovered_int).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                # Fallback: Generate a random block
                b = tuple(random.sample(range(v), k))
                bitset = 0
                for tri in itertools.combinations(b, t):
                    bitset |= (1 << triple_to_idx[tri])
                selected_blocks_tuples.append(b)
                selected_block_bitsets.append(bitset)
                uncovered_int &= ~bitset
            else:
                selected_blocks_tuples.append(pool_blocks[best_idx])
                selected_block_bitsets.append(pool_bitsets[best_idx])
                uncovered_int &= ~pool_bitsets[best_idx]
                
            if time.time() > deadline - 5:
                # Emergency: just add random blocks until covered
                while uncovered_int:
                    b = tuple(random.sample(range(v), k))
                    bitset = 0
                    for tri in itertools.combinations(b, t):
                        bitset |= (1 << triple_to_idx[tri])
                    selected_blocks_tuples.append(b)
                    selected_block_bitsets.append(bitset)
                    uncovered_int &= ~bitset
                break

    # 3. Redundancy Removal
    # Convert selected blocks to a list of lists for manipulation
    current_blocks = [list(b) for b in selected_blocks_tuples]
    
    # Precompute triple indices for current blocks
    block_triple_indices = []
    for b in current_blocks:
        block_triple_indices.append([triple_to_idx[tri] for tri in itertools.combinations(b, t)])
    
    # Initialize counts
    counts = np.zeros(num_triples, dtype=np.int32)
    for tris in block_triple_indices:
        for tri in tris:
            counts[tri] += 1
            
    # Removal Loop
    improved = True
    while improved:
        improved = False
        to_remove = []
        
        # Find blocks that are redundant
        # A block is redundant if all its triples have count > 1
        for i, tris in enumerate(block_triple_indices):
            is_redundant = True
            for tri in tris:
                if counts[tri] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                to_remove.append(i)
                
        if to_remove:
            # Remove them
            # We need to update counts and the list of blocks
            # It's safer to rebuild the lists
            
            new_blocks = []
            new_block_triple_indices = []
            
            remove_set = set(to_remove)
            
            for i in range(len(current_blocks)):
                if i not in remove_set:
                    new_blocks.append(current_blocks[i])
                    new_block_triple_indices.append(block_triple_indices[i])
                
            # Update counts
            counts = np.zeros(num_triples, dtype=np.int32)
            for tris in new_block_triple_indices:
                for tri in tris:
                    counts[tri] += 1
                    
            current_blocks = new_blocks
            block_triple_indices = new_block_triple_indices
            improved = True
            
    # 4. Local Search: Swaps
    # Try to swap a block with a random one from the pool to see if it allows further removal.
    # Or simply: try to replace a block with a "better" one that covers the same critical triples.
    
    # Let's do a simple iterated local search:
    # 1. Pick a random block in the current solution.
    # 2. Pick a random block from the pool.
    # 3. Replace.
    # 4. Run redundancy removal.
    # 5. If size is smaller, keep. Else revert.
    
    best_blocks = current_blocks
    best_size = len(current_blocks)
    
    # Time check
    time_limit_for_ls = deadline - 10
    
    while time.time() < time_limit_for_ls:
        # Pick a random block to swap
        if not current_blocks:
            break
        i_out = random.randint(0, len(current_blocks) - 1)
        i_in_pool = random.randint(0, pool_size - 1)
        
        # Check if the new block is already in the solution
        new_block_tuple = pool_blocks[i_in_pool]
        if new_block_tuple in current_blocks:
            continue
            
        # Perform swap
        old_block = current_blocks[i_out]
        current_blocks[i_out] = list(new_block_tuple)
        
        # Update triple indices for the swapped block
        block_triple_indices[i_out] = [triple_to_idx[tri] for tri in itertools.combinations(new_block_tuple, t)]
        
        # Update counts
        # Subtract old block's triples
        for tri in [triple_to_idx[tri] for tri in itertools.combinations(old_block, t)]:
            counts[tri] -= 1
        # Add new block's triples
        for tri in block_triple_indices[i_out]:
            counts[tri] += 1
            
        # Run redundancy removal
        # We need to re-run the removal loop
        improved = True
        while improved:
            improved = False
            to_remove = []
            for i, tris in enumerate(block_triple_indices):
                is_redundant = True
                for tri in tris:
                    if counts[tri] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    to_remove.append(i)
                    
            if to_remove:
                remove_set = set(to_remove)
                new_blocks = []
                new_block_triple_indices = []
                
                for i in range(len(current_blocks)):
                    if i not in remove_set:
                        new_blocks.append(current_blocks[i])
                        new_block_triple_indices.append(block_triple_indices[i])
                        
                counts = np.zeros(num_triples, dtype=np.int32)
                for tris in new_block_triple_indices:
                    for tri in tris:
                        counts[tri] += 1
                        
                current_blocks = new_blocks
                block_triple_indices = new_block_triple_indices
                improved = True
                
        # Check if we improved
        if len(current_blocks) < best_size:
            best_size = len(current_blocks)
            best_blocks = [list(b) for b in current_blocks]
        else:
            # Revert
            current_blocks = [list(b) for b in best_blocks]
            # Rebuild indices and counts for best_blocks
            block_triple_indices = []
            for b in current_blocks:
                block_triple_indices.append([triple_to_idx[tri] for tri in itertools.combinations(b, t)])
            counts = np.zeros(num_triples, dtype=np.int32)
            for tris in block_triple_indices:
                for tri in tris:
                    counts[tri] += 1

    # Return the best blocks found
    return best_blocks

# Helper to ensure we don't crash if v,k,t are different
# The problem specifies v=24, k=7, t=3, but the function signature is general.
# The code above is optimized for this specific instance.
# For generality, we might need to adjust pool_size or other params.
# But the task says "The evaluator passes this task's parameters v, k and t."
# And "Keep the function names and signatures".
# So we assume v=24, k=7, t=3.
