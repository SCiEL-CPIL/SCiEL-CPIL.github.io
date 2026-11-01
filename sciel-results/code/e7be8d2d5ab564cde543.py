import itertools
import time
import random
import sys

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    
    # Precomputation
    # 1. Generate all t-subsets and map them to indices
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    
    # Bitset for uncovered subsets
    # Python int can hold arbitrary size integers, so 2024 bits is fine.
    # Initialize with all bits set
    uncovered = (1 << n_subsets) - 1
    
    # Precompute for fast gain calculation:
    # We need to quickly check if a specific triple is uncovered.
    # We also need to efficiently find triples covered by a block.
    
    # Helper to check if a specific subset (tuple) is uncovered
    def is_uncovered(sub):
        idx = subset_to_idx[sub]
        return (uncovered >> idx) & 1
        
    # Helper to mark a subset as covered
    def mark_covered(sub):
        nonlocal uncovered
        idx = subset_to_idx[sub]
        uncovered &= ~(1 << idx)

    best_blocks = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    # Greedy construction function
    def greedy_construct(seed_rng):
        blocks = []
        # Local copy of uncovered state for this run? 
        # No, we want to modify the global uncovered state? 
        # It's better to make a local copy for each restart to allow parallelism or clean restarts.
        # But copying a 2024-bit int is fast.
        local_uncovered = (1 << n_subsets) - 1
        
        def local_is_uncovered(sub):
            idx = subset_to_idx[sub]
            return (local_uncovered >> idx) & 1
            
        def local_mark_covered(sub):
            nonlocal local_uncovered
            idx = subset_to_idx[sub]
            local_uncovered &= ~(1 << idx)

        while local_uncovered != 0:
            if time.time() > deadline:
                return None # Timeout
            
            # Select a seed subset from uncovered
            # Randomly pick an uncovered subset to start the block
            # Finding a random set bit in a large int:
            # We can iterate through set bits or just pick a random index and check.
            # Given 2024 subsets, picking random index and checking is O(1) avg.
            seed_idx = -1
            while seed_idx == -1:
                candidate = seed_rng.randint(0, n_subsets - 1)
                if (local_uncovered >> candidate) & 1:
                    seed_idx = candidate
            
            seed_sub = all_subsets[seed_idx]
            block = list(seed_sub)
            
            # Greedily add points
            while len(block) < k:
                block_set = set(block)
                best_point = None
                best_gain = -1
                
                # Candidates are all points not in block
                # To speed up, we can restrict candidates to points that appear in many uncovered subsets?
                # For now, iterate all v points.
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    # Gain is number of uncovered triples T such that x in T and T\{x} subset of block
                    # T has size t=3. So T = {x, y, z}. We need {y, z} subset of block.
                    # Iterate over pairs in block
                    # block has size m. Pairs are combinations of m elements taken 2 at a time.
                    for y, z in itertools.combinations(block, t - 1):
                        # Check if {x, y, z} is a valid triple (distinct elements)
                        if x == y or x == z:
                            continue
                        # y and z are distinct from each other because they come from combinations
                        # Check if this triple is uncovered
                        tri = tuple(sorted((x, y, z)))
                        if local_is_uncovered(tri):
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        # Optional: early break if gain is max possible?
                        # Max gain for a point is limited by number of pairs in block.
                        # If best_gain == len(list(itertools.combinations(block, t-1))), we can break?
                        # Not exactly, because some pairs might form covered triples.
                        
                if best_point is None or best_gain <= 0:
                    # No point improves coverage significantly, or no point available
                    # Fill with arbitrary points if needed to reach size k
                    # But if best_gain <= 0, we are stuck?
                    # Actually, if we picked a seed, there must be points to complete it.
                    # If gain is 0, it means no new triples are covered by adding any point.
                    # This can happen if the block is already "saturated" with respect to uncovered sets?
                    # Unlikely if we start from an uncovered seed.
                    # If we can't find a point with gain > 0, we pick one randomly to avoid infinite loop
                    available = [x for x in range(v) if x not in block_set]
                    if available:
                        best_point = seed_rng.choice(available)
                    else:
                        break
                
                block.append(best_point)
                
            # Ensure block size is k
            if len(block) < k:
                block_set = set(block)
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block_set:
                        block.append(x)
                        
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                local_mark_covered(sub)
                
        return blocks

    # Run multiple greedy restarts
    num_restarts = 0
    while time.time() < deadline - 20: # Leave time for post-processing
        seed_rng = random.Random(num_restarts)
        blocks = greedy_construct(seed_rng)
        if blocks is None:
            continue
            
        # Post-processing: Remove redundant blocks
        # A block is redundant if every t-subset it contains is covered by another block.
        # We need to know which blocks cover which subsets.
        
        # Efficient redundancy removal:
        # 1. Count how many blocks cover each subset.
        # 2. If count <= 1, that subset is "critical" for that block.
        # 3. If a block has no critical subsets, it is redundant.
        
        # To do this efficiently, we can iterate.
        # Since we have many restarts, we should only do this on promising solutions.
        
        if len(blocks) < best_count:
            # Try to prune
            pruned_blocks = list(blocks)
            
            # Build coverage map: subset_index -> set of block_indices
            # This might be memory heavy if we do it for every restart.
            # Alternative: For each block, check if it's redundant against current list.
            
            changed = True
            while changed and time.time() < deadline - 10:
                changed = False
                for i in range(len(pruned_blocks)):
                    # Check if block i is redundant
                    redundant = True
                    block_i = pruned_blocks[i]
                    for sub in itertools.combinations(block_i, t):
                        idx = subset_to_idx[sub]
                        # Check if any other block j covers sub
                        covered_by_other = False
                        for j in range(len(pruned_blocks)):
                            if i == j:
                                continue
                            if all(x in pruned_blocks[j] for x in sub):
                                covered_by_other = True
                                break
                        if not covered_by_other:
                            redundant = False
                            break
                    
                    if redundant:
                        pruned_blocks.pop(i)
                        changed = True
                        break # Restart scan
            
            if len(pruned_blocks) < best_count:
                best_count = len(pruned_blocks)
                best_blocks = pruned_blocks
                
        num_restarts += 1
        
    if best_blocks is None:
        # Fallback: return a valid covering even if not optimal
        # Just run one more greedy if we didn't finish any
        if time.time() < deadline - 5:
            seed_rng = random.Random(999)
            blocks = greedy_construct(seed_rng)
            if blocks:
                return blocks
        return []
        
    return best_blocks
