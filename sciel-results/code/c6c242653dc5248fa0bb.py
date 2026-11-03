"""Rewritten program for C(19,7,4) covering design using bitset-based randomized greedy with post-hoc redundancy removal."""
import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin before 120s kill
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each k-subset (block), precompute which t-subsets it covers
    # But C(19,7) = 50388, and each covers C(7,4)=35 t-subsets
    # Total: 50388 * 35 = 1,763,580 entries - manageable
    
    # Precompute coverage bitsets for all blocks
    # A block covers a t-subset if the t-subset is contained in the block
    # For efficiency, use numpy bitsets
    
    # For each point, precompute which t-subsets contain it
    # Actually, let's think differently:
    # For a partial block (set of points), the t-subsets it can still cover are those 
    # that are subsets of the block and not yet covered.
    
    # Better approach: represent coverage as a bit array of length n_tsubs
    # For each candidate block, compute how many uncovered t-subsets it covers
    
    # Precompute for each t-subset its elements as a frozenset for fast subset checks
    # Actually, for a block B (list of k points), a t-subset S is in B iff S ⊆ B
    
    # Let's use a different representation:
    # For each point x, precompute the list of t-subset indices that contain x
    point_tsubs = [[] for _ in range(v)]
    for idx, ts in enumerate(all_tsubs):
        for x in ts:
            point_tsubs[x].append(idx)
    
    # For a block B, the t-subsets it covers are those t-subsets S where S ⊆ B
    # We can compute this by: for each t-subset, check if all its elements are in B
    # But that's O(n_tsubs * t) per block which is 3876 * 4 = 15504 ops - fine
    
    # Better: precompute for each t-subset a bitmask of which points it uses
    # Then a block B covers t-subset S iff mask[S] & ~mask[B] == 0
    
    # Since v=19, we can use integers as bitmasks
    tsub_masks = [0] * n_tsubs
    for idx, ts in enumerate(all_tsubs):
        m = 0
        for x in ts:
            m |= (1 << x)
        tsub_masks[idx] = m
    
    # For a block B with bitmask bm, t-subset idx is covered iff tsub_masks[idx] & ~bm == 0
    
    def compute_block_coverage(block, uncovered_mask):
        """Return the count of uncovered t-subsets covered by this block."""
        bm = 0
        for x in block:
            bm |= (1 << x)
        # Count how many uncovered t-subsets are subsets of block
        count = 0
        for idx in range(n_tsubs):
            if uncovered_mask[idx] and (tsub_masks[idx] & ~bm) == 0:
                count += 1
        return count
    
    def compute_block_coverage_fast(block, uncovered_mask):
        """Faster version using precomputed point lists."""
        bm = 0
        for x in block:
            bm |= (1 << x)
        count = 0
        for idx in range(n_tsubs):
            if uncovered_mask[idx] and (tsub_masks[idx] & ~bm) == 0:
                count += 1
        return count
    
    # Greedy with randomization
    best_blocks = None
    best_count = float('inf')
    
    n_restarts = 20
    rng = random.Random(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize uncovered mask
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []
        
        while uncovered.any():
            if time.time() > deadline - 3.0:
                break
                
            # Pick a random uncovered t-subset to start
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
            start_tsub = all_tsubs[start_idx]
            
            # Build block greedily starting from start_tsub
            block = list(start_tsub)
            
            while len(block) < k:
                if time.time() > deadline - 3.0:
                    break
                    
                # Find the point that covers the most uncovered t-subsets
                best_gain = -1
                best_point = -1
                block_set = set(block)
                
                # For efficiency, only consider points not in block
                candidates = [x for x in range(v) if x not in block_set]
                
                # To speed up, we can sample a subset of candidates if too slow
                # But with v=19, all 19 candidates is fine
                
                for x in candidates:
                    new_block = block + [x]
                    # Compute gain: how many currently uncovered t-subsets would be covered
                    bm = 0
                    for y in new_block:
                        bm |= (1 << y)
                    gain = 0
                    for idx in range(n_tsubs):
                        if uncovered[idx] and (tsub_masks[idx] & ~bm) == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point == -1:
                    break
                block.append(best_point)
            
            if len(block) < k:
                # Fill remaining with random points
                block_set = set(block)
                for x in range(v):
                    if x not in block_set and len(block) < k:
                        block.append(x)
                    if len(block) >= k:
                        break
            
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered
            bm = 0
            for x in block:
                bm |= (1 << x)
            for idx in range(n_tsubs):
                if (tsub_masks[idx] & ~bm) == 0:
                    uncovered[idx] = False
        
        # Check if we have a complete covering
        if not uncovered.any():
            # Post-hoc redundancy removal
            blocks_list = [list(b) for b in blocks]
            n_blocks = len(blocks_list)
            
            # Try to remove blocks that are redundant
            changed = True
            while changed:
                if time.time() > deadline - 2.0:
                    break
                changed = False
                for i in range(len(blocks_list)):
                    if time.time() > deadline - 2.0:
                        break
                    # Temporarily remove block i
                    remaining = blocks_list[:i] + blocks_list[i+1:]
                    # Check if all t-subsets are still covered
                    all_covered = np.zeros(n_tsubs, dtype=bool)
                    for b in remaining:
                        bm = 0
                        for x in b:
                            bm |= (1 << x)
                        for idx in range(n_tsubs):
                            if (tsub_masks[idx] & ~bm) == 0:
                                all_covered[idx] = True
                    if all_covered.all():
                        blocks_list = remaining
                        changed = True
                        break
            
            if len(blocks_list) < best_count:
                best_count = len(blocks_list)
                best_blocks = [b for b in blocks_list]
        
        # If we've hit the known best, we can stop early
        if best_count <= 151:
            break
    
    if best_blocks is None:
        # Fallback: return whatever we have (shouldn't happen if we completed)
        # Generate a valid covering using pure greedy
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []
        while uncovered.any():
            # Find first uncovered t-subset
            for idx in range(n_tsubs):
                if uncovered[idx]:
                    break
            block = list(all_tsubs[idx])
            while len(block) < k:
                best_gain = -1
                best_point = -1
                block_set = set(block)
                for x in range(v):
                    if x not in block_set:
                        bm = 0
                        for y in block + [x]:
                            bm |= (1 << y)
                        gain = 0
                        for j in range(n_tsubs):
                            if uncovered[j] and (tsub_masks[j] & ~bm) == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                block.append(best_point)
            block.sort()
            blocks.append(block)
            bm = 0
            for x in block:
                bm |= (1 << x)
            for j in range(n_tsubs):
                if (tsub_masks[j] & ~bm) == 0:
                    uncovered[j] = False
        best_blocks = blocks
    
    return best_blocks
