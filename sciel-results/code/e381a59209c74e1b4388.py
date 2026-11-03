# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from math import comb

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using precomputed block masks and local search.
    """
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {frozenset(c): i for i, c in enumerate(all_tsubs)}
    
    # Precompute all k-subsets and their t-subset masks
    # C(19,7) = 50388 blocks. 
    # Each mask is an integer bitmask of n_tsubs bits.
    # n_tsubs = C(19,4) = 3876. 
    # 3876 bits requires ~61 uint64s. 
    # 50388 * 61 * 8 bytes ~ 24 MB. This is acceptable.
    
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    
    # Map k-subset index to its mask (list of uint64 or just a set of tsub indices for faster intersection?)
    # Using Python sets of tsub indices might be faster for small intersections than bit manipulation in numpy for this size.
    # Let's use sets of tsub indices.
    
    block_tsubs = []
    for b in all_ksubs:
        ts = set()
        for c in itertools.combinations(b, t):
            ts.add(tsub_to_idx[frozenset(c)])
        block_tsubs.append(ts)
        
    # Greedy Phase
    # uncovered: set of tsub indices not yet covered
    uncovered = set(range(n_tsubs))
    chosen_blocks = []
    
    # To speed up greedy, we can just iterate. 
    # Finding max gain:
    # For each candidate block, gain = len(block_tsubs[i] & uncovered)
    # This is O(n_ksubs * avg_size) per step. 
    # n_ksubs ~ 50k. avg_size ~ 35. 
    # 50k * 35 = 1.75M operations per step.
    # Number of steps ~ 150.
    # 150 * 1.75M = 262M operations. Might be slow in Python.
    
    # Optimization: 
    # Instead of scanning all blocks, we can pick a random uncovered t-subset, 
    # and only consider blocks that contain it.
    # How many blocks contain a specific t-subset?
    # A t-subset has t points. We need to choose k-t more points from v-t.
    # C(19-4, 7-4) = C(15,3) = 455 blocks.
    # This is MUCH faster!
    
    # Precompute: for each t-subset, which k-subsets contain it?
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for i, b in enumerate(all_ksubs):
        b_set = set(b)
        # Iterate over t-subsets of b
        for c in itertools.combinations(b, t):
            idx = tsub_to_idx[frozenset(c)]
            tsub_to_blocks[idx].append(i)
            
    rng = random.Random(42)
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick a random uncovered t-subset to focus on
        # Using random choice from a set is O(1) amortized if we convert to list, but set is good.
        # To be efficient, we can just pick one.
        # However, picking the "best" uncovered t-subset (one contained in fewest candidate blocks?) 
        # or just random is fine. Random is faster.
        
        # Pick random uncovered t-subset
        # Converting set to list every time is expensive if uncovered is large.
        # But uncovered shrinks.
        # Let's just pick an arbitrary element from the set.
        target_tsub = next(iter(uncovered))
        
        # Get candidate blocks that cover this t-subset
        candidates = tsub_to_blocks[target_tsub]
        
        # Find the candidate that covers the most *other* uncovered t-subsets
        best_block_idx = -1
        max_gain = 0
        
        for b_idx in candidates:
            # Gain is number of uncovered t-subsets in this block
            # block_tsubs[b_idx] is a set
            # Intersection size
            gain = len(block_tsubs[b_idx] & uncovered)
            if gain > max_gain:
                max_gain = gain
                best_block_idx = b_idx
                # Early exit if we found a perfect gain? Max possible is len(block_tsubs[b_idx])
                # if gain == len(block_tsubs[b_idx]): break # Unlikely to be perfect early
        
        if best_block_idx == -1:
            break # Should not happen if uncovered is not empty
            
        # Add block
        chosen_blocks.append(all_ksubs[best_block_idx])
        # Update uncovered
        uncovered -= block_tsubs[best_block_idx]
        
    # Local Search Phase
    # We have a valid cover. Now try to reduce size.
    # 1. Remove redundant blocks
    # 2. Replace blocks to allow removal of others?
    
    # Represent cover as list of block indices
    cover_indices = [all_ksubs.index(b) for b in chosen_blocks]
    
    # Coverage count for each t-subset
    cov_count = [0] * n_tsubs
    for b_idx in cover_indices:
        for ts_idx in block_tsubs[b_idx]:
            cov_count[ts_idx] += 1
            
    # Helper to check if removing a block is safe
    def can_remove(b_idx):
        for ts_idx in block_tsubs[b_idx]:
            if cov_count[ts_idx] == 1:
                return False
        return True
        
    # Helper to remove block
    def remove_block(b_idx):
        for ts_idx in block_tsubs[b_idx]:
            cov_count[ts_idx] -= 1
        cover_indices.remove(b_idx)
        
    # Helper to add block
    def add_block(b_idx):
        for ts_idx in block_tsubs[b_idx]:
            cov_count[ts_idx] += 1
        cover_indices.append(b_idx)
        
    # Iterative improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Remove redundant blocks
        i = 0
        while i < len(cover_indices):
            b_idx = cover_indices[i]
            if can_remove(b_idx):
                remove_block(b_idx)
                improved = True
                # Don't increment i, check next element at same index
            else:
                i += 1
                
        if time.time() > deadline:
            break
            
        # Pass 2: Try to replace a block with a "better" one?
        # Or try to swap?
        # Simple swap: For each block A in cover, try to replace it with B not in cover
        # such that B covers all t-subsets unique to A, and doesn't break others.
        # This is expensive. 
        # Alternative: Just try to find a block B that covers a set of t-subsets 
        # such that we can remove 2 blocks?
        
        # Let's stick to 1-exchange (replace A with B)
        # For each A in cover:
        #   Find t-subsets unique to A.
        #   If none, remove A (handled in Pass 1).
        #   If some, find B (not in cover) such that:
        #     1. B covers all unique t-subsets of A.
        #     2. For any t-subset in A but not in B, cov_count[ts] >= 2 (so removing A is safe for them).
        
        # Optimization: 
        # Precompute for each block A, the set of "unique" t-subsets? 
        # This changes as we modify the cover.
        
        # We will iterate through cover blocks.
        for i in range(len(cover_indices)):
            if time.time() > deadline:
                break
                
            A_idx = cover_indices[i]
            A_tsubs = block_tsubs[A_idx]
            
            # Find unique t-subsets for A
            unique_to_A = set()
            for ts in A_tsubs:
                if cov_count[ts] == 1:
                    unique_to_A.add(ts)
                    
            if not unique_to_A:
                # Should have been removed in Pass 1, but maybe added later?
                remove_block(A_idx)
                improved = True
                # Adjust indices? cover_indices changed.
                # Break and restart loop to be safe
                break
            else:
                # Try to find a replacement B
                # B must contain all points in the union of unique_to_A?
                # No, B must cover the t-subsets.
                # The union of points in unique_to_A must be a subset of B.
                union_pts = set()
                for ts in unique_to_A:
                    union_pts.update(all_tsubs[ts])
                    
                if len(union_pts) > k:
                    continue # No single block can cover these unique t-subsets
                
                # Candidates for B: blocks that contain union_pts
                # How to find them quickly?
                # We can iterate all blocks? 50k.
                # Or use the tsub_to_blocks structure?
                # B must contain every t-subset in unique_to_A.
                # So B must be in the intersection of tsub_to_blocks[ts] for all ts in unique_to_A.
                
                # Start with the set of blocks containing the first unique t-subset
                first_ts = next(iter(unique_to_A))
                candidate_blocks = set(tsub_to_blocks[first_ts])
                
                # Intersect with blocks containing other unique t-subsets
                for ts in unique_to_A:
                    if ts == first_ts: continue
                    candidate_blocks &= set(tsub_to_blocks[ts])
                    if not candidate_blocks:
                        break
                        
                # Remove A itself and other blocks in cover
                cover_set = set(cover_indices)
                candidate_blocks -= cover_set
                
                # Evaluate candidates
                best_B = -1
                # We want a B that minimizes "waste" or just any valid one?
                # If we replace A with B, we need to ensure that any t-subset in A \ B 
                # is still covered by other blocks.
                
                for B_idx in candidate_blocks:
                    if time.time() > deadline:
                        break
                        
                    B_tsubs = block_tsubs[B_idx]
                    
                    # Check condition: for all ts in A_tsubs - B_tsubs, cov_count[ts] >= 2
                    # Note: cov_count includes A. So we need cov_count[ts] - 1 >= 1 => cov_count[ts] >= 2.
                    # Also, we must ensure B covers all unique_to_A (guaranteed by construction).
                    
                    valid = True
                    for ts in A_tsubs:
                        if ts not in B_tsubs:
                            if cov_count[ts] < 2:
                                valid = False
                                break
                                
                    if valid:
                        # Perform swap
                        # Remove A
                        for ts in A_tsubs:
                            cov_count[ts] -= 1
                        # Add B
                        for ts in B_tsubs:
                            cov_count[ts] += 1
                        # Update cover_indices
                        cover_indices[i] = B_idx
                        improved = True
                        best_B = B_idx
                        break # Found one, restart search
                        
                if improved:
                    break # Restart outer loop
                    
    # Convert back to lists of points
    return [list(b) for b in (all_ksubs[i] for i in cover_indices)]

# EVOLVE-BLOCK-END
