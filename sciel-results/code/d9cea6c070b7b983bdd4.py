"""Covering design C(v,k,t) via greedy set cover with on-the-fly candidate generation."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    
    # Precompute all t-subsets and map them to indices for fast lookup?
    # Actually, we just need to track which t-subsets are covered.
    # Total t-subsets: C(19,5) = 11628. This is small enough to track in a boolean array.
    all_t_subs = list(itertools.combinations(all_points, t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    covered = np.zeros(n_t, dtype=np.uint8)
    uncovered_count = n_t
    blocks = []
    
    # To speed up gain calculation, we can precompute the t-subsets for each potential block?
    # No, blocks are generated on the fly.
    # But we can precompute, for each t-subset, the list of points NOT in it?
    # Or better: for each t-subset S, the candidates are S U X where X is a (k-t)-subset of (All \ S).
    
    # Precompute complements?
    # For a given t-subset S, the available points for the rest of the block are All \ S.
    # We need to choose (k-t) from these.
    
    # Optimization:
    # 1. Pick an uncovered t-subset S.
    # 2. Generate all k-blocks containing S.
    # 3. For each such block B, calculate gain = number of uncovered t-subsets in B.
    # 4. Pick B with max gain.
    
    # How to calculate gain fast?
    # A block B contains C(k,t) t-subsets.
    # We can iterate through all t-subsets of B and check if they are uncovered.
    # C(8,5) = 56 checks per candidate block.
    # Number of candidates per step: C(v-t, k-t) = C(14,3) = 364.
    # Total checks per step: 364 * 56 ~ 20,000.
    # Number of steps: roughly n_blocks ~ 300.
    # Total operations: 300 * 20,000 = 6,000,000. This is very fast in Python/NumPy.
    
    # We need a way to quickly get the index of a t-subset to check `covered`.
    # t_sub_to_idx is a dict. Lookup is O(1).
    
    while uncovered_count > 0 and time_ok():
        # Find an uncovered t-subset to anchor the search.
        # Heuristic: Pick one with fewest possible extensions? 
        # All t-subsets have the same number of extensions C(v-t, k-t).
        # So any uncovered one is equally "constrained" in terms of extension count.
        # However, some might be covered by fewer *already selected* blocks? No, we only add blocks.
        # Let's just pick a random uncovered one to diversify, or the first one.
        # Random might help avoid local minima in greedy choice.
        
        # Find indices of uncovered t-subsets
        # This scan is O(n_t) = 11628. 300 steps * 11628 ~ 3.5M ops. Acceptable.
        # We can optimize by maintaining a list of uncovered indices.
        
        # Let's maintain a list of uncovered t-subset indices for faster selection.
        # But updating it is O(n_t) if we filter.
        # Given n_t is small, scanning is fine.
        
        uncovered_indices = np.where(covered == 0)[0]
        if len(uncovered_indices) == 0:
            break
            
        # Pick a random uncovered t-subset
        anchor_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
        anchor_t_sub = all_t_subs[anchor_idx]
        anchor_set = set(anchor_t_sub)
        
        # Generate all k-blocks containing anchor_t_sub
        remaining_points = [p for p in all_points if p not in anchor_set]
        # Need to choose (k-t) from remaining_points
        k_minus_t = k - t
        candidates = []
        
        # Pre-compute t-subsets for a block? No, do it on the fly.
        best_block = None
        best_gain = -1
        
        # Iterate through all combinations of (k-t) points from remaining_points
        for ext in itertools.combinations(remaining_points, k_minus_t):
            block = list(anchor_t_sub) + list(ext)
            # Calculate gain: how many uncovered t-subsets are in this block?
            gain = 0
            # Iterate through all t-subsets of this block
            for t_sub in itertools.combinations(block, t):
                # Check if this t_sub is uncovered
                # We need its index
                # Using dict lookup
                idx = t_sub_to_idx[t_sub]
                if covered[idx] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block is None:
            break
            
        # Add best_block to our solution
        blocks.append(best_block)
        
        # Update coverage
        for t_sub in itertools.combinations(best_block, t):
            idx = t_sub_to_idx[t_sub]
            if covered[idx] == 0:
                covered[idx] = 1
                uncovered_count -= 1
                
    # Reduction Phase: Remove redundant blocks
    # Sort blocks? No, just iterate.
    # For each block, check if it can be removed.
    # A block can be removed if every t-subset in it is covered by at least one OTHER block.
    
    # To do this efficiently, we need cover counts.
    # Re-calculate cover counts for the current list of blocks.
    if blocks:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in blocks:
            for t_sub in itertools.combinations(b, t):
                idx = t_sub_to_idx[t_sub]
                cover_count[idx] += 1
        
        # Multi-pass removal
        changed = True
        while changed and time_ok():
            changed = False
            new_blocks = []
            for b in blocks:
                # Check if any t-subset in b has cover_count == 1
                keep = False
                for t_sub in itertools.combinations(b, t):
                    idx = t_sub_to_idx[t_sub]
                    if cover_count[idx] == 1:
                        keep = True
                        break
                if keep:
                    new_blocks.append(b)
                else:
                    # Remove block b
                    changed = True
                    for t_sub in itertools.combinations(b, t):
                        idx = t_sub_to_idx[t_sub]
                        cover_count[idx] -= 1
            blocks = new_blocks
    
    return blocks
# EVOLVE-BLOCK-END
