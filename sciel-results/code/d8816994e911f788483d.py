import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s (or 150s) kill

    # Precompute all t-subsets and map to indices for fast lookup
    all_tsubsets = list(itertools.combinations(range(v), t))
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubsets)}
    n_tsubs = len(all_tsubsets)
    
    # Represent coverage as a numpy array for speed
    # covered[i] = number of blocks covering t-subset i
    # We actually just need to know if it's uncovered (0) or not.
    # But for local improvement, counts are useful.
    
    best_blocks = None
    best_len = float('inf')
    
    # Strategy:
    # 1. Run multiple greedy constructions with different random seeds/strategies.
    # 2. After each construction, run local improvement (remove redundant, swap blocks).
    # 3. Keep the best valid solution found.
    
    # Optimization: Precompute for each point x, the list of t-subsets containing x?
    # No, that's huge.
    # Instead, during greedy, for a candidate block B, we need to know which t-subsets it covers.
    # A block B of size k covers C(k, t) t-subsets.
    # We can generate these on the fly.
    
    # To speed up greedy selection:
    # When adding a point x to current block prefix P (size < k-1),
    # the gain is the number of uncovered t-subsets that can be formed by adding x to P.
    # Specifically, if P has size j, and we add x, the new block will have size j+1.
    # The t-subsets covered by the final block that involve x and elements from P
    # are not quite right because the block isn't finished.
    #
    # Standard greedy for set cover:
    # At each step, pick the block that covers the most uncovered t-subsets.
    # But we are constructing blocks one element at a time?
    # The current code constructs a block by picking a seed t-subset, then adding points one by one.
    # This is a heuristic for finding a good block.
    
    # Let's refine the greedy block construction.
    # 1. Pick an uncovered t-subset S.
    # 2. We must include all elements of S in the new block.
    # 3. We need to add k - t more elements to S to form a k-subset B.
    # 4. We want B to cover as many other uncovered t-subsets as possible.
    
    # Let's try a different greedy approach:
    # Maintain a set of uncovered t-subsets.
    # While uncovered is not empty:
    #   - Find the "best" k-subset to add.
    #   - "Best" is determined by how many uncovered t-subsets it covers.
    #   - To find the best k-subset efficiently:
    #     - Pick an uncovered t-subset S.
    #     - Any valid block must contain S.
    #     - So we only need to search k-subsets that contain S.
    #     - There are C(v-t, k-t) such subsets.
    #     - For v=24, k=7, t=3: C(21, 4) = 5985. This is small enough to enumerate!
    
    # This is a much better strategy than the incremental point addition.
    # It guarantees the new block covers S, and we can exactly calculate the gain for all candidates containing S.
    
    def _greedy_cover_fast(v, k, t, all_tsubsets, tsub_to_idx, deadline, seed_val):
        rng = random.Random(seed_val)
        
        # Use a boolean array or set for uncovered?
        # Set of indices is probably fine if we use numpy for counting?
        # Let's use a numpy array `covered` where covered[i] > 0 means covered.
        # Or just a set of uncovered indices.
        
        uncovered_indices = set(range(len(all_tsubsets)))
        blocks = []
        
        while uncovered_indices:
            if time.time() > deadline:
                return None, False
            
            # Pick a random uncovered t-subset to anchor our search
            anchor_idx = rng.choice(list(uncovered_indices))
            anchor_tsub = all_tsubsets[anchor_idx]
            anchor_set = set(anchor_tsub)
            
            # Generate all k-subsets containing anchor_tsub
            # We need to choose (k-t) additional elements from the remaining (v-t) points
            remaining_points = [x for x in range(v) if x not in anchor_set]
            
            best_gain = -1
            best_block = None
            best_tiebreakers = []
            
            # Iterate over all combinations of (k-t) points from remaining_points
            # C(21, 4) = 5985 combinations.
            for combo in itertools.combinations(remaining_points, k - t):
                candidate_block = tuple(sorted(list(anchor_tsub) + list(combo)))
                
                # Calculate gain: how many uncovered t-subsets does this block cover?
                gain = 0
                # Generate all t-subsets of candidate_block
                for tsub in itertools.combinations(candidate_block, t):
                    idx = tsub_to_idx[tsub]
                    if idx in uncovered_indices:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
                    best_tiebreakers = [candidate_block]
                elif gain == best_gain and gain > 0:
                    best_tiebreakers.append(candidate_block)
            
            if best_block is None or best_gain == 0:
                # Should not happen if uncovered is not empty and we pick from uncovered
                return None, False
                
            # If there are tiebreakers, pick one randomly
            chosen_block = rng.choice(best_tiebreakers)
            
            blocks.append(chosen_block)
            
            # Update uncovered set
            for tsub in itertools.combinations(chosen_block, t):
                idx = tsub_to_idx[tsub]
                uncovered_indices.discard(idx)
        
        return blocks, True

    def _local_improve(blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline):
        if not blocks:
            return blocks
        
        # Build coverage count
        # Using a dict or array? Array is faster.
        cover_count = np.zeros(len(all_tsubsets), dtype=np.int32)
        
        for block in blocks:
            for tsub in itertools.combinations(block, t):
                idx = tsub_to_idx[tsub]
                cover_count[idx] += 1
        
        improved = list(blocks)
        n = len(improved)
        
        # 1. Remove redundant blocks
        # A block is redundant if all its t-subsets have cover_count > 1
        # We iterate multiple times because removing one block might make another redundant.
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for i in range(len(improved) - 1, -1, -1):
                block = improved[i]
                redundant = True
                for tsub in itertools.combinations(block, t):
                    idx = tsub_to_idx[tsub]
                    if cover_count[idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block i
                    for tsub in itertools.combinations(block, t):
                        idx = tsub_to_idx[tsub]
                        cover_count[idx] -= 1
                    improved.pop(i)
                    changed = True
        
        # 2. Try to replace two blocks with one? (2-for-1)
        # This is complex. Maybe just stick with removal and maybe simple swaps.
        # Let's try a simple local search: try to replace a block with a better one?
        # Or just run the greedy again on the uncovered parts? No, that's complicated.
        
        # Let's try to see if we can swap a block with another that covers more?
        # Since we already removed redundant ones, any remaining block is essential.
        # Swapping might be hard to verify quickly.
        
        return improved

    # Main loop
    trial = 0
    while time.time() < deadline:
        trial += 1
        
        # Use different seeds for diversity
        seed = trial
        
        blocks, valid = _greedy_cover_fast(v, k, t, all_tsubsets, tsub_to_idx, deadline, seed)
        
        if valid:
            # Local improvement
            improved_blocks = _local_improve(blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline)
            
            if len(improved_blocks) < best_len:
                best_len = len(improved_blocks)
                best_blocks = improved_blocks
                
            # Early stop if we hit the known best (76)
            if best_len <= 76:
                break

    if best_blocks is None:
        # Fallback
        blocks, valid = _greedy_cover_fast(v, k, t, all_tsubsets, tsub_to_idx, deadline, 0)
        if valid:
            best_blocks = blocks
            best_blocks = _local_improve(best_blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline)
        else:
            best_blocks = []
            
    return best_blocks
