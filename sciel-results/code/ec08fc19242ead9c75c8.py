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
    
    best_blocks = None
    best_len = float('inf')
    
    def _greedy_cover_fast(v, k, t, all_tsubsets, tsub_to_idx, deadline, seed_val):
        rng = random.Random(seed_val)
        
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
            remaining_points = [x for x in range(v) if x not in anchor_set]
            
            best_ratio = -1.0
            best_block = None
            best_tiebreakers = []
            
            # Iterate over all combinations of (k-t) points from remaining_points
            for combo in itertools.combinations(remaining_points, k - t):
                candidate_block = tuple(sorted(list(anchor_tsub) + list(combo)))
                
                # Calculate gain: how many uncovered t-subsets does this block cover?
                gain = 0
                for tsub in itertools.combinations(candidate_block, t):
                    idx = tsub_to_idx[tsub]
                    if idx in uncovered_indices:
                        gain += 1
                
                if gain == 0:
                    continue
                    
                # Use gain per point as the selection criterion
                # This penalizes blocks that cover many already-covered triplets
                ratio = gain / len(candidate_block)
                
                if ratio > best_ratio + 1e-12:
                    best_ratio = ratio
                    best_block = candidate_block
                    best_tiebreakers = [candidate_block]
                elif abs(ratio - best_ratio) < 1e-12:
                    best_tiebreakers.append(candidate_block)
            
            if best_block is None or best_ratio <= 0:
                return None, False
                
            # If there are tiebreakers, pick one randomly with slight preference
            # for blocks with higher absolute gain (secondary criterion)
            if len(best_tiebreakers) > 1:
                # Weight by gain for better tie-breaking
                gains = []
                for b in best_tiebreakers:
                    g = 0
                    for tsub in itertools.combinations(b, t):
                        idx = tsub_to_idx[tsub]
                        if idx in uncovered_indices:
                            g += 1
                    gains.append(g)
                # Use softmax-like weighting
                max_g = max(gains)
                weights = [g / (max_g + 1e-9) for g in gains]
                chosen_block = rng.choices(best_tiebreakers, weights=weights)[0]
            else:
                chosen_block = best_tiebreakers[0]
            
            blocks.append(chosen_block)
            
            # Update uncovered set
            for tsub in itertools.combinations(chosen_block, t):
                idx = tsub_to_idx[tsub]
                uncovered_indices.discard(idx)
        
        return blocks, True

    def _fast_prune(blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline):
        """
        Fast iterative redundancy removal (Reverse Greedy).
        A block is redundant if all its t-subsets are covered by other blocks.
        This is equivalent to finding a minimal subcover.
        """
        if not blocks:
            return blocks
            
        # Precompute t-subsets for each block to avoid recomputation
        block_tsubs = [list(itertools.combinations(block, t)) for block in blocks]
        
        # Count coverage
        cover_count = np.zeros(len(all_tsubsets), dtype=np.int32)
        for tsubs in block_tsubs:
            for tsub in tsubs:
                idx = tsub_to_idx[tsub]
                cover_count[idx] += 1
        
        improved_blocks = list(blocks)
        improved_tsubs = list(block_tsubs)
        
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            # Iterate backwards to allow safe removal
            for i in range(len(improved_blocks) - 1, -1, -1):
                redundant = True
                for tsub in improved_tsubs[i]:
                    idx = tsub_to_idx[tsub]
                    if cover_count[idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block i
                    for tsub in improved_tsubs[i]:
                        idx = tsub_to_idx[tsub]
                        cover_count[idx] -= 1
                    improved_blocks.pop(i)
                    improved_tsubs.pop(i)
                    changed = True
        
        return improved_blocks

    def _local_improve(blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline):
        if not blocks:
            return blocks
        
        # Use the fast prune function
        improved = _fast_prune(blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline)
        
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
