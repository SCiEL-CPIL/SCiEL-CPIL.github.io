import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115.0
    
    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # block_tsubs[i] = list of t-subset indices covered by block i
    block_tsubs = [None] * n_blocks
    # tsub_blocks[j] = list of block indices covering t-subset j
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = tsubs
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
            
    # Convert to numpy arrays for faster access if needed, 
    # but lists are fine for the logic below.
    
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = [True] * n_tsubsets
        # gain[i] = number of uncovered t-subsets in block i
        gain = [0] * n_blocks
        for bi in range(n_blocks):
            gain[bi] = len(block_tsubs[bi])
            
        selected = []
        # Use a heap or just max? 
        # With 50k blocks, finding max is O(N). 
        # We can use a heap of (-gain, bi).
        # But gains change. 
        # Simple O(N) scan is 50k * 150 steps = 7.5M ops. Very fast.
        
        while True:
            if time.time() > deadline - 5:
                break
                
            # Find block with max gain
            max_gain = -1
            best_bi = -1
            # Random tie-breaking
            candidates = []
            for bi in range(n_blocks):
                if gain[bi] > max_gain:
                    max_gain = gain[bi]
                    candidates = [bi]
                elif gain[bi] == max_gain and gain[bi] > 0:
                    candidates.append(bi)
                    
            if max_gain <= 0:
                break
                
            # Pick random from candidates
            bi = candidates[rng.randint(0, len(candidates) - 1)]
            selected.append(bi)
            
            # Update gains
            for ts_idx in block_tsubs[bi]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    # Decrement gain for all blocks covering this ts_idx
                    for bj in tsub_blocks[ts_idx]:
                        gain[bj] -= 1
                        
            gain[bi] = -1 # Prevent re-selection
            
        return selected

    def local_search(blocks, seed):
        rng = random.Random(seed)
        # blocks is a list of indices
        # We work with a set for O(1) lookup
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        
        # Initialize cover_count
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
                
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Phase 1: Remove redundant blocks
            # A block is redundant if all its t-subsets are covered by other blocks
            # i.e., cover_count[ts] > 1 for all ts in block
            to_remove = []
            for bi in block_set:
                redundant = True
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(bi)
                    
            if to_remove:
                for bi in to_remove:
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue # Restart loop as structure changed significantly
            
            # Phase 2: Try swaps
            # For each block, try to replace it with another that covers its unique subsets
            # and potentially allows further reduction
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            for bi in current_blocks:
                if time.time() > deadline - 2:
                    break
                if bi not in block_set: # Might have been removed in previous iterations? No, we continue above.
                    continue
                    
                # Find unique subsets for bi
                unique_tsubs = []
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)
                        
                if not unique_tsubs:
                    # Should have been caught in Phase 1, but just in case
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break # Restart
                    
                # Find candidate blocks that cover all unique_tsubs
                # Candidate must not be in block_set
                # Intersection of tsub_blocks[ts] for ts in unique_tsubs
                # Start with the one with fewest blocks
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_blocks[min_ts])
                
                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= set(tsub_blocks[ts_idx])
                    
                # Remove current block from candidates
                candidates.discard(bi)
                
                if not candidates:
                    continue
                    
                # Try candidates
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                swapped = False
                for bj in cand_list[:20]: # Limit trials per block
                    if bj in block_set:
                        continue
                        
                    # Check if swapping bi -> bj is valid
                    # It is valid if bj covers all unique_tsubs of bi
                    # We already ensured this by intersection.
                    # But we must ensure that removing bi doesn't uncover any other subset
                    # that is NOT covered by bj.
                    # Actually, if bj covers all unique_tsubs of bi, then:
                    # - Subsets with count > 1 remain covered (by others).
                    # - Subsets with count == 1 (unique to bi) are now covered by bj.
                    # So the swap is always valid in terms of coverage!
                    # Wait, does bj cover *only* those? No, bj covers 35 subsets.
                    # Some might be new, some might be already covered.
                    # The condition "bj covers all unique_tsubs of bi" is sufficient for validity.
                    
                    # Perform swap
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                        
                    block_set.add(bj)
                    for ts_idx in block_tsubs[bj]:
                        cover_count[ts_idx] += 1
                        
                    improved = True
                    swapped = True
                    break # Restart local search loop
                    
                if swapped:
                    break # Restart while loop
                    
        return list(block_set)

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy starts
    num_trials = 100
    for i in range(num_trials):
        if time.time() > deadline - 20:
            break
            
        blocks = greedy_cover(i)
        if not blocks:
            continue
            
        # Local search
        blocks = local_search(blocks, i)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    # If we have time, try to improve the best solution further
    if best_blocks is not None and time.time() < deadline - 10:
        # Run local search again on the best
        best_blocks = local_search(best_blocks, 99999)
        
    # Fallback if something went wrong
    if best_blocks is None:
        best_blocks = greedy_cover(0)
        
    return [all_blocks[i] for i in best_blocks]
