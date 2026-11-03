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
            
    # Convert tsub_blocks to lists of ints for faster iteration
    # (they are already lists of ints)

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = [True] * n_tsubsets
        # gain[i] = number of uncovered t-subsets in block i
        gain = [0] * n_blocks
        for bi in range(n_blocks):
            gain[bi] = len(block_tsubs[bi])
            
        selected = []
        selected_set = set()
        
        while True:
            if time.time() > deadline - 5:
                break
                
            # Find block with max gain
            max_gain = -1
            best_bi = -1
            # Random tie-breaking
            candidates = []
            for bi in range(n_blocks):
                if bi in selected_set:
                    continue
                g = gain[bi]
                if g > max_gain:
                    max_gain = g
                    candidates = [bi]
                elif g == max_gain and g > 0:
                    candidates.append(bi)
                    
            if max_gain <= 0:
                break
                
            # Pick random from candidates
            bi = candidates[rng.randint(0, len(candidates) - 1)]
            selected.append(bi)
            selected_set.add(bi)
            
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
        stall_count = 0
        max_stalls = 3
        
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
                stall_count = 0
                continue # Restart loop as structure changed significantly
            
            # Phase 2: Try swaps
            # For each block, try to replace it with another that covers its unique subsets
            # and potentially allows further reduction
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            # Precompute unique t-subsets for all blocks to avoid recomputation
            unique_tsubs_cache = {}
            for bi in current_blocks:
                if bi not in block_set:
                    continue
                unique_tsubs_cache[bi] = [ts_idx for ts_idx in block_tsubs[bi] if cover_count[ts_idx] == 1]
            
            for bi in current_blocks:
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue
                    
                unique_tsubs = unique_tsubs_cache.get(bi, [])
                
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
                    
                # Try candidates, ordered by how many additional t-subsets they cover
                cand_list = list(candidates)
                # Score candidates by how many new t-subsets they cover (not in current set)
                def score_candidate(bj):
                    score = 0
                    for ts_idx in block_tsubs[bj]:
                        if cover_count[ts_idx] == 0:
                            score += 1
                    return score
                cand_list.sort(key=score_candidate, reverse=True)
                rng.shuffle(cand_list)  # Add randomness to break ties
                
                swapped = False
                for bj in cand_list[:20]: # Limit trials per block
                    if bj in block_set:
                        continue
                        
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
                    stall_count = 0
                    break # Restart while loop
                    
            if not improved:
                stall_count += 1
                if stall_count >= max_stalls:
                    break
        return list(block_set)

    def two_block_swap_search(blocks, seed):
        """
        Try to replace two blocks (bi, bj) with a single block bk that covers
        all unique t-subsets of bi and bj. This is a 2-for-1 improvement.
        """
        rng = random.Random(seed)
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
                
        # Precompute t-subset sets for each block for faster intersection
        block_tsub_sets = {bi: set(block_tsubs[bi]) for bi in block_set}
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            # Try pairs of blocks
            # To keep it fast, we limit the number of pairs we check per pass
            n_checked = 0
            for i, bi in enumerate(current_blocks):
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue
                    
                # Pick a random partner bj from the remaining blocks
                if len(block_set) < 2:
                    break
                others = [b for b in block_set if b != bi]
                if not others:
                    continue
                bj = rng.choice(others)
                
                n_checked += 1
                if n_checked > 3000: # Limit checks per pass to avoid O(N^2) explosion
                    break
                
                # Find unique t-subsets for bi and bj
                req_bi = [ts for ts in block_tsubs[bi] if cover_count[ts] == 1]
                req_bj = [ts for ts in block_tsubs[bj] if cover_count[ts] == 1]
                
                # Also, any ts in intersection of bi and bj with count == 2 must be covered by bk
                set_bi = block_tsub_sets.get(bi, set(block_tsubs[bi]))
                set_bj = block_tsub_sets.get(bj, set(block_tsubs[bj]))
                
                required = set(req_bi)
                required.update(req_bj)
                
                # Check intersection
                inter = set_bi & set_bj
                for ts in inter:
                    if cover_count[ts] == 2:
                        required.add(ts)
                
                if not required:
                    # Both blocks are redundant? Phase 1 should have handled this.
                    continue
                
                # Find candidate blocks bk that cover all ts in required
                # Pick ts with fewest covering blocks
                min_ts = min(required, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_blocks[min_ts])
                
                for ts in required:
                    if ts == min_ts:
                        continue
                    if not candidates:
                        break
                    candidates &= set(tsub_blocks[ts])
                    
                # Remove bi and bj from candidates
                candidates.discard(bi)
                candidates.discard(bj)
                
                if not candidates:
                    continue
                
                cand_list = list(candidates)
                
                # Score candidates by how many additional t-subsets they cover
                def score_candidate(bk):
                    score = 0
                    for ts_idx in block_tsubs[bk]:
                        if cover_count[ts_idx] == 0:
                            score += 1
                    return score
                cand_list.sort(key=score_candidate, reverse=True)
                rng.shuffle(cand_list)  # Add randomness to break ties
                
                swapped = False
                for bk in cand_list[:10]: # Limit trials
                    if bk in block_set:
                        continue
                        
                    # Perform swap: remove bi, bj; add bk
                    block_set.remove(bi)
                    for ts in block_tsubs[bi]:
                        cover_count[ts] -= 1
                        
                    block_set.remove(bj)
                    for ts in block_tsubs[bj]:
                        cover_count[ts] -= 1
                        
                    block_set.add(bk)
                    for ts in block_tsubs[bk]:
                        cover_count[ts] += 1
                        
                    # Update precomputed set for new block
                    block_tsub_sets[bk] = set(block_tsubs[bk])
                    # Clean up removed blocks
                    block_tsub_sets.pop(bi, None)
                    block_tsub_sets.pop(bj, None)
                    
                    improved = True
                    swapped = True
                    break
                    
                if swapped:
                    break # Restart while loop
                    
        return list(block_set)

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy starts
    num_trials = 50
    for i in range(num_trials):
        if time.time() > deadline - 25:
            break
            
        blocks = greedy_cover(i)
        if not blocks:
            continue
            
        # Local search (single swaps)
        blocks = local_search(blocks, i)
        
        # Two-block swap search
        if time.time() < deadline - 12:
            blocks = two_block_swap_search(blocks, i + 1000)
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
            
    # If we have time, try to improve the best solution further
    if best_blocks is not None and time.time() < deadline - 8:
        best_blocks = local_search(best_blocks, 99999)
        if time.time() < deadline - 4:
            best_blocks = two_block_swap_search(best_blocks, 88888)

    # Fallback if something went wrong
    if best_blocks is None:
        best_blocks = greedy_cover(0)
        
    return [all_blocks[i] for i in best_blocks]
