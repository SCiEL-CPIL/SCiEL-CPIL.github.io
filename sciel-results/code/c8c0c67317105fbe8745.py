"""Covering design C(24,7,3) - randomized greedy with local search removal."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    deadline = time.time() + 135.0  # safety margin before 150s kill
    rng = random.Random(42)
    
    best_blocks = None
    best_count = float('inf')
    
    def greedy_cover(seed):
        """Randomized greedy set cover with reverse indexing."""
        local_rng = random.Random(seed)
        # Map each t-subset to a unique index
        all_tsubs = list(itertools.combinations(range(v), t))
        n_tsubs = len(all_tsubs)
        tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
        
        # For each point, precompute which t-subsets it participates in
        # point -> list of t-subset indices containing that point
        point_to_tsubs = [[] for _ in range(v)]
        for i, ts in enumerate(all_tsubs):
            for p in ts:
                point_to_tsubs[p].append(i)
        
        # Track which t-subsets are still uncovered
        uncovered = bytearray([1] * n_tsubs)
        uncovered_count = n_tsubs
        
        blocks = []
        
        while uncovered_count > 0:
            # Find a random uncovered t-subset to start the block
            uncov_indices = [i for i in range(n_tsubs) if uncovered[i]]
            if not uncov_indices:
                break
            start_idx = local_rng.choice(uncov_indices)
            start_tsub = all_tsubs[start_idx]
            
            # Build block: start with one point from the uncovered t-subset
            block = [local_rng.choice(start_tsub)]
            
            while len(block) < k:
                # For each candidate point, compute gain (how many new uncovered t-subsets it covers)
                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in block:
                        continue
                    # Count uncovered t-subsets that contain x and (k-t-1) points from block
                    gain = 0
                    # A t-subset covered by adding x to block must contain x and (t-1) points from block
                    # But it also needs to be currently uncovered
                    # Efficient: for each (t-1)-subset of block, check if the t-subset (that + x) is uncovered
                    if t == 1:
                        if uncovered[tsub_to_idx[(x,)]]:
                            gain = 1
                    else:
                        for rest in itertools.combinations(block, t - 1):
                            ts = tuple(sorted(rest + (x,)))
                            idx = tsub_to_idx.get(ts)
                            if idx is not None and uncovered[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    # Early termination if we can't beat current best
                    # (not easily doable, but we can skip if gain is 0 and best_gain > 0)
                
                if best_point is None:
                    # Fallback: pick random point
                    remaining = [x for x in range(v) if x not in block]
                    if remaining:
                        best_point = local_rng.choice(remaining)
                    else:
                        break
                
                block.append(best_point)
            
            if len(block) < k:
                break
            
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                if uncovered[idx]:
                    uncovered[idx] = 0
                    uncovered_count -= 1
        
        return [list(b) for b in blocks]
    
    def local_search_remove(blocks, max_time=30.0):
        """Try to remove redundant blocks."""
        if not blocks:
            return blocks
        
        start_time = time.time()
        n_blocks = len(blocks)
        
        # For each block, count how many t-subsets it uniquely covers
        # A block is removable if all its t-subsets are covered by other blocks
        
        # Build coverage: for each t-subset, which blocks cover it
        # This is expensive, so use a different approach:
        # For each block, temporarily remove it and check if all its t-subsets are still covered
        
        # More efficient: for each t-subset, count how many blocks cover it
        all_tsubs = list(itertools.combinations(range(v), t))
        tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
        n_tsubs = len(all_tsubs)
        
        # coverage_count[i] = number of blocks that cover t-subset i
        coverage_count = np.zeros(n_tsubs, dtype=np.int32)
        
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage_count[idx] += 1
        
        # A block is removable if all t-subsets it covers have coverage_count > 1
        # (i.e., they are covered by at least one other block)
        
        removable = []
        for bi, block in enumerate(blocks):
            is_removable = True
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                if coverage_count[idx] <= 1:
                    is_removable = False
                    break
            if is_removable:
                removable.append(bi)
        
        # Remove all removable blocks (iteratively, since removing one might make others removable)
        changed = True
        while changed and time.time() - start_time < max_time:
            changed = False
            removable = []
            for bi, block in enumerate(blocks):
                is_removable = True
                for ts in itertools.combinations(block, t):
                    idx = tsub_to_idx[ts]
                    if coverage_count[idx] <= 1:
                        is_removable = False
                        break
                if is_removable:
                    removable.append(bi)
            
            if removable:
                # Remove one at a time to update coverage correctly
                # Remove the last one to avoid index issues
                bi = removable[-1]
                block = blocks[bi]
                for ts in itertools.combinations(block, t):
                    idx = tsub_to_idx[ts]
                    coverage_count[idx] -= 1
                blocks.pop(bi)
                changed = True
        
        return blocks
    
    # Run multiple greedy constructions with different seeds
    n_attempts = 10
    for attempt in range(n_attempts):
        if time.time() > deadline - 60:  # Leave time for local search
            break
        
        seed = 42 + attempt
        blocks = greedy_cover(seed)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        # Apply local search removal
        if time.time() > deadline - 45:
            break
        
        reduced = local_search_remove([list(b) for b in best_blocks] if best_blocks else [], max_time=20.0)
        if reduced and len(reduced) < best_count:
            best_count = len(reduced)
            best_blocks = reduced
    
    # Final local search pass on best solution
    if best_blocks and time.time() < deadline - 15:
        final = local_search_remove(best_blocks, max_time=15.0)
        if final and len(final) < best_count:
            best_blocks = final
            best_count = len(final)
    
    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = greedy_cover(0)
    
    return best_blocks
# EVOLVE-BLOCK-END
