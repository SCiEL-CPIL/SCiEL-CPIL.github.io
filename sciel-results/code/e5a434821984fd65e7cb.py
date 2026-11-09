"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict


def construct(v, k, t):
    deadline = time.time() + 145.0  # safety margin before 150s kill
    rng = random.Random(42)

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each point, precompute which t-subsets it participates in
    # point -> list of t-subset indices containing that point
    point_tsubs = defaultdict(list)
    for i, ts in enumerate(all_tsubs):
        for p in ts:
            point_tsubs[p].append(i)

    def greedy_cover(seed_offset=0):
        """Randomized greedy set cover with reverse indexing."""
        # Track uncovered t-subsets using a set of indices
        uncovered = set(range(n_tsubs))
        blocks = []
        
        # For efficient gain computation, track for each (partial_block, candidate_point)
        # how many uncovered t-subsets would be covered
        # We'll use a different approach: for each t-subset, track which points cover it
        
        # Actually, let's use the standard greedy approach but with precomputation
        # For a partial block B and candidate x, gain = number of uncovered t-subsets
        # that contain x and are subsets of B ∪ {x}
        
        # More efficient: for each t-subset s, it's covered by block B if s ⊆ B
        # When adding x to B, the newly covered t-subsets are those s ⊆ B∪{x} with x ∈ s
        # i.e., s' = s \ {x} ⊆ B and |s'| = t-1
        
        # Precompute for each (t-1)-subset, which points extend it to a t-subset
        tminus1_to_points = defaultdict(list)
        for i, ts in enumerate(all_tsubs):
            # For each (t-1)-subset of ts, the remaining point extends it
            for j in range(t):
                tminus1 = tuple(ts[:j] + ts[j+1:])
                tminus1_to_points[tminus1].append((ts[j], i))
        
        # Now greedy
        while uncovered:
            if time.time() > deadline - 5:
                return None  # timeout
            
            # Pick the first uncovered t-subset (or randomize)
            # For deterministic behavior, pick min; for randomized, pick random
            if seed_offset == 0:
                # Deterministic: pick lexicographically smallest
                min_idx = min(uncovered)
            else:
                # Randomized: pick random uncovered
                min_idx = rng.choice(list(uncovered))
            
            # Start block with the t-subset
            start_tsub = all_tsubs[min_idx]
            block = list(start_tsub)
            
            # Mark this t-subset as covered
            uncovered.discard(min_idx)
            
            # Add points to fill the block
            while len(block) < k:
                if time.time() > deadline - 5:
                    return None
                
                # For each candidate point x not in block, compute gain
                # gain(x) = number of uncovered t-subsets s such that x ∈ s and s\{x} ⊆ block
                # s\{x} is a (t-1)-subset of block, and x extends it to s
                
                best_x = None
                best_gain = -1
                
                # Collect all (t-1)-subsets of block
                block_set = set(block)
                tminus1_subs = set(itertools.combinations(block, t-1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Count uncovered t-subsets that contain x and whose (t-1) part is in block
                    gain = 0
                    # For each (t-1)-subset in tminus1_subs, check if x extends it
                    # We need: (tminus1, x) is a t-subset and that t-subset is uncovered
                    for tmin1 in tminus1_subs:
                        # Check if (tmin1 ∪ {x}) is a valid t-subset
                        # tmin1 is already sorted, so we need to insert x
                        ts_candidate = tuple(sorted(tmin1 + (x,)))
                        idx = tsub_to_idx.get(ts_candidate)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    
                    # Early termination if we can't do better
                    # (optional optimization)
                
                if best_x is None:
                    # No point improves coverage; pick arbitrary
                    remaining = [x for x in range(v) if x not in block_set]
                    if not remaining:
                        break
                    best_x = rng.choice(remaining)
                
                block.append(best_x)
                block_set.add(best_x)
                
                # Remove newly covered t-subsets from uncovered
                # A t-subset is newly covered if it's a subset of the new block
                # and was previously uncovered
                for ts in itertools.combinations(block, t):
                    idx = tsub_to_idx.get(ts)
                    if idx is not None:
                        uncovered.discard(idx)
            
            block.sort()
            blocks.append(tuple(block))
        
        return blocks

    def local_search_remove(blocks):
        """Try to remove blocks that are redundant."""
        blocks = list(blocks)
        if not blocks:
            return blocks
        
        # For each t-subset, track which blocks cover it
        tsub_to_blocks = defaultdict(list)
        for bi, block in enumerate(blocks):
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None:
                    tsub_to_blocks[idx].append(bi)
        
        # Try to remove blocks one by one
        changed = True
        while changed:
            changed = False
            for bi in range(len(blocks)):
                if bi >= len(blocks):
                    break
                block = blocks[bi]
                # Check if removing this block leaves any t-subset uncovered
                can_remove = True
                for ts in itertools.combinations(block, t):
                    idx = tsub_to_idx.get(ts)
                    if idx is not None:
                        covering_blocks = [b for b in tsub_to_blocks[idx] if b != bi]
                        if not covering_blocks:
                            can_remove = False
                            break
                
                if can_remove:
                    # Remove this block
                    del blocks[bi]
                    # Update tsub_to_blocks
                    for ts in itertools.combinations(block, t):
                        idx = tsub_to_idx.get(ts)
                        if idx is not None:
                            tsub_to_blocks[idx] = [b for b in tsub_to_blocks[idx] if b != bi]
                    changed = True
                    break  # Restart from beginning
        
        return blocks

    # Try multiple strategies
    best_blocks = None
    best_count = float('inf')
    
    # Strategy 1: Deterministic greedy
    result = greedy_cover(0)
    if result is not None:
        result = local_search_remove(result)
        if len(result) < best_count:
            best_count = len(result)
            best_blocks = result
    
    # Strategy 2: Randomized greedy with different seeds
    for seed in range(1, 10):
        if time.time() > deadline - 10:
            break
        result = greedy_cover(seed)
        if result is not None:
            result = local_search_remove(result)
            if len(result) < best_count:
                best_count = len(result)
                best_blocks = result
    
    if best_blocks is None:
        # Fallback: return empty (shouldn't happen)
        return []
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
