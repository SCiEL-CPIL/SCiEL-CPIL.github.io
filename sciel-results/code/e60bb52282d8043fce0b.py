"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _greedy_cover(v, k, t, rng):
    """Randomized greedy set cover: pick uncovered t-subset, build block greedily."""
    all_tsubs = set(itertools.combinations(range(v), t))
    uncovered = set(all_tsubs)
    blocks = []
    while uncovered:
        # Pick a random uncovered t-subset (or the first one for determinism seed)
        if rng is not None and rng.random() < 0.3:
            # Random selection to diversify
            candidates = list(uncovered)
            target = candidates[rng.randrange(len(candidates))]
        else:
            target = min(uncovered)
        
        block = list(target)
        block_set = set(block)
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Compute gain: how many uncovered t-subsets would be covered
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_tsub = tuple(sorted(rest + (x,)))
                    if new_tsub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1 or best_gain == 0:
                # Pick a random unused point
                unused = [x for x in range(v) if x not in block_set]
                if unused:
                    best_x = unused[rng.randrange(len(unused))] if rng else unused[0]
                else:
                    break
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        blocks.append(block)
        # Remove all t-subsets covered by this block
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
    
    return blocks


def _local_search(v, k, t, blocks, rng, time_limit=110.0):
    """Local search: try to remove redundant blocks and re-cover gaps."""
    start_time = time.time()
    blocks = [list(b) for b in blocks]
    
    # Precompute coverage counts
    def build_coverage(blocks):
        """Return dict: t-subset -> count of blocks covering it."""
        cov = {}
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cov[ts] = cov.get(ts, 0) + 1
        return cov
    
    cov = build_coverage(blocks)
    
    while time.time() - start_time < time_limit:
        # Try to remove a block that doesn't make any t-subset uncovered
        removed_any = False
        # Shuffle blocks to try different orders
        indices = list(range(len(blocks)))
        if rng:
            rng.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time > time_limit:
                break
            block_i = blocks[i]
            # Check if removing this block leaves all t-subsets covered
            can_remove = True
            for ts in itertools.combinations(block_i, t):
                if cov[ts] <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove block i
                for ts in itertools.combinations(block_i, t):
                    cov[ts] -= 1
                blocks.pop(i)
                removed_any = True
                # Restart the scan from beginning
                break
        
        if not removed_any:
            # Try a more aggressive move: remove a block and re-cover gaps
            # Pick a random block to try removing
            if len(blocks) > 0:
                i = rng.randrange(len(blocks)) if rng else 0
                block_i = blocks[i]
                
                # Simulate removal
                gap_tsubs = []
                for ts in itertools.combinations(block_i, t):
                    if cov[ts] <= 1:
                        gap_tsubs.append(ts)
                
                if gap_tsubs:
                    # Try to cover the gaps with a new block
                    # Greedily build a block to cover as many gap t-subsets as possible
                    gap_set = set(gap_tsubs)
                    new_block = []
                    new_block_set = set()
                    
                    # Start with a random gap t-subset
                    target = gap_set.pop() if gap_set else None
                    if target:
                        new_block = list(target)
                        new_block_set = set(new_block)
                        gap_set.discard(target)
                    
                    while len(new_block) < k and gap_set:
                        best_x = -1
                        best_gain = -1
                        for x in range(v):
                            if x in new_block_set:
                                continue
                            gain = 0
                            for rest in itertools.combinations(new_block, t - 1):
                                new_ts = tuple(sorted(rest + (x,)))
                                if new_ts in gap_set:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x == -1 or best_gain == 0:
                            break
                        new_block.append(best_x)
                        new_block_set.add(best_x)
                        # Update gap_set
                        for rest in itertools.combinations(new_block, t):
                            gap_set.discard(tuple(sorted(rest)))
                    
                    # Check if this new block covers all gaps
                    new_block.sort()
                    covered_new = set(itertools.combinations(new_block, t))
                    if gap_set.issubset(covered_new):
                        # Accept: remove old block, add new one
                        for ts in itertools.combinations(block_i, t):
                            cov[ts] -= 1
                        blocks.pop(i)
                        for ts in itertools.combinations(new_block, t):
                            cov[ts] = cov.get(ts, 0) + 1
                        blocks.append(new_block)
                        removed_any = True
                        break
        
        if not removed_any:
            # No improvement found; try a few more random restarts of the scan
            break
    
    return blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using randomized greedy + local search."""
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Multiple randomized greedy restarts
    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
        rng = random.Random(r * 7919 + 13)
        blocks = _greedy_cover(v, k, t, rng)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Local search on the best solution
    if best_blocks is not None:
        if time.time() < deadline - 5:
            rng = random.Random(42)
            remaining = min(105.0, deadline - time.time() - 3.0)
            best_blocks = _local_search(v, k, t, best_blocks, rng, time_limit=remaining)
    
    # Final validation and cleanup
    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = _greedy_cover(v, k, t, None)
    
    # Ensure blocks are distinct and valid
    seen = set()
    final_blocks = []
    for b in best_blocks:
        key = tuple(sorted(b))
        if key not in seen and len(set(b)) == k and all(0 <= x < v for x in b):
            seen.add(key)
            final_blocks.append(list(b))
    
    return final_blocks
# EVOLVE-BLOCK-END
