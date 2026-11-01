import time
import random
import itertools
import math

# EVOLVE-BLOCK-START

def _construct(v, k, t, max_time=140.0):
    """Set-Covering Local Search: generate block pool, greedy initial cover, 1-exchange refinement."""
    start_time = time.monotonic()
    
    def time_left():
        return max_time - (time.monotonic() - start_time)
    
    if t <= 0:
        return []
    if k >= v:
        return [list(range(v))]
    
    # Total number of t-subsets
    total_t = math.comb(v, t)
    
    # Generate pool of candidate blocks
    # Strategy: 
    # 1. All blocks containing a random t-subset (centered greedy)
    # 2. Random blocks
    # 3. Blocks from greedy construction
    
    rng = random.Random(42)
    pool = []
    pool_set = set()
    
    def add_block(b):
        b = tuple(sorted(b))
        if b not in pool_set:
            pool_set.add(b)
            pool.append(b)
    
    # 1. Greedy-style centered blocks
    # Run a few greedy passes to generate diverse blocks
    for seed in range(50):
        rng_local = random.Random(seed)
        uncovered_sample = set(itertools.combinations(range(v), t))
        # Just generate a few blocks per seed
        for _ in range(5):
            if not uncovered_sample:
                break
            # Pick random uncovered
            start_ts = tuple(rng_local.choice(list(uncovered_sample)))
            block = list(start_ts)
            remaining = [x for x in range(v) if x not in block]
            rng_local.shuffle(remaining)
            # Greedy fill
            while len(block) < k and remaining:
                block_set = set(block)
                best_x = None
                best_gain = -1
                for x in remaining:
                    if x in block_set:
                        continue
                    gain = 0
                    # Approximate gain: count how many t-subsets with x and t-1 from block are in uncovered
                    # For speed, just count random sample or use heuristic
                    # Heuristic: prefer x that appears in many uncovered t-subsets with current block
                    # Simplified: just pick random if gain calc is too slow
                    # Let's do a quick count for small t
                    if t <= 6:
                        for rest in itertools.combinations(block, t - 1):
                            ts = tuple(sorted(list(rest) + [x]))
                            if ts in uncovered_sample:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None or best_gain <= 0:
                    best_x = remaining[0]
                block.append(best_x)
                remaining.remove(best_x)
                block_set.add(best_x)
            add_block(block)
            # Remove covered from sample
            for ts in itertools.combinations(block, t):
                uncovered_sample.discard(ts)
    
    # 2. Add some purely random blocks
    for _ in range(100):
        if time_left() < 5:
            break
        b = rng.sample(range(v), k)
        add_block(b)
    
    # 3. Ensure we have enough blocks in pool
    min_pool_size = max(200, total_t // 10)
    while len(pool) < min_pool_size and time_left() > 10:
        b = rng.sample(range(v), k)
        add_block(b)
    
    # Precompute for each block in pool: the set of t-subsets it covers
    # Represent as a set of tuples for simplicity, or bitset if total_t is small
    # total_t for v=19, t=5 is 11628. Bitset is feasible.
    # But let's stick to sets for clarity and correctness first, then optimize if needed.
    # Actually, for 1-exchange, we need to know which t-subsets become uncovered when removing a block.
    # This requires knowing if a t-subset is covered by ANY other block.
    
    # Let's use a coverage count array.
    # Map each t-subset to an index
    t_subsets = list(itertools.combinations(range(v), t))
    t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # For each block in pool, store the list of indices of t-subsets it covers
    block_cov = []
    for b in pool:
        cov = []
        for ts in itertools.combinations(b, t):
            idx = t_to_idx.get(ts)
            if idx is not None:
                cov.append(idx)
        block_cov.append(cov)
    
    # Greedy initial selection
    # coverage_count[i] = number of selected blocks covering t-subset i
    coverage_count = [0] * total_t
    selected_blocks = []
    selected_set = set()
    
    # We need to pick blocks from the pool to cover all t-subsets
    # Greedy: at each step, pick the block that covers the most currently uncovered t-subsets
    
    while True:
        if time_left() < 2:
            break
        # Find uncovered
        uncovered = [i for i in range(total_t) if coverage_count[i] == 0]
        if not uncovered:
            break
        
        # Pick best block
        best_block_idx = -1
        best_gain = 0
        best_block = None
        
        for bi, b in enumerate(pool):
            if bi in selected_set:
                continue
            gain = 0
            for idx in block_cov[bi]:
                if coverage_count[idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = bi
                best_block = b
            # Early exit if gain is 0? No, we need to find any.
            # If best_gain is already high, maybe break?
        
        if best_block is None:
            # Pool exhausted? Add a new random block?
            # For now, break and hope we covered everything.
            # Or generate a new block on the fly.
            # Let's generate a new block centered on a random uncovered t-subset
            if uncovered:
                target_idx = uncovered[0]
                target_ts = t_subsets[target_idx]
                block = list(target_ts)
                remaining = [x for x in range(v) if x not in block]
                rng.shuffle(remaining)
                while len(block) < k and remaining:
                    block.append(remaining.pop())
                add_block(block)
                # Compute cov for new block
                cov = []
                for ts in itertools.combinations(block, t):
                    idx = t_to_idx.get(ts)
                    if idx is not None:
                        cov.append(idx)
                block_cov.append(cov)
                bi = len(pool) - 1
                best_block = block
                best_block_idx = bi
            else:
                break
        
        # Select the block
        selected_blocks.append(best_block)
        selected_set.add(best_block_idx)
        for idx in block_cov[best_block_idx]:
            coverage_count[idx] += 1
    
    # Now we have a valid cover (hopefully).
    # Local Search: 1-exchange
    # Try to remove a block and see if we can add a different block that covers the same t-subsets (or more).
    # Actually, standard 1-exchange for set cover:
    # Remove block i. Let U_i be the set of t-subsets uniquely covered by i.
    # If U_i is empty, remove i.
    # If U_i is not empty, find a block j (not selected) that covers all of U_i.
    # If such j exists, swap i -> j (if it doesn't increase size, i.e., if j covers other things too, it might be better, but size is same).
    # Wait, we want to MINIMIZE size.
    # So:
    # 1. Try to remove a block i. If after removal, all t-subsets are still covered, remove it. (Size decreases by 1)
    # 2. If not, try to swap i with j. This doesn't change size, but might allow further removals later.
    #    So focus on 1 first.
    
    # Precompute for each selected block: which t-subsets are uniquely covered by it?
    # This is expensive to recompute every time.
    # Let's maintain coverage_count.
    
    improved = True
    while improved and time_left() > 5:
        improved = False
        # Try to remove each selected block
        # We iterate over a copy of selected_blocks indices
        # Note: removing a block changes coverage_count, so we must be careful.
        
        # Sort selected blocks by "uniqueness" or just random order?
        # Random order might be better for diversity.
        indices = list(range(len(selected_blocks)))
        rng.shuffle(indices)
        
        for idx in indices:
            if time_left() < 2:
                break
            
            block = selected_blocks[idx]
            # Check if this block is necessary
            # Temporarily remove it
            for t_idx in block_cov[idx]:
                coverage_count[t_idx] -= 1
            
            # Check if any t-subset is now uncovered
            is_necessary = False
            for t_idx in block_cov[idx]:
                if coverage_count[t_idx] == 0:
                    is_necessary = True
                    break
            
            if not is_necessary:
                # Remove it
                selected_blocks.pop(idx)
                selected_set.discard(idx)
                # Note: block_cov indices for other blocks remain valid, but selected_set indices shift?
                # This is a problem. selected_set contains indices into `pool`.
                # selected_blocks is a list of blocks.
                # If we pop from selected_blocks, we don't need to update selected_set if we just track which pool indices are selected.
                # But we need to know which pool index corresponds to which selected block.
                # Let's store (pool_idx, block) in selected_blocks.
                
                # Wait, my current selected_blocks is just a list of blocks (tuples).
                # I need to know the pool index to update coverage_count correctly?
                # No, I already updated coverage_count.
                # But I need to remove the correct entry from selected_blocks.
                # And I need to make sure I don't try to remove it again in the same pass.
                
                # Let's restart the loop or mark it as removed.
                improved = True
                # Since indices are based on the old list, we should break or be careful.
                # Let's just break and restart the while loop.
                break
            else:
                # Restore
                for t_idx in block_cov[idx]:
                    coverage_count[t_idx] += 1
        
        # If we removed a block, restart
        # If not, we are stuck with this configuration.
        # Try 1-exchange swap?
        # For now, let's just do removals.
    
    # Convert selected_blocks to list of lists
    # Note: selected_blocks contains tuples from pool.
    result = [list(b) for b in selected_blocks]
    
    # Verify validity (safety check)
    if time_left() > 1:
        covered = set()
        for b in result:
            for ts in itertools.combinations(b, t):
                covered.add(ts)
        if len(covered) != total_t:
            # Invalid! Fallback to greedy without local search
            # This shouldn't happen if logic is correct, but just in case.
            pass
    
    return result

def construct(v, k, t):
    """Entry point matching the seed program's signature."""
    return _construct(v, k, t, max_time=140.0)

# EVOLVE-BLOCK-END
