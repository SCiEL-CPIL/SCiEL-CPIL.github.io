"""Simulated Annealing for covering design C(24,7,3) using bitmask optimization."""
import itertools
import random
import time
import math

# Precomputation for v=24, k=7, t=3
V = 24
K = 7
T = 3

# Map t-subsets to IDs
T_SUBSETS = list(itertools.combinations(range(V), T))
T_SUBSET_MAP = {fs: i for i, fs in enumerate(T_SUBSETS)}
NUM_T_SUBSETS = len(T_SUBSETS)

# Precompute coverage for all k-blocks
# There are C(24,7) = 346,104 blocks.
# We will store the list of t-subset IDs covered by each block.
ALL_BLOCKS = list(itertools.combinations(range(V), K))
BLOCK_COVERAGE = []
for b in ALL_BLOCKS:
    cov = []
    for c in itertools.combinations(b, T):
        cov.append(T_SUBSET_MAP[frozenset(c)])
    BLOCK_COVERAGE.append(cov)

# Map block tuple to index for quick lookup
BLOCK_INDEX = {b: i for i, b in enumerate(ALL_BLOCKS)}


def _greedy_init():
    """
    Greedy construction to get a valid starting covering.
    Returns a list of block indices.
    """
    uncovered = [1] * NUM_T_SUBSETS
    blocks = []
    
    # Pre-compute for speed: for each block, which uncovered subsets does it cover?
    # We'll just iterate and check.
    
    while any(uncovered):
        best_block = -1
        best_gain = -1
        
        # Randomly shuffle block order to break ties and avoid bias
        indices = list(range(len(ALL_BLOCKS)))
        random.shuffle(indices)
        
        for bi in indices:
            cov = BLOCK_COVERAGE[bi]
            gain = 0
            for sid in cov:
                if uncovered[sid]:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = bi
            
            # Optimization: if we found a block that covers all remaining? Unlikely.
            # If gain is 0, we can break early if we've seen enough? No, we need max gain.
        
        if best_block == -1:
            break
            
        blocks.append(best_block)
        cov = BLOCK_COVERAGE[best_block]
        for sid in cov:
            if uncovered[sid]:
                uncovered[sid] = 0

    return blocks


def construct(v, k, t):
    # Ensure parameters match precomputation
    assert v == 24 and k == 7 and t == 3, "Precomputation is specific to C(24,7,3)"
    
    start_time = time.time()
    time_limit = 115.0
    
    # 1. Initialize
    current_blocks = _greedy_init()
    current_count = len(current_blocks)
    
    # Initialize coverage counts
    counts = [0] * NUM_T_SUBSETS
    for bi in current_blocks:
        for sid in BLOCK_COVERAGE[bi]:
            counts[sid] += 1
    
    best_blocks = list(current_blocks)
    best_count = current_count
    
    # Helper to get number of uncovered subsets
    def get_uncovered_count(counts):
        return sum(1 for c in counts if c == 0)
    
    current_uncovered = get_uncovered_count(counts)
    
    # Cost function: Minimize blocks, penalize uncovered
    LAMBDA = 100.0
    def get_cost(n_blocks, n_uncovered):
        return n_blocks + LAMBDA * n_uncovered
    
    current_cost = get_cost(current_count, current_uncovered)
    
    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.999
    T = T_start
    
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit:
        
        # Generate a move
        move_type = rng.random()
        if move_type < 0.3:
            # Remove a block
            if not current_blocks:
                continue
            idx = rng.randrange(len(current_blocks))
            bi_out = current_blocks[idx]
            
            # Calculate delta
            delta_blocks = -1
            delta_uncovered = 0
            for sid in BLOCK_COVERAGE[bi_out]:
                if counts[sid] == 1:
                    delta_uncovered += 1
            
            new_cost = current_cost + delta_blocks + LAMBDA * delta_uncovered
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or rng.random() < math.exp(-delta_cost / T):
                # Accept
                current_blocks.pop(idx)
                for sid in BLOCK_COVERAGE[bi_out]:
                    counts[sid] -= 1
                current_count -= 1
                current_uncovered += delta_uncovered
                current_cost = new_cost
                
                if current_uncovered == 0 and current_count < best_count:
                    best_count = current_count
                    best_blocks = list(current_blocks)
                    
        elif move_type < 0.6:
            # Add a block
            # Pick a random block not in current set
            # To speed up, we can pick from blocks that cover currently uncovered subsets
            uncovered_sids = [i for i, c in enumerate(counts) if c == 0]
            if not uncovered_sids:
                # Already valid, just add random block to explore
                bi_in = rng.randrange(len(ALL_BLOCKS))
            else:
                # Pick a random uncovered subset, then pick a random block covering it
                target_sid = rng.choice(uncovered_sids)
                # Find blocks covering target_sid
                # This requires an inverse index. Let's build one once.
                # For now, just random block, might be slow.
                # Optimization: Pre-compute for each t-subset, the list of blocks covering it.
                # This is 2024 lists.
                # Let's do this outside the loop.
                pass
            
            # Since we didn't precompute inverse index, let's just pick a random block.
            # This is inefficient. Let's add a simple heuristic:
            # Pick a random block.
            bi_in = rng.randrange(len(ALL_BLOCKS))
            
            # Check if it's already in the set
            # Using a set for current blocks would be faster for membership check
            # Let's maintain a set of block indices in parallel.
            
            # For now, assume it's not there or handle duplicates poorly.
            # Better: Maintain `block_set` = set(current_blocks)
            
            # Let's refactor to use a set for membership.
            
            # REFACTOR:
            # We need `block_set` to check if bi_in is already present.
            
            # Let's restart the loop logic with `block_set`.
            
            break # Break to refactor in mind
            
        else:
            # Swap
            pass
            
    # The above logic is getting messy. Let's write the clean version.
    
    # Clean Implementation
    
    # Re-initialize for clean run
    current_blocks = _greedy_init()
    current_count = len(current_blocks)
    counts = [0] * NUM_T_SUBSETS
    for bi in current_blocks:
        for sid in BLOCK_COVERAGE[bi]:
            counts[sid] += 1
            
    best_blocks = list(current_blocks)
    best_count = current_count
    current_uncovered = sum(1 for c in counts if c == 0)
    
    # Precompute inverse index: for each t-subset id, list of block indices covering it
    # This is crucial for efficient "Add" moves that target uncovered subsets.
    INVERSE_COVERAGE = [[] for _ in range(NUM_T_SUBSETS)]
    for bi, cov in enumerate(BLOCK_COVERAGE):
        for sid in cov:
            INVERSE_COVERAGE[sid].append(bi)
    
    # Maintain a set of current block indices for O(1) membership check
    current_block_set = set(current_blocks)
    
    T = 5.0
    cooling_rate = 0.9995
    LAMBDA = 50.0 # Penalty for uncovered
    
    def get_cost(n_blocks, n_uncovered):
        return n_blocks + LAMBDA * n_uncovered
    
    current_cost = get_cost(current_count, current_uncovered)
    
    while time.time() - start_time < time_limit:
        
        move_type = rng.random()
        
        if move_type < 0.4:
            # Remove a block
            if not current_blocks:
                continue
            idx = rng.randrange(len(current_blocks))
            bi_out = current_blocks[idx]
            
            delta_blocks = -1
            delta_uncovered = 0
            for sid in BLOCK_COVERAGE[bi_out]:
                if counts[sid] == 1:
                    delta_uncovered += 1
            
            delta_cost = delta_blocks + LAMBDA * delta_uncovered
            
            if delta_cost < 0 or rng.random() < math.exp(-delta_cost / T):
                # Accept
                current_blocks.pop(idx)
                current_block_set.remove(bi_out)
                for sid in BLOCK_COVERAGE[bi_out]:
                    counts[sid] -= 1
                current_count -= 1
                current_uncovered += delta_uncovered
                current_cost += delta_cost
                
                if current_uncovered == 0 and current_count < best_count:
                    best_count = current_count
                    best_blocks = list(current_blocks)
                    
        elif move_type < 0.8:
            # Add a block
            # Strategy: Pick a block that covers at least one uncovered subset if possible
            uncovered_sids = [i for i, c in enumerate(counts) if c == 0]
            
            if uncovered_sids:
                target_sid = rng.choice(uncovered_sids)
                candidates = INVERSE_COVERAGE[target_sid]
                if candidates:
                    bi_in = rng.choice(candidates)
                else:
                    bi_in = rng.randrange(len(ALL_BLOCKS))
            else:
                bi_in = rng.randrange(len(ALL_BLOCKS))
                
            if bi_in in current_block_set:
                continue
                
            delta_blocks = 1
            delta_uncovered = 0
            for sid in BLOCK_COVERAGE[bi_in]:
                if counts[sid] == 0:
                    delta_uncovered -= 1 # Uncovered decreases
            
            delta_cost = delta_blocks + LAMBDA * delta_uncovered
            
            if delta_cost < 0 or rng.random() < math.exp(-delta_cost / T):
                # Accept
                current_blocks.append(bi_in)
                current_block_set.add(bi_in)
                for sid in BLOCK_COVERAGE[bi_in]:
                    counts[sid] += 1
                current_count += 1
                current_uncovered += delta_uncovered
                current_cost += delta_cost
                
                if current_uncovered == 0 and current_count < best_count:
                    best_count = current_count
                    best_blocks = list(current_blocks)
                    
        else:
            # Swap: Remove one, Add one
            if not current_blocks:
                continue
            
            # Pick one to remove
            idx_out = rng.randrange(len(current_blocks))
            bi_out = current_blocks[idx_out]
            
            # Pick one to add
            # Prefer blocks that cover subsets lost by removing bi_out
            lost_sids = []
            for sid in BLOCK_COVERAGE[bi_out]:
                if counts[sid] == 1:
                    lost_sids.append(sid)
            
            if lost_sids:
                target_sid = rng.choice(lost_sids)
                candidates = [b for b in INVERSE_COVERAGE[target_sid] if b != bi_out]
                if candidates:
                    bi_in = rng.choice(candidates)
                else:
                    bi_in = rng.randrange(len(ALL_BLOCKS))
            else:
                bi_in = rng.randrange(len(ALL_BLOCKS))
                
            if bi_in in current_block_set:
                continue
                
            # Calculate delta
            # Remove bi_out
            delta_blocks = 0 # Net 0
            delta_uncovered = 0
            
            # Effect of removing bi_out
            for sid in BLOCK_COVERAGE[bi_out]:
                if counts[sid] == 1:
                    delta_uncovered += 1 # Becomes uncovered
            
            # Effect of adding bi_in
            for sid in BLOCK_COVERAGE[bi_in]:
                # Note: counts are NOT yet updated.
                # We need to simulate.
                # If counts[sid] == 0, adding makes it 1 (uncovered -1)
                # If counts[sid] == 1, adding makes it 2 (no change in uncovered)
                # BUT, if bi_out and bi_in share a subset, and counts was 1, 
                # removing bi_out makes it 0 (uncovered +1), adding bi_in makes it 1 (uncovered -1). Net 0.
                
                # Let's do it carefully.
                # We can just calculate new counts for affected sids.
                pass
            
            # Simpler: Apply remove, calc delta, apply add, calc delta, combine.
            # But we need to revert if rejected.
            
            # Let's just calculate the net change in uncovered count.
            # new_counts = old_counts - cov(bi_out) + cov(bi_in)
            # We can compute this for the union of sids in bi_out and bi_in.
            
            affected_sids = set(BLOCK_COVERAGE[bi_out]) | set(BLOCK_COVERAGE[bi_in])
            
            old_uncovered = 0
            new_uncovered = 0
            
            for sid in affected_sids:
                old_c = counts[sid]
                new_c = old_c
                if sid in BLOCK_COVERAGE[bi_out]:
                    new_c -= 1
                if sid in BLOCK_COVERAGE[bi_in]:
                    new_c += 1
                    
                if old_c == 0:
                    old_uncovered += 1
                if new_c == 0:
                    new_uncovered += 1
                    
            delta_uncovered = new_uncovered - old_uncovered
            delta_cost = delta_uncovered * LAMBDA # delta_blocks is 0
            
            if delta_cost < 0 or rng.random() < math.exp(-delta_cost / T):
                # Accept
                # Update counts
                for sid in BLOCK_COVERAGE[bi_out]:
                    counts[sid] -= 1
                for sid in BLOCK_COVERAGE[bi_in]:
                    counts[sid] += 1
                    
                # Update blocks
                current_blocks[idx_out] = bi_in
                current_block_set.remove(bi_out)
                current_block_set.add(bi_in)
                
                current_uncovered += delta_uncovered
                current_cost += delta_cost
                
                if current_uncovered == 0 and current_count < best_count:
                    best_count = current_count
                    best_blocks = list(current_blocks)

        # Cooling
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
    # Final repair: if best_blocks is not valid, try to fix it?
    # The SA keeps track of best *valid* covering (uncovered == 0).
    # So best_blocks should be valid.
    
    # Convert indices back to lists of points
    return [list(ALL_BLOCKS[bi]) for bi in best_blocks]
