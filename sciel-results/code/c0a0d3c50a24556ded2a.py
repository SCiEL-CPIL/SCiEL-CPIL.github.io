"""Simulated Annealing + greedy polish for covering design C(v,k,t)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from math import log, exp

# Pre-compute all t-subsets for the specific task parameters
# We'll build this lazily / cached per (v,k,t)

def _all_tsubsets(v, t):
    return list(itertools.combinations(range(v), t))

def _block_tsubsets(block, t):
    return set(itertools.combinations(block, t))

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill
    
    all_tsubs = _all_tsubsets(v, t)
    n_tsubs = len(all_tsubs)
    
    # Index each t-subset to its index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Greedy initialization
    def greedy_init():
        uncovered = set(range(n_tsubs))
        blocks = []
        while uncovered:
            # Pick the first uncovered t-subset
            first_ts = min(uncovered)
            block = list(all_tsubs[first_ts])
            while len(block) < k:
                # Find point x that covers most uncovered t-subsets
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Count uncovered t-subsets that would be covered by adding x
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if tsub_to_idx.get(ts, -1) in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(tuple(block))
            # Remove covered t-subsets
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None:
                    uncovered.discard(idx)
        return blocks
    
    blocks = greedy_init()
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    # Compute coverage count for each t-subset
    def compute_coverage(blocks):
        cov = [0] * n_tsubs
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None:
                    cov[idx] += 1
        return cov
    
    def is_covered(blocks):
        cov = compute_coverage(blocks)
        return all(c > 0 for c in cov)
    
    def count_uncovered(blocks):
        cov = compute_coverage(blocks)
        return sum(1 for c in cov if c == 0)
    
    # Simulated Annealing
    # State: list of blocks (tuples)
    # Moves: 
    #   1. Remove a redundant block (if it doesn't leave any t-subset uncovered)
    #   2. Add a new block (greedy for the most uncovered t-subsets)
    #   3. Swap a block with a better one
    
    def try_remove_block(blocks, block_idx):
        """Try removing block at block_idx. Return (new_blocks, feasible) if removal is valid."""
        # Check if block is redundant
        # A block is redundant if every t-subset it covers is covered by at least one other block
        cov = compute_coverage(blocks)
        block = blocks[block_idx]
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts)
            if idx is not None and cov[idx] == 1:
                return None, False  # Not redundant
        new_blocks = blocks[:block_idx] + blocks[block_idx+1:]
        return new_blocks, True
    
    def greedy_add_block(blocks):
        """Add a block that covers the most uncovered t-subsets."""
        cov = compute_coverage(blocks)
        uncovered = [i for i, c in enumerate(cov) if c == 0]
        if not uncovered:
            return None
        # Pick the most uncovered t-subset
        # Actually, pick a t-subset and build a block around it
        first_ts = all_tsubs[uncovered[0]]
        block = list(first_ts)
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(ts, -1)
                    if idx is not None and cov[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        block.sort()
        return tuple(block)
    
    # SA loop
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    current_blocks = list(blocks)
    current_uncovered = count_uncovered(current_blocks)
    
    sa_start = time.time()
    max_sa_time = 60  # spend up to 60s on SA
    
    while time.time() - sa_start < max_sa_time and T > T_min:
        # Random move
        r = random.random()
        
        if len(current_blocks) > 1 and r < 0.5:
            # Try removing a random block
            idx = random.randint(0, len(current_blocks) - 1)
            new_blocks, feasible = try_remove_block(current_blocks, idx)
            if feasible:
                new_uncovered = 0  # Should still be covered
                # Accept
                current_blocks = new_blocks
                current_uncovered = 0
                if len(current_blocks) < best_count:
                    best_count = len(current_blocks)
                    best_blocks = list(current_blocks)
        elif r < 0.8:
            # Try adding a block (only if we can later remove)
            new_block = greedy_add_block(current_blocks)
            if new_block is None:
                continue
            new_blocks = current_blocks + [new_block]
            # This increases count, so we only accept if it helps us remove more later
            # For now, skip pure additions in SA; rely on swaps
            pass
        else:
            # Swap a random block with a greedy replacement
            if not current_blocks:
                continue
            idx = random.randint(0, len(current_blocks) - 1)
            # Try to find a better replacement
            # Remove block idx temporarily and see what's uncovered
            temp_blocks = current_blocks[:idx] + current_blocks[idx+1:]
            cov = compute_coverage(temp_blocks)
            uncovered = [i for i, c in enumerate(cov) if c == 0]
            if not uncovered:
                # Block was redundant, remove it
                current_blocks = temp_blocks
                if len(current_blocks) < best_count:
                    best_count = len(current_blocks)
                    best_blocks = list(current_blocks)
                continue
            # Build a new block to cover uncovered
            first_ts = all_tsubs[uncovered[0]]
            new_block = list(first_ts)
            while len(new_block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in new_block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(new_block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx2 = tsub_to_idx.get(ts, -1)
                        if idx2 is not None and cov[idx2] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                new_block.append(best_x)
            new_block.sort()
            new_block = tuple(new_block)
            
            # Check if new block is better than old
            old_block = current_blocks[idx]
            # Count how many uncovered t-subsets the new block covers
            new_covers = 0
            for ts in itertools.combinations(new_block, t):
                idx2 = tsub_to_idx.get(ts)
                if idx2 is not None and cov[idx2] == 0:
                    new_covers += 1
            
            # Count how many t-subsets the old block uniquely covered
            old_unique = 0
            for ts in itertools.combinations(old_block, t):
                idx2 = tsub_to_idx.get(ts)
                if idx2 is not None and cov[idx2] == 0:
                    old_unique += 1
            
            # Accept if new covers more uncovered
            if new_covers > old_unique:
                current_blocks[idx] = new_block
                current_uncovered = count_uncovered(current_blocks)
                if len(current_blocks) < best_count:
                    best_count = len(current_blocks)
                    best_blocks = list(current_blocks)
            else:
                # Accept with probability exp(-(delta)/T)
                delta = old_unique - new_covers
                if delta > 0 and random.random() < exp(-delta / T):
                    current_blocks[idx] = new_block
                    current_uncovered = count_uncovered(current_blocks)
        
        T *= cooling
    
    # Final greedy polish: try to remove redundant blocks
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        cov = compute_coverage(best_blocks)
        for i in range(len(best_blocks) - 1, -1, -1):
            block = best_blocks[i]
            redundant = True
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None and cov[idx] == 1:
                    redundant = False
                    break
            if redundant:
                best_blocks.pop(i)
                changed = True
                break
    
    # Convert to lists
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
