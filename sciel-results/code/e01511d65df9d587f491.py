"""Improved covering design search for C(19,8,5) with deadline guard."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import math


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # All t-subsets of v points
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}

    def elapsed():
        return time.time() - start_time

    def run_greedy_v2(seed):
        rng = random.Random(seed)
        is_uncovered = [True] * n_tsubsets
        blocks = []
        
        # Helper to get a random uncovered index
        def get_random_uncovered():
            # Retry a few times
            for _ in range(20):
                idx = rng.randint(0, n_tsubsets - 1)
                if is_uncovered[idx]:
                    return idx
            # If failed, scan
            for i in range(n_tsubsets):
                if is_uncovered[i]:
                    return i
            return -1

        while True:
            if elapsed() > deadline:
                break
            
            idx = get_random_uncovered()
            if idx == -1:
                break
                
            core_ts = all_tsubsets[idx]
            block = list(core_ts)
            block_set = set(core_ts)
            
            while len(block) < k:
                best_point = None
                best_gain = -1
                
                for p in range(v):
                    if p in block_set:
                        continue
                    
                    # Calculate gain: number of uncovered t-subsets S such that S subset (block_set U {p}) and p in S
                    # S must contain p and t-1 elements from block_set
                    gain = 0
                    for rest in itertools.combinations(block_set, t - 1):
                        # Create tuple for ts to use as key
                        # rest is a tuple of t-1 elements
                        # We need to add p and sort
                        # Since rest is sorted and p is not in block_set, we can merge
                        # But sorting a small tuple is fine
                        ts_tuple = tuple(sorted(rest + (p,)))
                        ts_idx = tsubset_index[ts_tuple]
                        if is_uncovered[ts_idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                
                if best_point is None:
                    break
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update coverage
            for ts in itertools.combinations(block, t):
                ts_idx = tsubset_index[ts]
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
        
        return blocks

    # Phase 1: Run greedy with multiple seeds to find a good initial solution
    best_blocks = []
    best_len = float('inf')
    
    num_seeds = 100
    for i in range(num_seeds):
        if elapsed() > deadline - 20: # Leave time for local search
            break
        blocks = run_greedy_v2(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    # Phase 2: Local Search (Deletion)
    current_blocks = list(best_blocks)
    
    # Re-verify coverage for current_blocks
    def get_coverage_counts(blocks):
        counts = [0] * n_tsubsets
        for b in blocks:
            for ts in itertools.combinations(b, t):
                counts[tsubset_index[ts]] += 1
        return counts

    counts = get_coverage_counts(current_blocks)
    
    # Try removing blocks
    improved = True
    while improved:
        improved = False
        # Shuffle blocks to try different orders
        rng = random.Random()
        indices = list(range(len(current_blocks)))
        rng.shuffle(indices)
        
        for idx in indices:
            if elapsed() > deadline - 5:
                break
            
            # Try removing block at idx
            b = current_blocks[idx]
            
            # Check if removing b leaves any ts with count 0
            can_remove = True
            for ts in itertools.combinations(b, t):
                ts_idx = tsubset_index[ts]
                if counts[ts_idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove block
                current_blocks.pop(idx)
                # Update counts
                for ts in itertools.combinations(b, t):
                    counts[tsubset_index[ts]] -= 1
                improved = True
                break # Restart loop with new list
                
    if len(current_blocks) < best_len:
        best_len = len(current_blocks)
        best_blocks = current_blocks

    return best_blocks
# EVOLVE-BLOCK-END
