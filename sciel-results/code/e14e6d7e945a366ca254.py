"""Covering design C(24,7,3) - optimized greedy with local search and time budget."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 138.0  # safety margin below 150s kill
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute for each point x, the list of t-subsets containing x
    # This helps with faster gain computation
    point_to_tsubsets = [[] for _ in range(v)]
    for ts in all_t_subsets:
        for x in ts:
            point_to_tsubsets[x].append(ts)
    
    def fast_greedy(seed, mode="random"):
        """Fast greedy covering with optimized gain computation."""
        rng = random.Random(seed)
        uncovered = set(all_t_subsets)
        blocks = []
        
        if not uncovered:
            return blocks
        
        uncovered_list = list(uncovered)
        
        while uncovered:
            if time.time() > deadline - 3.0:
                return blocks
            
            if mode == "random":
                start_ts = rng.choice(uncovered_list)
            else:
                start_ts = uncovered_list[0]
            
            block_set = set(start_ts)
            block = list(start_ts)
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                
                # For each candidate point, compute gain
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Gain = number of uncovered t-subsets that contain x and (t-1) elements from block
                    gain = 0
                    # Pre-computed: for each (t-1) subset of block, count uncovered t-subsets with x
                    # But we need to check if the t-subset is in uncovered
                    # Alternative: iterate over t-subsets containing x and check if rest is in block
                    for ts in point_to_tsubsets[x]:
                        if ts in uncovered:
                            # Check if ts has exactly t-1 elements in block (and x is the new one)
                            rest = [y for y in ts if y != x]
                            if all(y in block_set for y in rest):
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x is None:
                    break
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Remove covered t-subsets
            removed = []
            for ts in itertools.combinations(block, t):
                if ts in uncovered:
                    uncovered.discard(ts)
                    removed.append(ts)
            
            # Update uncovered_list (remove items that were removed)
            if removed:
                removed_set = set(removed)
                uncovered_list = [ts for ts in uncovered_list if ts not in removed_set]
        
        return blocks
    
    def local_search(blocks, seed=0):
        """Try to remove redundant blocks from the covering."""
        rng = random.Random(seed)
        n = len(blocks)
        if n <= 1:
            return blocks
        
        # Build coverage count: for each t-subset, how many blocks cover it
        cover_count = defaultdict(int)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                cover_count[ts] += 1
        
        # A block is removable if all its t-subsets are covered by other blocks
        # i.e., for every ts in block, cover_count[ts] > 1
        removable = []
        for i, block in enumerate(blocks):
            can_remove = True
            for ts in itertools.combinations(block, t):
                if cover_count[ts] <= 1:
                    can_remove = False
                    break
            if can_remove:
                removable.append(i)
        
        # Try removing blocks one at a time (in random order)
        if removable:
            rng.shuffle(removable)
            for idx in removable:
                if time.time() > deadline - 2.0:
                    break
                # Remove block idx
                block = blocks.pop(idx)
                for ts in itertools.combinations(block, t):
                    cover_count[ts] -= 1
                # Check if covering is still valid
                valid = True
                for ts in itertools.combinations(block, t):
                    if cover_count[ts] == 0:
                        valid = False
                        break
                if valid:
                    # Keep it removed
                    # Re-index: no need, we just popped
                    pass
                else:
                    # Put it back
                    blocks.insert(idx, block)
                    for ts in itertools.combinations(block, t):
                        cover_count[ts] += 1
        
        return blocks
    
    def local_search_aggressive(blocks, seed=0):
        """More aggressive local search: try swapping and removing."""
        rng = random.Random(seed)
        n = len(blocks)
        if n <= 2:
            return blocks
        
        # Build coverage count
        cover_count = defaultdict(int)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                cover_count[ts] += 1
        
        improved = True
        while improved:
            if time.time() > deadline - 2.0:
                break
            improved = False
            n = len(blocks)
            
            # Try removing each block
            order = list(range(n))
            rng.shuffle(order)
            for idx in order:
                if time.time() > deadline - 2.0:
                    break
                block = blocks[idx]
                can_remove = True
                for ts in itertools.combinations(block, t):
                    if cover_count[ts] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove
                    blocks.pop(idx)
                    for ts in itertools.combinations(block, t):
                        cover_count[ts] -= 1
                    improved = True
                    break
        
        return blocks
    
    from collections import defaultdict
    
    # Strategy 1: Fast greedy with multiple seeds
    best_blocks = None
    best_count = float('inf')
    
    # Deterministic first
    blocks = fast_greedy(0, "deterministic")
    if len(blocks) < best_count:
        best_count = len(blocks)
        best_blocks = blocks
    
    # Random seeds
    num_random = 15
    for i in range(num_random):
        if time.time() > deadline - 15.0:
            break
        blocks = fast_greedy(i + 1, "random")
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Apply local search to best
    if best_blocks is not None:
        improved = local_search_aggressive(best_blocks[:], seed=42)
        if len(improved) < best_count:
            best_count = len(improved)
            best_blocks = improved
    
    # Try more local search passes with different seeds
    if best_blocks is not None:
        for ls_seed in range(5):
            if time.time() > deadline - 5.0:
                break
            improved = local_search_aggressive(best_blocks[:], seed=100 + ls_seed)
            if len(improved) < best_count:
                best_count = len(improved)
                best_blocks = improved
    
    # Final verification
    if best_blocks is None:
        return []
    
    covered = set()
    for block in best_blocks:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    
    if covered != set(all_t_subsets):
        # Fallback
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            start_ts = min(uncovered)
            block = list(start_ts)
            block_set = set(block)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for ts in point_to_tsubsets[x]:
                        if ts in uncovered:
                            rest = [y for y in ts if y != x]
                            if all(y in block_set for y in rest):
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
        return blocks
    
    return best_blocks
