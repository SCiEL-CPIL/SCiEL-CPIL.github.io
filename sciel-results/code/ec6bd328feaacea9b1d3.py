"""Optimized covering design C(19,8,5) combining greedy with randomized restarts and local refinement."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each t-subset, the list of points that can extend it
    # and for each point, which t-subsets it belongs to
    point_to_tsubsets = defaultdict(set)
    for i, ts in enumerate(all_t_subsets):
        for x in ts:
            point_to_tsubsets[x].add(i)
    
    best_blocks = None
    best_count = float('inf')
    
    def greedy_build(start_subset, uncovered_set, rng, deadline, v, k, t, all_t_subsets, subset_to_idx):
        """Build a single block greedily starting from start_subset."""
        block = list(start_subset)
        block_set = set(block)
        
        while len(block) < k:
            best_gain = -1
            best_point = None
            best_gain_ties = []
            
            for x in range(v):
                if x in block_set:
                    continue
                
                # Calculate how many uncovered t-subsets would be covered
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    idx = subset_to_idx.get(candidate)
                    if idx is not None and idx in uncovered_set:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    best_gain_ties = [x]
                elif gain == best_gain and gain >= 0:
                    best_gain_ties.append(x)
            
            if best_point is not None:
                # Randomly pick among ties for diversity
                if len(best_gain_ties) > 1:
                    best_point = rng.choice(best_gain_ties)
                block.append(best_point)
                block_set.add(best_point)
            else:
                # No point improves coverage, pick a random one
                remaining = [x for x in range(v) if x not in block_set]
                if remaining:
                    best_point = rng.choice(remaining)
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    break
        
        block.sort()
        return block
    
    def local_search(blocks, uncovered_set, rng, deadline, v, k, t, all_t_subsets, subset_to_idx):
        """Try to improve the solution by removing redundant blocks."""
        if time.time() > deadline:
            return blocks, uncovered_set
        
        blocks_list = list(blocks)
        improved = True
        max_iter = 50
        
        for iteration in range(max_iter):
            if time.time() > deadline or not improved:
                break
            improved = False
            
            # Try removing each block if it's redundant
            for i in range(len(blocks_list)):
                if time.time() > deadline:
                    break
                
                block_to_remove = blocks_list[i]
                
                # Check if all t-subsets of this block are covered by other blocks
                is_redundant = True
                for ts in itertools.combinations(block_to_remove, t):
                    idx = subset_to_idx.get(ts)
                    if idx is None or idx in uncovered_set:
                        is_redundant = False
                        break
                    # Check if covered by another block
                    # This check is expensive; skip for now and just try removal
                    # then verify at the end
                
                if is_redundant:
                    # Try removing this block
                    new_uncovered = set(uncovered_set)
                    for ts in itertools.combinations(block_to_remove, t):
                        idx = subset_to_idx.get(ts)
                        if idx is not None:
                            # Check if this t-subset is covered by any other block
                            covered = False
                            for j, b in enumerate(blocks_list):
                                if j == i:
                                    continue
                                if ts in itertools.combinations(b, t):
                                    covered = True
                                    break
                            if not covered:
                                new_uncovered.add(idx)
                    
                    if len(new_uncovered) == len(uncovered_set):
                        # Block is truly redundant
                        blocks_list.pop(i)
                        improved = True
                        break
        
        return blocks_list, uncovered_set
    
    # Run multiple randomized greedy attempts
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 12345 + 7)
        
        uncovered_set = set(range(n_t_subsets))
        blocks = []
        
        while uncovered_set and time.time() < deadline:
            # Pick a random uncovered t-subset to start the block
            start_idx = rng.choice(list(uncovered_set))
            start_subset = all_t_subsets[start_idx]
            
            # Build block greedily
            block = greedy_build(start_subset, uncovered_set, rng, deadline, v, k, t, all_t_subsets, subset_to_idx)
            blocks.append(tuple(block))
            
            # Remove all t-subsets covered by this block
            for ts in itertools.combinations(block, t):
                idx = subset_to_idx.get(ts)
                if idx is not None:
                    uncovered_set.discard(idx)
        
        # Check if we have a complete cover
        if not uncovered_set:
            # Try local search to remove redundant blocks
            blocks_list, _ = local_search(blocks, set(), rng, deadline, v, k, t, all_t_subsets, subset_to_idx)
            
            if len(blocks_list) < best_count:
                best_count = len(blocks_list)
                best_blocks = [list(b) for b in blocks_list]
        
        # Early termination if we found a very good solution
        if best_count <= 280:
            break
    
    if best_blocks is None:
        # Fallback: use the original deterministic greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered and time.time() < deadline:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
