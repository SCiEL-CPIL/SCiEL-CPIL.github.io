import itertools
import random
import time

def construct(v, k, t):
    # Precompute all k-subsets and their t-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    k_subset_to_t_subsets = {}
    for ks in all_k_subsets:
        ts = list(itertools.combinations(ks, t))
        k_subset_to_t_subsets[ks] = ts

    # Initialize uncovered set
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    
    blocks = []
    
    # Greedy Phase
    while uncovered:
        best_block = None
        max_covered = 0
        
        # To speed this up, we can filter candidates, but for v=19, 50k candidates is okay if optimized.
        # However, iterating 50k * 14 (4-subsets per 7-subset) per step might be slow if many steps.
        # Number of steps is ~150. 150 * 50k * 14 = 105 million operations. Might be too slow in Python.
        
        # Optimization: Only consider blocks that cover at least one uncovered subset.
        # We can maintain a list of candidate blocks.
        
        # Let's try a faster heuristic: 
        # Pick a random uncovered subset, generate all k-subsets containing it, 
        # and pick the one that covers the most other uncovered subsets.
        
        seed_subset = next(iter(uncovered))
        
        # Generate all k-subsets containing seed_subset
        remaining_points = [p for p in range(v) if p not in seed_subset]
        need_points = k - t
        
        candidates = []
        for combo in itertools.combinations(remaining_points, need_points):
            block = tuple(sorted(seed_subset + combo))
            # Count how many uncovered subsets this block covers
            # We can do this by checking its t-subsets
            count = 0
            for ts in k_subset_to_t_subsets[block]:
                if ts in uncovered:
                    count += 1
            if count > 0:
                candidates.append((count, block))
                
        if not candidates:
            # Should not happen if uncovered is not empty
            break
            
        # Pick the best candidate
        candidates.sort(reverse=True)
        best_count, best_block = candidates[0]
        
        blocks.append(list(best_block))
        
        # Update uncovered
        for ts in k_subset_to_t_subsets[best_block]:
            uncovered.discard(ts)
            
    # Local Search Phase: Remove redundant blocks
    # We need to track coverage counts
    # Rebuild coverage counts from the current blocks
    coverage_count = {}
    for b in blocks:
        for ts in k_subset_to_t_subsets[tuple(b)]:
            coverage_count[ts] = coverage_count.get(ts, 0) + 1
            
    improved = True
    while improved:
        improved = False
        # Try to remove each block if it is redundant
        # To avoid modifying list while iterating, iterate backwards or make a copy
        for i in range(len(blocks) - 1, -1, -1):
            b = tuple(blocks[i])
            redundant = True
            for ts in k_subset_to_t_subsets[b]:
                if coverage_count[ts] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block i
                for ts in k_subset_to_t_subsets[b]:
                    coverage_count[ts] -= 1
                
                blocks.pop(i)
                improved = True
                # Restart the loop to check for new redundancies caused by this removal?
                # Actually, removing a block only decreases counts, so it cannot make a previously non-redundant block redundant.
                # It can only make the *remaining* blocks more critical.
                # So we can continue the loop.
                
    return blocks
