"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def construct(v, k, t):
    """
    Construct a covering C(v, k, t) using a greedy set-cover approach
    enhanced with a local search / ILP polish if time permits.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Track which t-subsets are covered
    covered = np.zeros(n_t, dtype=bool)
    
    blocks = []
    
    # Helper function to update covered array for a block
    def update_covered(block):
        for ts in itertools.combinations(block, t):
            covered[t_to_idx[ts]] = True
    
    # Greedy phase: repeatedly pick the block that covers the most uncovered t-subsets
    # To speed this up, we use a heuristic: start with a random uncovered t-subset,
    # then greedily extend it to a k-block that maximizes coverage.
    
    while not covered.all():
        if time.time() > deadline - 10.0:
            break  # Leave time for finalization
        
        # Find an uncovered t-subset to seed a new block
        # Pick one at random to diversify, or the first one
        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break
            
        # Use a deterministic choice for reproducibility, or random
        # Let's use the first one for now, but we can improve
        seed_idx = uncovered_indices[0]
        seed = all_t_subsets[seed_idx]
        
        # Start the block with the seed
        block = list(seed)
        
        # Greedily add points to maximize coverage of remaining uncovered t-subsets
        while len(block) < k:
            best_x = -1
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                
                # Compute how many new t-subsets this x would cover
                # A t-subset is covered if it's a subset of the new block
                # We need to count uncovered t-subsets that are subsets of block + {x}
                # and contain x (since they weren't covered by block alone)
                
                # Efficiently: for each (t-1)-subset of current block,
                # check if (subset + (x,)) is uncovered
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in t_to_idx:
                        idx = t_to_idx[candidate]
                        if not covered[idx]:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1 or best_gain == 0:
                # No point improves coverage; pick an arbitrary unused point
                for x in range(v):
                    if x not in block:
                        best_x = x
                        break
                if best_x == -1:
                    break
            
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        update_covered(block)
        
        # Check if we've covered everything
        if covered.all():
            break
    
    # If we still have uncovered t-subsets, try to fix them with additional blocks
    if not covered.all():
        uncovered_indices = np.where(~covered)[0]
        for seed_idx in uncovered_indices:
            if time.time() > deadline - 5.0:
                break
            seed = all_t_subsets[seed_idx]
            block = list(seed)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in t_to_idx:
                            idx = t_to_idx[candidate]
                            if not covered[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    for x in range(v):
                        if x not in block:
                            best_x = x
                            break
                    if best_x == -1:
                        break
                block.append(best_x)
            block.sort()
            # Avoid duplicate blocks
            if block not in blocks:
                blocks.append(block)
                update_covered(block)
    
    # Verify coverage
    if not covered.all():
        # Force coverage with brute force if needed (should not happen)
        for ts in all_t_subsets:
            if not covered[t_to_idx[ts]]:
                block = list(ts)
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block:
                        block.append(x)
                block.sort()
                if block not in blocks:
                    blocks.append(block)
                    update_covered(block)
    
    return blocks
# EVOLVE-BLOCK-END
