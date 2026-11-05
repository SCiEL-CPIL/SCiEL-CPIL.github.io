import itertools
import numpy as np
import time

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Initialize uncovered array
    uncovered = np.ones(n_subsets, dtype=bool)
    
    blocks = []
    
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    while uncovered.any():
        if time.time() - start_time > time_limit:
            break
            
        # Find the first uncovered subset
        first_idx = np.where(uncovered)[0][0]
        first_subset = all_t_subsets[first_idx]
        
        # Initialize block with this subset
        block = list(first_subset)
        
        # Mark this subset as covered
        uncovered[first_idx] = False
        
        # Greedy expansion
        while len(block) < k:
            best_gain = -1
            best_point = -1
            best_indices = [] # Indices of subsets covered by adding best_point
            
            # Candidates are points not in block
            # Optimization: Precompute complement? 
            # For v=19, list comprehension is fast enough
            candidates = [x for x in range(v) if x not in block]
            
            for x in candidates:
                gain = 0
                indices_to_cover = []
                
                # The new subsets covered are those containing x and (t-1) points from block
                for rest in itertools.combinations(block, t - 1):
                    # Create sorted tuple for lookup
                    # rest is sorted, x is int
                    # Efficient way to merge sorted list and single element
                    # Since t is small (5), simple sort is fine
                    s = tuple(sorted(rest + (x,)))
                    idx = subset_to_idx.get(s)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                        indices_to_cover.append(idx)
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    best_indices = indices_to_cover
            
            if best_point == -1:
                # No point added coverage? Should not happen if uncovered exists
                break
                
            block.append(best_point)
            
            # Update uncovered array
            if best_indices:
                uncovered[best_indices] = False
        
        blocks.append(sorted(block))
        
    return blocks
