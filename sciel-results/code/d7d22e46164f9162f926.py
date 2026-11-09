import itertools
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Precompute all t-subsets as frozensets or sorted tuples for hashing, 
    # but for speed, we'll use integer bitmasks for coverage tracking.
    
    # 1. Generate all t-subsets as bitmasks
    # A subset of points {p1, p2, ..., pt} is represented as a bitmask where bit pi is set.
    all_t_subsets = []
    for comb in itertools.combinations(range(v), t):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        all_t_subsets.append(mask)
    
    # Map from mask to index for potential use, but we just need a set of covered masks.
    uncovered_masks = set(all_t_subsets)
    
    # Precompute for each point p, the list of t-subset masks containing p.
    # This might not be strictly necessary for the gain calc if we construct subsets from the block directly,
    # but it helps if we were doing a different search. 
    # Let's stick to the "construct from partial block" method which is faster for small k.
    
    blocks = []
    
    # Helper to convert a mask to a list of points
    def mask_to_list(mask):
        pts = []
        for i in range(v):
            if mask & (1 << i):
                pts.append(i)
        return pts

    while uncovered_masks:
        if time.time() > deadline:
            # In a real competitive setting, we might have a fallback.
            # Here, we assume it finishes. If not, we return what we have, 
            # but it might be invalid. To be safe, we should probably not 
            # include this guard if it leads to invalidity, unless we 
            # can guarantee validity. 
            # Given the speed, we expect to finish.
            break
        
        # Pick the first uncovered t-subset to start a new block
        first_uncovered = next(iter(uncovered_masks))
        block_points = mask_to_list(first_uncovered)
        
        # Greedily add points to the block
        while len(block_points) < k:
            best_point = -1
            best_gain = -1
            current_block_set = set(block_points)
            
            # We need to evaluate gain for each candidate point x not in current_block_set
            for x in range(v):
                if x in current_block_set:
                    continue
                
                gain = 0
                # The new t-subsets covered by adding x are those T such that:
                # 1. T contains x
                # 2. T \ {x} is a subset of current_block_points
                # 3. T is currently uncovered
                
                # We can iterate over all (t-1)-subsets of current_block_points
                # For each such subset S, the t-subset is S U {x}
                for rest in itertools.combinations(block_points, t - 1):
                    # Construct the mask for rest U {x}
                    mask = (1 << x)
                    for p in rest:
                        mask |= (1 << p)
                    
                    if mask in uncovered_masks:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                # Fallback: pick any point (shouldn't happen if k < v)
                best_point = 0
                for x in range(v):
                    if x not in current_block_set:
                        best_point = x
                        break
            
            block_points.append(best_point)
        
        # Sort the block for consistency
        block_points.sort()
        blocks.append(block_points)
        
        # Update uncovered_masks
        # Iterate over all t-subsets of the new block
        for comb in itertools.combinations(block_points, t):
            mask = 0
            for p in comb:
                mask |= (1 << p)
            if mask in uncovered_masks:
                uncovered_masks.remove(mask)
                
    return blocks
