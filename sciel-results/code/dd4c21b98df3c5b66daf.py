import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    timeout = 140.0 # Safety margin
    
    # Generate all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_subsets)
    
    if not uncovered:
        return []
        
    blocks = []
    
    # Precompute all possible k-subsets? No, too many (50k). 
    # Instead, we use the seed-based local search.
    
    while uncovered:
        if time.time() - start_time > timeout:
            break
            
        # Pick the first uncovered subset as a seed
        # Using min() ensures deterministic behavior and picks "early" subsets
        seed = min(uncovered)
        
        # Points not in the seed
        seed_set = set(seed)
        remaining_points = [x for x in range(v) if x not in seed_set]
        
        # We need to add (k - t) points to the seed to form a block of size k
        num_to_add = k - t
        
        best_gain = -1
        best_block = None
        
        # Iterate through all combinations of points to add
        # For v=19, k=7, t=4, num_to_add=3. C(15,3)=455. Very fast.
        for add_combo in itertools.combinations(remaining_points, num_to_add):
            # Form the candidate block
            # Note: seed is a tuple, add_combo is a tuple
            candidate_block = tuple(sorted(seed + add_combo))
            
            # Calculate gain: how many uncovered t-subsets does this block cover?
            # A block of size k contains C(k, t) t-subsets.
            gain = 0
            for sub in itertools.combinations(candidate_block, t):
                if sub in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = candidate_block
            elif gain == best_gain and best_block is not None:
                # Tie-breaking: prefer lexicographically smaller block? 
                # Or just keep the first one found.
                if candidate_block < best_block:
                    best_block = candidate_block
            
        if best_block is None:
            # Should not happen if k >= t and v >= k
            break
            
        blocks.append(list(best_block))
        
        # Update uncovered set
        for sub in itertools.combinations(best_block, t):
            uncovered.discard(sub)
            
    return blocks
