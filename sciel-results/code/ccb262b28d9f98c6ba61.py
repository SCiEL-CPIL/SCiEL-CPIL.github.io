import itertools

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a lazy greedy set cover algorithm.
    """
    # Initialize the set of uncovered t-subsets
    uncovered = set(itertools.combinations(range(v), t))
    
    blocks = []
    
    # Precompute the list of all points for faster filtering if needed, 
    # but v is small (19), so list comprehension is fine.
    
    while uncovered:
        # Pick an arbitrary uncovered t-subset to anchor our search
        anchor_t = next(iter(uncovered))
        anchor_set = set(anchor_t)
        
        # Points not in the anchor
        available_points = [p for p in range(v) if p not in anchor_set]
        
        best_block = None
        best_score = -1
        
        # Generate all k-subsets that contain anchor_t
        for extra_points in itertools.combinations(available_points, k - t):
            # Combine anchor and extra points
            # Using a tuple for the block to store in result later, 
            # but we need to iterate its t-subsets.
            candidate_block = tuple(sorted(anchor_t + extra_points))
            
            # Calculate the score: how many uncovered t-subsets does this block cover?
            score = 0
            for t_sub in itertools.combinations(candidate_block, t):
                if t_sub in uncovered:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_block = candidate_block
                
        if best_block is None:
            break
            
        blocks.append(list(best_block))
        
        # Update the uncovered set
        for t_sub in itertools.combinations(best_block, t):
            if t_sub in uncovered:
                uncovered.remove(t_sub)
                
    return blocks
