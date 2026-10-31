import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # Precompute all t-subsets as bitmasks
    t_subsets = []
    for comb in itertools.combinations(range(v), t):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        t_subsets.append(mask)
    
    # Map mask to index for fast lookup? Or just use a set of masks.
    # A set of masks is fine for 3876 elements.
    uncovered = set(t_subsets)
    blocks = []
    
    # Helper to check if a mask is a subset of another
    # subset_mask & full_mask == subset_mask
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick the first uncovered subset
        # To be deterministic and simple, we pick min(uncovered)
        # But min on a set of ints is fast.
        first_uncovered = min(uncovered)
        
        # Initialize block with the elements of first_uncovered
        block = []
        temp_mask = first_uncovered
        for i in range(v):
            if temp_mask & (1 << i):
                block.append(i)
        
        # Greedily add points
        while len(block) < k:
            best_x = -1
            best_gain = -1
            
            # Precompute (t-1) subsets of current block?
            # Or just calculate gain on the fly.
            # Since |block| grows, we can just iterate.
            
            for x in range(v):
                if x in block:
                    continue
                
                # Calculate gain
                gain = 0
                # We need to count uncovered S such that S contains x and S \ {x} is a subset of block.
                # This is equivalent to: S is an uncovered t-subset containing x, 
                # and S \ {x} is a (t-1)-subset of block.
                
                # Optimization: 
                # If len(block) < t-1, gain is 0.
                if len(block) >= t - 1:
                    # Iterate over (t-1) subsets of block
                    for p_comb in itertools.combinations(block, t - 1):
                        # Form mask for p_comb
                        p_mask = 0
                        for i in p_comb:
                            p_mask |= (1 << i)
                        
                        # The candidate t-subset is p_comb U {x}
                        # Its mask is p_mask | (1 << x)
                        s_mask = p_mask | (1 << x)
                        
                        # Check if s_mask is in uncovered
                        if s_mask in uncovered:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            # If no gain, we still need to add a point to reach size k
            # If best_gain is 0, pick any x not in block.
            # The loop above will pick the first x with gain 0 if all are 0.
            # But we should ensure we pick *some* x.
            if best_x == -1:
                # This shouldn't happen if k <= v
                break
            block.append(best_x)
        
        # Sort block for consistency (not strictly required but good)
        block.sort()
        
        # Add block to list
        blocks.append(block)
        
        # Update uncovered set
        # Find all t-subsets contained in block
        for comb in itertools.combinations(block, t):
            mask = 0
            for i in comb:
                mask |= (1 << i)
            if mask in uncovered:
                uncovered.discard(mask)
                
    return blocks
