def construct():
    from math import lcm
    best_sum = 0
    best_set = []
    
    # Precompute LCMs or just use math.lcm
    # We will try each x in [2, 100] as the "anchor"
    # Then greedily add the best compatible numbers
    
    candidates = list(range(2, 257))
    
    for anchor in candidates:
        current_set = [anchor]
        # Sort remaining candidates by 1/x descending
        # To be efficient, we can just iterate through all and check
        # But checking compatibility with all in current_set is O(N)
        # N is at most 255.
        
        # Greedy fill
        # We want to maximize sum.
        # Sort candidates by weight descending
        # Exclude anchor
        others = [x for x in candidates if x != anchor]
        others.sort(key=lambda x: 1/x, reverse=True)
        
        for x in others:
            compatible = True
            for y in current_set:
                if lcm(x, y) <= 256:
                    compatible = False
                    break
            if compatible:
                current_set.append(x)
                
        # Calculate sum
        current_sum = sum(1/x for x in current_set)
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = current_set
            
    return best_set
