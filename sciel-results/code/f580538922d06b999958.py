# EVOLVE-BLOCK-START
def construct():
    n = 12
    result = []
    # forbidden contains all differences s_i - s_j for current subset sums
    # initially {0}
    forbidden = {0}
    
    for _ in range(n):
        x = 1
        while x in forbidden:
            x += 1
        
        result.append(x)
        
        # Update forbidden set
        # New differences are d, d+x, d-x for all d in old forbidden
        # We iterate over a copy because we are modifying the set
        old_forbidden = list(forbidden)
        for d in old_forbidden:
            forbidden.add(d + x)
            forbidden.add(d - x)
            
    return result
# EVOLVE-BLOCK-END
