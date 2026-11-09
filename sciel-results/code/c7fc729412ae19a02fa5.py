"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 12
    subset_sums = {0}
    chosen = []
    candidate = 1
    
    while len(chosen) < n:
        collision = False
        for s in subset_sums:
            if (s + candidate) in subset_sums:
                collision = True
                break
        
        if not collision:
            chosen.append(candidate)
            # Use set union with a generator for clarity and potentially better performance
            subset_sums = subset_sums | {s + candidate for s in subset_sums}
            candidate += 1
        else:
            candidate += 1
    
    return chosen
# EVOLVE-BLOCK-END
