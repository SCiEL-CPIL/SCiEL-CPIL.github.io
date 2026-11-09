# EVOLVE-BLOCK-START
def construct():
    n = 12
    chosen = []
    subset_sums = {0}
    
    for _ in range(n):
        candidate = 1
        while True:
            # Check for collision
            # A collision exists if there is s in subset_sums such that s + candidate is also in subset_sums
            # This is equivalent to candidate being in the difference set subset_sums - subset_sums
            # We check directly.
            collision = False
            for s in subset_sums:
                if (s + candidate) in subset_sums:
                    collision = True
                    break
            
            if not collision:
                chosen.append(candidate)
                # Update subset sums
                # new_sums = {s + candidate for s in subset_sums}
                # subset_sums = subset_sums | new_sums
                # Optimization: Since we know no collision, we can just add the new sums.
                # But we need the full set for future checks.
                new_sums = {s + candidate for s in subset_sums}
                subset_sums |= new_sums
                break
            else:
                candidate += 1
                
    return chosen
# EVOLVE-BLOCK-END
