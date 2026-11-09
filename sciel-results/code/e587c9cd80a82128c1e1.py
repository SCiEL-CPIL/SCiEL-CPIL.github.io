"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # We want to select a subset of [1, 100] with no subset summing to 200.
    # Greedy approach: iterate from 1 to 100.
    # Include x if it doesn't complete a sum of 200 with any existing subset.
    
    # reachable_sums stores all possible sums of subsets of the selected numbers so far.
    # We only care about sums <= 200.
    reachable_sums = {0}
    selected = []
    
    for x in range(1, 101):
        # If we add x, can we reach 200?
        # This happens if 200 - x is already in reachable_sums.
        if 200 - x in reachable_sums:
            continue
        
        # It is safe to add x.
        selected.append(x)
        
        # Update reachable_sums
        # We create a new set of sums by adding x to all current sums
        # We only keep sums <= 200
        new_sums = set()
        for s in reachable_sums:
            ns = s + x
            if ns <= 200:
                new_sums.add(ns)
        
        # Union with existing sums
        reachable_sums |= new_sums

    return selected
# EVOLVE-BLOCK-END
