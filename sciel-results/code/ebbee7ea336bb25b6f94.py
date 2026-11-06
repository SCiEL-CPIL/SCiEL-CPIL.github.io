"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import random
    
    def is_valid(S):
        """Check if no subset of S sums to 200 using DP."""
        reachable = [False] * 201
        reachable[0] = True
        for x in S:
            if x > 200:
                continue
            for s in range(200 - x, -1, -1):
                if reachable[s]:
                    reachable[s + x] = True
        return not reachable[200]

    # Known good solution: {66, ..., 100} has size 35
    base_set = set(range(66, 101))
    
    if is_valid(base_set):
        best_set = base_set
        best_size = 35
    else:
        best_set = set()
        best_size = 0

    # Try to find a larger set (size 36) via random search
    # We can also try to improve the base set by swapping
    rng = random.Random(42)
    
    # Search for size 36
    found_36 = False
    for _ in range(10000):
        cand = set(rng.sample(range(1, 101), 36))
        if is_valid(cand):
            best_set = cand
            best_size = 36
            found_36 = True
            break

    # If 36 not found, try to improve 35 by swapping one element out and two in?
    # Or just stick with 35.
    # Let's try a few more random 35s just in case base was bad (it shouldn't be)
    if not found_36 and best_size < 35:
        for _ in range(1000):
            cand = set(rng.sample(range(1, 101), 35))
            if is_valid(cand):
                best_set = cand
                best_size = 35
                break

    return list(best_set)
# EVOLVE-BLOCK-END
