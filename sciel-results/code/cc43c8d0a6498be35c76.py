"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from itertools import combinations
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] + [pow(x, -1, 101) for x in range(1, 11)]
    
    # For each residue, find the shortest subset of {1..10}
    # whose inverse-sum mod 101 equals that residue.
    # Iterate by subset size so the first hit is the shortest.
    best = {}
    
    for k in range(1, 11):
        for combo in combinations(range(1, 11), k):
            s = sum(inv[x] for x in combo) % 101
            if s not in best:
                best[s] = combo
    
    # Build result: row r gets the shortest subset summing to r mod 101
    result = []
    for r in range(101):
        if r in best:
            result.append(list(best[r]))
        else:
            # Fallback (should not occur for p=101, bound=10)
            result.append([1])
    
    return result
# EVOLVE-BLOCK-END
