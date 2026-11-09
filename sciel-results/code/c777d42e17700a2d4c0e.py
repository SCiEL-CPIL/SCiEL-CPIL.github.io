"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    MOD = 1009
    N = 1008
    
    # Compute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Identify valid range [2, 1007]
    # Pairs (a, b) with a < b, both in [2, 1007]
    pairs = []
    seen = set()
    for x in range(2, 1008):
        y = inv[x]
        if y == x:
            continue
        if y < 2 or y > 1007:
            continue
        a, b = min(x, y), max(x, y)
        if (a, b) not in seen:
            seen.add((a, b))
            pairs.append((a, b))
    
    # Sort pairs by a
    pairs.sort()
    
    # Precompute min_b_for_a
    min_b_for_a = [1009] * (MOD + 1)
    for a, b in pairs:
        if b < min_b_for_a[a]:
            min_b_for_a[a] = b
    
    # Suffix minimum
    suffix_min = [1009] * (MOD + 1)
    suffix_min[MOD] = 1009
    for i in range(MOD - 1, 1, -1):
        suffix_min[i] = min(min_b_for_a[i], suffix_min[i + 1])
    
    best_start = 2
    best_len = 0
    best_end = 2
    
    for s in range(2, 1008):
        max_e = suffix_min[s] - 1
        if max_e > 1007:
            max_e = 1007
        if max_e < s:
            continue
        length = max_e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
            best_end = max_e
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
