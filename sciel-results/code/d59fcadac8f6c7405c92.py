import time

def construct():
    start_time = time.time()
    LIMIT = 85.0  # 90s wall-clock, safety margin

    p = 101
    denominators = list(range(1, 11))
    n = 10

    # Precompute modular inverses
    inverses = [pow(x, -1, p) for x in denominators]

    # best[r] = (min_size, mask) for residue r
    # We will search for subsets of size 1, then 2, then 3.
    # If a residue is covered by a subset of size 3, we stop searching for larger sizes.
    # This guarantees max row length <= 3 if coverage is complete.
    
    best = [None] * p

    # Helper to update best if current size is smaller
    def update_best(r, sz, mask):
        if best[r] is None or sz < best[r][0]:
            best[r] = (sz, mask)

    # Iterate over all subsets of size 1, 2, 3
    # We can just iterate all masks and filter by bit count, or use combinations.
    # With n=10, total masks is 1024. Iterating all is fast enough.
    
    for mask in range(1, 1 << n):
        sz = bin(mask).count('1')
        if sz > 3:
            continue
        
        s = 0
        # Calculate sum of inverses for this mask
        for i in range(n):
            if mask & (1 << i):
                s += inverses[i]
        
        r = s % p
        update_best(r, sz, mask)

    # Check if all residues are covered
    uncovered = [r for r in range(p) if best[r] is None]
    
    # If some residues are not covered by size <= 3, we must expand to size 4, etc.
    # However, the previous program found a solution with max size 4.
    # Let's assume size 3 is sufficient. If not, we fallback to size 4.
    if uncovered:
        # Expand search to size 4
        for mask in range(1, 1 << n):
            sz = bin(mask).count('1')
            if sz != 4:
                continue
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += inverses[i]
            r = s % p
            if best[r] is None:
                best[r] = (sz, mask)
        
        # Re-check
        uncovered = [r for r in range(p) if best[r] is None]
        if uncovered:
            # If still uncovered, expand to all sizes (fallback to original logic)
            for mask in range(1, 1 << n):
                sz = bin(mask).count('1')
                if sz > 4:
                    continue
                s = 0
                for i in range(n):
                    if mask & (1 << i):
                        s += inverses[i]
                r = s % p
                if best[r] is None or sz < best[r][0]:
                    best[r] = (sz, mask)

    # Final verification
    for r in range(p):
        if best[r] is None:
            raise RuntimeError(f"Residue {r} not covered")

    # Build result
    result = []
    for r in range(p):
        sz, mask = best[r]
        combo = [denominators[i] for i in range(n) if mask & (1 << i)]
        result.append(combo)

    # Verify correctness
    for r in range(p):
        s = sum(inverses[d - 1] for d in result[r])
        if s % p != r:
            raise AssertionError(f"Residue {r}: got {s % p}")

    return result
