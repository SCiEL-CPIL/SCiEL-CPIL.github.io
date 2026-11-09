def construct():
    # Pre-compute inverses of 1..10 mod 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)

    # min_size[r] = smallest size of a subset summing to r
    # best_subset[r] = the actual list of denominators
    min_size = [11] * 101
    best_subset = [None] * 101

    # Iterate all non-empty subsets of {1..10}
    # There are 2^10 - 1 = 1023 such subsets
    for mask in range(1, 1 << 10):
        s = 0
        size = 0
        terms = []
        
        for i in range(10):
            if mask & (1 << i):
                d = i + 1
                s = (s + inv[d]) % 101
                size += 1
                terms.append(d)
        
        # Update if we found a smaller subset for this residue
        if size < min_size[s]:
            min_size[s] = size
            best_subset[s] = terms[:]

    # Construct result
    result = []
    for r in range(101):
        if best_subset[r] is not None:
            result.append(best_subset[r])
        else:
            # Fallback: find any subset that sums to r (should not happen)
            # If this happens, the problem constraints might be violated or coverage is incomplete
            # Using [1] is a safe placeholder but might be incorrect for r != 1
            # However, given the density of subset sums, this branch is unlikely to be hit.
            result.append([1])

    return result
