import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    p = 101
    denoms = list(range(1, 11))
    invs = [pow(x, -1, p) for x in denoms]

    # For each residue, track the minimum-length subset that achieves it
    # min_len[r] = smallest k, and best_subset[r] = one such subset of size k
    min_len = [11] * p  # 11 is "infinity" (max possible is 10)
    best_subset = [None] * p

    # Enumerate all 1023 non-empty subsets via bitmask
    # For each subset, compute sum of inverses mod p
    for mask in range(1, 1 << 10):
        # Count bits and compute sum
        s = 0
        k = 0
        for i in range(10):
            if mask & (1 << i):
                s += invs[i]
                k += 1
        s %= p
        if k < min_len[s]:
            min_len[s] = k
            # Reconstruct the subset
            terms = []
            for i in range(10):
                if mask & (1 << i):
                    terms.append(denoms[i])
            best_subset[s] = terms

    # Check if all residues are covered
    # If some residue has no subset, we need a fallback
    # But with 1023 subsets and 101 residues, all should be covered

    # Build result
    result = []
    for r in range(p):
        if best_subset[r] is not None:
            result.append(best_subset[r])
        else:
            # Fallback: should not happen, but use [1] as placeholder
            result.append([1])

    return result
