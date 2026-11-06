import time

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin before 90s kill

    P = 101
    D = list(range(1, 11))

    # Precompute modular inverses
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, P)

    # Enumerate all non-empty subsets of {1..10}, group by size
    # subsets_by_size[k] = list of (residue, terms)
    subsets_by_size = [[] for _ in range(11)]

    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                terms.append(x)
                s = (s + inv[x]) % P
        subsets_by_size[len(terms)].append((s, terms))

    # For each residue, find the smallest-size subset that maps to it.
    # Since subsets may repeat residues at the same size, we just take the
    # first one found at the minimum size. This gives the minimum possible
    # row length per residue, hence minimizes the maximum row length.
    best = [None] * P
    best_size = [11] * P  # sentinel larger than max possible

    for k in range(1, 11):
        for residue, terms in subsets_by_size[k]:
            if best[residue] is None:
                best[residue] = terms
                best_size[residue] = k

    # Build result
    result = []
    for r in range(P):
        if best[r] is not None:
            result.append(best[r])
        else:
            # Should not happen (L=10 covers all), but handle gracefully
            result.append([1])

    return result
