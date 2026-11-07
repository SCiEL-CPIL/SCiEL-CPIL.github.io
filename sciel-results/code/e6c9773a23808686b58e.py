"""Direct exhaustive search over all vertex subsets to maximize distinct induced degrees."""
import time

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    minv = P['minimum_vertices']
    edges = P['edges']
    start = time.time()
    deadline = start + 55.0   # safety margin well under the 60s limit

    # Build adjacency as bitmask for fast induced-degree computation
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Precompute index for each single-bit mask to avoid repeated bit_length calls
    bit_to_idx = {}
    for i in range(n):
        bit_to_idx[1 << i] = i

    best_count = -1
    best_mask = (1 << minv) - 1  # fallback: first minv vertices

    # Iterate over all masks with popcount >= minv.
    # 2^17 = 131072 subsets; each degree computation is O(k) bit ops.
    for mask in range(1 << n):
        c = mask.bit_count()
        if c < minv:
            continue
        # compute degrees of each set bit using precomputed index lookup
        degs = set()
        m = mask
        while m:
            lb = m & (-m)
            v = bit_to_idx[lb]
            degs.add((adj[v] & mask).bit_count())
            m ^= lb
        nd = len(degs)
        if nd > best_count:
            best_count = nd
            best_mask = mask
            # Early termination: max possible distinct degrees for k vertices is k
            # (degrees 0..k-1). If we hit that, we can't do better.
            if nd >= c:
                break
        # time guard
        if time.time() > deadline:
            break

    # Build vertex list from best_mask
    verts = []
    m = best_mask
    while m:
        lb = m & (-m)
        verts.append(int(bit_to_idx[lb]))
        m ^= lb

    # Ensure it's a list of >= minv distinct Python ints
    verts = list(dict.fromkeys(verts))
    return verts
# EVOLVE-BLOCK-END
