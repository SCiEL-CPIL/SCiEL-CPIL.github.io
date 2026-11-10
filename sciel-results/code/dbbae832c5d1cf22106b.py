"""Rewritten program: exact minimum cochromatic partition via DP + branch-and-bound."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    n = P['n']
    edge_set = set()
    for e in P['edges']:
        edge_set.add((e[0], e[1]))
        edge_set.add((e[1], e[0]))

    # Precompute adjacency as bitmasks for speed
    adj = [0] * n
    for u, v in edge_set:
        if u < v:  # avoid double processing
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    # Helper: check if a subset (as bitmask) is a clique
    def is_clique(mask):
        if mask == 0:
            return True
        # Pick lowest set bit
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        # All other vertices in mask must be adjacent to v
        rest = mask ^ lsb
        while rest:
            lsb2 = rest & (-rest)
            v2 = lsb2.bit_length() - 1
            if not (adj[v] & lsb2):
                return False
            rest ^= lsb2
        return True

    # Helper: check if a subset (as bitmask) is an independent set
    def is_independent(mask):
        if mask == 0:
            return True
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        while rest:
            lsb2 = rest & (-rest)
            v2 = lsb2.bit_length() - 1
            if (adj[v] & lsb2):
                return False
            rest ^= lsb2
        return True

    # Precompute valid masks (clique or independent)
    full_mask = (1 << n) - 1
    valid_masks = []
    valid_mask_set = set()
    for mask in range(1, 1 << n):
        if is_clique(mask) or is_independent(mask):
            valid_masks.append(mask)
            valid_mask_set.add(mask)

    # Sort valid masks by popcount descending to try larger parts first
    valid_masks.sort(key=lambda m: m.bit_count(), reverse=True)

    # DP: min number of valid parts to cover exactly 'mask'
    # dp[mask] = min parts to cover mask
    INF = 10**9
    dp = [INF] * (1 << n)
    dp[0] = 0

    for mask in range(1, 1 << n):
        # Try all valid submasks of mask
        sub = mask
        while sub:
            if sub in valid_mask_set:
                if dp[mask ^ sub] + 1 < dp[mask]:
                    dp[mask] = dp[mask ^ sub] + 1
            sub = (sub - 1) & mask

    # dp[full_mask] is the minimum number of parts
    min_parts = dp[full_mask]

    # Now reconstruct the partition
    # We need to find which valid masks were used
    # Backtrack from full_mask
    parts = []
    remaining = full_mask
    while remaining:
        # Find a valid submask s of remaining such that dp[remaining ^ s] + 1 == dp[remaining]
        sub = remaining
        found = False
        while sub:
            if sub in valid_mask_set:
                if dp[remaining ^ sub] + 1 == dp[remaining]:
                    parts.append(sub)
                    remaining ^= sub
                    found = True
                    break
            sub = (sub - 1) & remaining
        if not found:
            # Should not happen, but fallback
            parts.append(remaining)
            break

    # Convert bitmask parts to lists of vertex indices
    result = []
    for mask in parts:
        part = []
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            part.append(v)
            m ^= lsb
        part.sort()
        result.append(part)

    # Ensure types are correct
    final_result = [list(p) for p in result]

    # Check time
    if time.time() > deadline:
        # Return best we have
        pass

    return final_result

# EVOLVE-BLOCK-END
