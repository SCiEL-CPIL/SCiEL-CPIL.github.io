"""Cochromatic partition of a pinned random graph via subset DP."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    n = P['n']
    edges = P['edges']

    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    full_mask = (1 << n) - 1
    num_subsets = 1 << n

    # Precompute for each subset S: is it a clique? is it an independent set?
    # We use the following trick:
    # A set S is a clique iff for every vertex v in S, all other vertices in S are in adj[v].
    # Equivalently, S is a clique iff for the "least significant" vertex v in S, 
    # S is a subset of {v} union adj[v]. But we need this to hold for ALL v.
    # A simpler characterization: S is a clique iff S & ~adj[v] & ~(1<<v) == 0 for all v in S.
    #
    # For independent set: S is independent iff S & adj[v] == 0 for all v in S.
    #
    # Efficient precomputation approach:
    # For each subset S, we can check by iterating vertices in S.
    # But we can do even better with a DP-style check.

    # Let's precompute is_clique[S] and is_ind[S] for all S.
    is_clique = [False] * num_subsets
    is_ind = [False] * num_subsets

    # Single vertex sets: both clique and independent
    for v in range(n):
        m = 1 << v
        is_clique[m] = True
        is_ind[m] = True

    # For larger subsets:
    # S is a clique iff:
    #   Let v = lowest set bit in S. Then (S \ {v}) must be a clique and 
    #   (S \ {v}) must be a subset of adj[v].
    # S is an independent set iff:
    #   Let v = lowest set bit in S. Then (S \ {v}) must be an independent set and
    #   (S \ {v}) must be a subset of ~adj[v] (i.e., no edge between v and any other in S).

    for S in range(1, num_subsets):
        # Only process if not already set (single vertices)
        if S & (S - 1) == 0:
            continue  # single vertex, already set

        # Get lowest set bit
        lb = S & (-S)
        v = lb.bit_length() - 1
        rest = S ^ lb

        # Check clique: rest must be clique and rest subset of adj[v]
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[S] = True

        # Check independent: rest must be independent and rest subset of complement of adj[v]
        # rest & adj[v] must be 0
        if is_ind[rest] and (rest & adj[v]) == 0:
            is_ind[S] = True

    # is_valid[S] = is_clique[S] or is_ind[S]
    is_valid = [False] * num_subsets
    for S in range(1, num_subsets):
        if is_clique[S] or is_ind[S]:
            is_valid[S] = True

    # DP: dp[S] = minimum number of parts to partition vertex set S
    # Each part must be a valid subset (clique or independent set)
    INF = 10 ** 9
    dp = [INF] * num_subsets
    parent_part = [0] * num_subsets
    dp[0] = 0

    # Iterate subsets in increasing order of popcount (or just increasing mask value)
    # We need dp[rest] to be computed before dp[S] where rest = S \ T
    # Iterating by mask value works if we iterate submasks properly.
    # Actually, we iterate S from 1 to full_mask, and for each S we iterate submasks T of S.
    # dp[S \ T] must already be computed. Since S \ T < S in integer value? Not necessarily.
    # But if we iterate S in increasing order of popcount, or use the fact that S\T is a proper subset...
    # Actually, the standard approach: iterate S from 1 to full_mask. For each S, 
    # iterate all submasks T of S. Since T is a subset of S, S\T is also a subset of S.
    # But S\T might be > S in integer value? No, S\T < S always? 
    # Wait, S\T = S & ~T. Since T is a non-empty subset of S, S\T has fewer bits, 
    # but its integer value could be larger than S? No, removing bits can only decrease or keep value.
    # Actually, S\T is always < S when T is non-empty and T subset of S? 
    # Not necessarily: e.g., S = 1010 (10), T = 0010 (2), S\T = 1000 (8) < 10. Yes.
    # S = 1100 (12), T = 0100 (4), S\T = 1000 (8) < 12. Yes.
    # In general, if T is a non-empty subset of S, then S\T < S.
    # Proof: Let k be the highest bit where S\T and S differ. S has 1 there, S\T has 0.
    # So S\T < S. Correct.
    # So iterating S in increasing order works.

    for S in range(1, num_subsets):
        # Try all valid submasks T of S
        # Optimization: we can restrict T to include the lowest set bit of S to avoid symmetric duplicates
        # But for n=14, total submask iterations across all S is 3^n ~ 4.78M, which is fine.
        
        best = INF
        best_part = 0
        
        # Iterate submasks of S
        sub = S
        while sub:
            if is_valid[sub]:
                rest = S ^ sub
                val = dp[rest] + 1
                if val < best:
                    best = val
                    best_part = sub
            sub = (sub - 1) & S
            
            # Deadline check
            if time.time() - start_time > 55:
                # Return best found so far (partial result is invalid per rules, but we need to return something)
                # Actually, we must return a valid partition. If we're here, we have partial DP.
                # Let's just break and hope we have enough.
                # For safety, if we haven't completed, we can fall back to a greedy partition.
                # But n=14 should finish well within time.
                break

        dp[S] = best
        parent_part[S] = best_part

    # Reconstruct partition
    parts = []
    S = full_mask
    while S:
        part_mask = parent_part[S]
        if part_mask == 0:
            # Fallback: take single vertex
            part_mask = S & (-S)
        part_verts = [int(i) for i in range(n) if (part_mask >> i) & 1]
        parts.append(part_verts)
        S = S ^ part_mask

    # Ensure types
    final_parts = []
    for part in parts:
        final_parts.append([int(v) for v in part])

    return final_parts

# EVOLVE-BLOCK-END
