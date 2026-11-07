"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_size = P['minimum_vertices']
    deadline = time.monotonic() + 80.0

    # Build adjacency as bitmasks for speed
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    best_distinct = 0
    best_mask = (1 << min_size) - 1  # initial: first min_size vertices

    # Order vertices by full-graph degree (descending) for better B&B pruning
    vertex_order = sorted(range(n), key=lambda v: adj[v].bit_count(), reverse=True)
    # Map: position in order -> vertex id
    # We'll build masks in this order

    # Precompute: for each vertex, its adjacency bitmask
    # For B&B: process vertices in vertex_order, at each step decide include/exclude

    def branch_and_bound():
        nonlocal best_distinct, best_mask
        # Recursive: at step i, we've decided on first i vertices in vertex_order
        # state: included_mask, excluded_count, current_degree_set
        # We use iterative deepening or just recursion with pruning

        # Use a stack-based approach for speed
        # State: (vertex_idx, included_mask, degree_set_as_frozenset, included_count)
        # But frozensets are expensive. Instead, we'll compute degrees at the end
        # for leaf nodes, and use bounds for pruning.

        # Better: recursive with memoization-like pruning
        # At step i, remaining = n - i
        # Max possible distinct degrees = len(current_degs) + remaining (each new vertex adds at most 1)
        # Also need included_count + remaining >= min_size (enough vertices left to meet minimum)

        # Let's do a clean recursive B&B
        # Process vertices in vertex_order
        # At each vertex: include or exclude
        # Prune if:
        #   1. included_count + remaining < min_size (can't meet minimum)
        #   2. len(degs) + remaining <= best_distinct (can't beat best)
        #   3. Time exceeded

        import sys
        sys.setrecursionlimit(10000)

        def rec(idx, mask, degs, count):
            nonlocal best_distinct, best_mask

            if time.monotonic() > deadline:
                return

            remaining = n - idx

            # Prune: not enough vertices to meet minimum
            if count + remaining < min_size:
                return

            # Prune: can't beat best
            if len(degs) + remaining <= best_distinct:
                return

            # Base case: all vertices processed
            if idx == n:
                if count >= min_size:
                    nd = len(degs)
                    if nd > best_distinct:
                        best_distinct = nd
                        best_mask = mask
                return

            v = vertex_order[idx]
            bit = 1 << v

            # Option 1: include vertex v
            new_mask = mask | bit
            # Compute degree of v in new_mask
            d_v = (adj[v] & new_mask).bit_count()
            # Check if this degree is new
            if d_v not in degs:
                new_degs = degs | {d_v}
            else:
                new_degs = degs
            rec(idx + 1, new_mask, new_degs, count + 1)

            # Option 2: exclude vertex v
            rec(idx + 1, mask, degs, count)

        # Initial call
        rec(0, 0, frozenset(), 0)

    branch_and_bound()

    # Extract vertices from best_mask
    verts = []
    m = best_mask
    while m:
        low = m & (-m)
        v = low.bit_length() - 1
        verts.append(v)
        m ^= low

    # Ensure return type is list of ints
    return [int(v) for v in verts]
# EVOLVE-BLOCK-END
