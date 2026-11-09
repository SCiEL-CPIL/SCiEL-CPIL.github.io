# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a maximum square-difference-free set modulo 205 using a bitset-based
    branch-and-bound Maximum Independent Set solver with a greedy coloring bound.

    The forbidden differences are the nonzero quadratic residues mod 205.
    We model the problem as finding a maximum independent set in the Cayley
    graph Cay(Z/205Z, F) where F is the set of forbidden differences.
    """
    m = 205
    k = 2

    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)

    # Build adjacency bitsets using Python's arbitrary-precision integers as bitsets
    # adj[i] is a bitset where bit j is set if i and j are adjacent (forbidden difference)
    adj = [0] * m
    for i in range(m):
        for f in forbidden:
            j1 = (i + f) % m
            j2 = (i - f) % m
            adj[i] |= (1 << j1)
            adj[i] |= (1 << j2)
    # Remove self-loops (shouldn't exist since 0 not in forbidden, but be safe)
    for i in range(m):
        adj[i] &= ~(1 << i)

    # Greedy coloring of the remaining candidate set to get an upper bound
    # We use a simple greedy coloring that assigns colors in order
    def greedy_color_bound(candidates_bitset):
        """
        Given a bitset of candidate vertices, greedily color them to get an upper
        bound on the maximum independent set size within this set.
        
        The coloring is done on the subgraph induced by candidates.
        We use a sequential greedy approach: repeatedly pick a maximal independent
        set as one color class.
        """
        if candidates_bitset == 0:
            return 0
        
        remaining = candidates_bitset
        color_count = 0
        
        while remaining != 0:
            # Find a maximal independent set in the remaining subgraph
            # Greedy: pick vertices in order, adding if no conflict with already picked
            independent_set = 0
            temp = remaining
            while temp != 0:
                # Pick the lowest set bit
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                # Check if v is adjacent to any vertex in independent_set
                # v is compatible with independent_set if no neighbor of v is in independent_set
                if (adj[v] & independent_set) == 0:
                    independent_set |= lb
                    # Remove v and its neighbors from temp
                    temp &= ~(1 << v)
                    temp &= ~adj[v]
            color_count += 1
            # Remove the independent set from remaining
            remaining &= ~independent_set
        
        return color_count

    # Branch and bound
    best_size = 0
    best_set = 0  # bitset representation of the best independent set found

    def branch_and_bound(candidates_bitset, current_set, current_size):
        nonlocal best_size, best_set
        
        if candidates_bitset == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
        
        # Upper bound: current_size + max possible from candidates
        # Use greedy coloring bound
        ub = current_size + greedy_color_bound(candidates_bitset)
        if ub <= best_size:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick the vertex with highest degree in the candidate subgraph
        # to prune more effectively
        temp = candidates_bitset
        best_v = -1
        best_deg = -1
        while temp != 0:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            # Degree in candidate subgraph
            deg = bin(adj[v] & candidates_bitset).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lb
        
        if best_v == -1:
            return
        
        # Branch 1: Include best_v
        # Remove best_v and all its neighbors from candidates
        new_candidates = candidates_bitset & ~(1 << best_v) & ~adj[best_v]
        new_set = current_set | (1 << best_v)
        branch_and_bound(new_candidates, new_set, current_size + 1)
        
        # Branch 2: Exclude best_v
        new_candidates = candidates_bitset & ~(1 << best_v)
        branch_and_bound(new_candidates, current_set, current_size)

    # Initialize with a greedy solution to get a good lower bound
    # Greedy: pick vertices in order, add if no conflict
    greedy_set = 0
    for v in range(m):
        if (adj[v] & greedy_set) == 0:
            greedy_set |= (1 << v)
    best_size = bin(greedy_set).count('1')
    best_set = greedy_set

    # Run branch and bound
    full_candidates = (1 << m) - 1
    branch_and_bound(full_candidates, 0, 0)

    # Extract the best set
    result = []
    for v in range(m):
        if (best_set >> v) & 1:
            result.append(v)

    return np.array(result, dtype=np.int64)


# EVOLVE-BLOCK-END
