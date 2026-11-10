# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys
from collections import defaultdict

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find the largest subset of Z/205Z with no two elements differing by a
    nonzero square mod 205.

    Uses:
      - Bitset-based conflict graph for fast compatibility checks
      - Branch-and-bound maximum independent set (with correct coloring bound)
      - Efficient local search with greedy refill
      - Multiple random restarts
    """
    m = 205
    deadline = time.time() + 590  # safety margin before 630s

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_set = forbidden

    # Build conflict graph as bitsets: conflict_mask[v] is a 205-bit int
    # where bit u is set iff (v - u) % m is in forbidden_set
    conflict_mask = [0] * m
    for v in range(m):
        mask = 0
        for u in range(m):
            if u == v:
                continue
            d = (v - u) % m
            if d in forbidden_set:
                mask |= (1 << u)
        conflict_mask[v] = mask

    # Complement adjacency: comp_adj[v] has bits set for vertices NOT adjacent to v (and not v itself)
    comp_adj = [0] * m
    for v in range(m):
        comp_adj[v] = ((1 << m) - 1) ^ conflict_mask[v] ^ (1 << v)

    def is_compatible(candidate, selected_bits):
        """Check if candidate is compatible with all elements in selected_bits (a bitmask)."""
        return (conflict_mask[candidate] & selected_bits) == 0

    def greedy_color_bound(candidates_bits):
        """
        Greedy coloring of the complement subgraph induced by candidates.
        A clique in the original graph = independent set in complement.
        The number of colors needed to color the complement subgraph
        is an upper bound on the size of a clique in the original subgraph,
        which equals the size of an independent set in the original subgraph.
        
        Wait - let me think again. We want to find an independent set in the ORIGINAL graph.
        An independent set in G is a clique in complement(G).
        So we need to color the ORIGINAL subgraph, and the chromatic number is an upper bound
        on the clique number of the complement, which is the independence number of G.
        
        Actually: alpha(G[S]) <= chi(G[S]) for any subset S.
        So we color G[S] (the original conflict graph restricted to candidates).
        """
        cands = []
        temp = candidates_bits
        while temp:
            low = temp & (-temp)
            idx = low.bit_length() - 1
            cands.append(idx)
            temp ^= low

        if not cands:
            return 0

        # Greedy coloring of the original graph induced by cands
        # Process vertices in order of decreasing degree in the subgraph
        cand_set = set(cands)
        cands.sort(key=lambda v: -bin(conflict_mask[v] & candidates_bits).count('1'))

        color = {}
        num_colors = 0

        for v in cands:
            # Find the smallest color not used by any neighbor in cands
            used = set()
            v_conflicts = conflict_mask[v] & candidates_bits
            t = v_conflicts
            while t:
                low = t & (-t)
                idx = low.bit_length() - 1
                if idx in color:
                    used.add(color[idx])
                t ^= low

            c = 0
            while c in used:
                c += 1
            color[v] = c
            num_colors = max(num_colors, c + 1)

        return num_colors

    # Branch and bound for maximum independent set
    best_size = [0]
    best_set = [0]
    nodes_explored = [0]

    def branch_and_bound(candidates_bits, current_size, current_set_bits):
        """
        Branch-and-bound for maximum independent set.
        """
        nonlocal best_size, best_set, nodes_explored

        if time.time() > deadline:
            return

        nodes_explored[0] += 1

        # Upper bound
        ub = current_size + greedy_color_bound(candidates_bits)
        if ub <= best_size[0]:
            return

        if candidates_bits == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = current_set_bits
            return

        # Pick a vertex to branch on - choose one with high degree in subgraph
        cands = []
        temp = candidates_bits
        while temp:
            low = temp & (-temp)
            idx = low.bit_length() - 1
            cands.append(idx)
            temp ^= low

        # Sort by degree descending
        cands.sort(key=lambda v: -bin(conflict_mask[v] & candidates_bits).count('1'))

        v = cands[0]

        # Branch 1: include v
        new_candidates = candidates_bits & comp_adj[v]  # remove v and its neighbors
        branch_and_bound(new_candidates, current_size + 1, current_set_bits | (1 << v))

        # Branch 2: exclude v
        new_candidates = candidates_bits & ~(1 << v)
        branch_and_bound(new_candidates, current_size, current_set_bits)

    # Greedy solution
    def greedy_solution(order):
        selected = 0
        count = 0
        for v in order:
            if is_compatible(v, selected):
                selected |= (1 << v)
                count += 1
        return count, selected

    # Multiple restarts with different orderings
    rng = random.Random(42)
    best_size[0] = 0
    best_set[0] = 0

    for restart in range(30):
        if time.time() > deadline - 120:
            break

        if restart == 0:
            ord_ = list(range(m))
        elif restart == 1:
            ord_ = sorted(range(m), key=lambda v: bin(conflict_mask[v]).count('1'))
        elif restart == 2:
            ord_ = sorted(range(m), key=lambda v: -bin(conflict_mask[v]).count('1'))
        else:
            ord_ = list(range(m))
            rng.shuffle(ord_)

        size, set_bits = greedy_solution(ord_)
        if size > best_size[0]:
            best_size[0] = size
            best_set[0] = set_bits

    # Local search with greedy refill
    def local_search(set_bits, max_iter=500):
        current_size = bin(set_bits).count('1')
        improved = True
        iterations = 0

        while improved and iterations < max_iter:
            if time.time() > deadline - 60:
                break
            improved = False
            iterations += 1

            # Try adding any vertex not in current set
            for v in range(m):
                if not (set_bits & (1 << v)) and is_compatible(v, set_bits):
                    set_bits |= (1 << v)
                    current_size += 1
                    improved = True
                    break

            if improved:
                continue

            # Try 1-1 swap with greedy refill
            for u in range(m):
                if set_bits & (1 << u):
                    new_bits = set_bits & ~(1 << u)
                    # Try to add vertices greedily
                    test_bits = new_bits
                    test_size = current_size - 1
                    for v in range(m):
                        if not (test_bits & (1 << v)) and is_compatible(v, test_bits):
                            test_bits |= (1 << v)
                            test_size += 1
                    if test_size > current_size:
                        set_bits = test_bits
                        current_size = test_size
                        improved = True
                        break

        return current_size, set_bits

    # Apply local search
    if best_size[0] > 0:
        size, set_bits = local_search(best_set[0])
        if size > best_size[0]:
            best_size[0] = size
            best_set[0] = set_bits

    # Run branch-and-bound if time permits
    if time.time() < deadline - 30:
        # Use greedy solution as initial bound
        branch_and_bound((1 << m) - 1, 0, 0)

    # Extract the final set
    result = []
    for v in range(m):
        if best_set[0] & (1 << v):
            result.append(v)

    if not result:
        return np.array([0], dtype=np.int64)

    return np.array(sorted(result), dtype=np.int64)


# EVOLVE-BLOCK-END
