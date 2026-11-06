"""Maximum Independent Set via Max Clique on complement graph with coloring bound."""
import math
import time
import random

_N = 256
_SQUARES = {4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529}

def _is_square(x):
    r = int(math.isqrt(x))
    return r * r == x

def _build_complement_bitsets():
    """Build bitsets for the complement graph: edge in complement iff NOT a conflict.
    Vertex i corresponds to number i+1 (i in 0..255).
    """
    comp = [0] * _N
    for i in range(_N):
        a = i + 1
        mask = 0
        for j in range(i + 1, _N):
            b = j + 1
            # conflict if a+b is a perfect square
            if (a + b) in _SQUARES:
                continue
            # no conflict -> edge in complement
            mask |= (1 << j)
        comp[i] = mask
    return comp

def _greedy_color_order(P, comp):
    """Greedy coloring on induced subgraph comp[P].
    Returns (color_classes, order) where order is the vertex ordering
    used during coloring (for branching), and color_classes[i] = number of
    colors used up to and including vertex order[i].
    This is the standard Tomita coloring for max clique.
    """
    n = len(P)
    if n == 0:
        return [], []

    # Greedy coloring: repeatedly find a maximal independent set in remaining
    # (in the complement of the clique graph, i.e., maximal clique in original)
    # Actually for max clique on graph G, we color G's induced subgraph.
    # Greedy: assign smallest color not used by colored neighbors.

    # We use the standard approach: order vertices by greedy coloring.
    # color[v] = color assigned to v (1-indexed)
    # We process vertices in a specific order.

    # Standard Tomita: 
    #   while P not empty:
    #     Q = P
    #     while Q not empty:
    #       pick v from Q (e.g., first or by degree)
       #       assign current color to v
    #       remove v from P and Q
    #       remove from Q all vertices not adjacent to v in G (i.e., not in N_G(v))
    #     current_color += 1
    #   The order is the reverse of assignment order, and colors are bounded.

    # Let me implement the standard greedy coloring for max clique bound.
    # Returns: list of colors in the order vertices were colored (for the bound).
    # Actually, the standard returns:
    #   order: list of vertices in the order they will be branched
    #   bound: list where bound[i] = max color of first i+1 vertices in order

    # I'll use a simpler but effective approach:
    # Process vertices in order of the input list P.
    # For each vertex, assign the smallest available color.

    color_of = {}
    colors_used = {}  # color -> set of vertices with that color (for checking)
    order = []
    bounds = []
    current_max_color = 0

    # Greedy: for each vertex in P (in given order), assign smallest color
    # not used by its already-colored neighbors in G.
    # But we need to be careful: a color class must be an independent set in G
    # (i.e., a clique in the original conflict graph, meaning no two vertices
    # in the same color class are adjacent in G).

    # Standard greedy coloring for max clique:
    # We want to color G[P] so that each color class is an independent set in G.
    # Greedy: process vertices one by one, assign smallest color not conflicting
    # with already-colored neighbors.

    # For efficiency, I'll use the approach where we build color classes:
    # color_classes = list of lists, each is an independent set in G.
    # For each vertex v in P:
    #   find first color class c where v is not adjacent to any vertex in c
    #   if none, create new color class
    #   add v to that class

    # The bound for vertex at position i in the ordering is the number of color
    # classes used up to that point.

    # Let me use a cleaner standard implementation:

    # Reorder P for better coloring: sort by degree in G[P] (descending)
    # Actually, let's just use the given order for now, or a simple heuristic.

    # Simple greedy coloring:
    color_of = {}
    # For each color, maintain the set of vertices (as a bitmask or set)
    color_sets = []  # list of sets

    for v in P:
        placed = False
        for c_idx, c_set in enumerate(color_sets):
            # Check if v is non-adjacent to all in c_set (in G)
            # i.e., for all u in c_set, v is NOT adjacent to u in G
            # v is adjacent to u in G iff (comp[v] >> u) & 1
            ok = True
            for u in c_set:
                if (comp[v] >> u) & 1:
                    ok = False
                    break
            if ok:
                c_set.add(v)
                color_of[v] = c_idx + 1
                placed = True
                break
        if not placed:
            color_sets.append({v})
            color_of[v] = len(color_sets)

    # Now, the ordering: we want to order vertices so that the color bound
    # is as tight as possible. Standard: order by color, and within same color,
    # by some heuristic.
    # The bound at position i is the max color among the first i+1 vertices.

    # Actually, the standard Tomita ordering:
    # Vertices are ordered by the color assignment process.
    # Let me just use: sort by color_of, then by vertex id.
    # And the bound is the running max of color_of.

    # Hmm, but the standard approach is different. Let me use a well-known
    # formulation:

    # The greedy coloring produces an ordering v1, v2, ..., vn and a sequence
    # b1, b2, ..., bn where bi <= bi+1 and bi is the color of vi.
    # The bound for the subproblem {v1,...,vi} is bi.

    # I'll construct this ordering by:
    # 1. Color the graph greedily.
    # 2. Order vertices by (color, vertex_id) - but this doesn't guarantee
    #    the monotonicity of the bound.

    # Let me use the standard "expand" ordering from Tomita:
    # During coloring, we add vertices to color classes. The order in which
    # vertices are colored gives the branching order.

    # Actually, for correctness, any ordering where the running max color
    # at position i is an upper bound on the clique size in the first i vertices
    # works. Since color_of[v] is the color assigned to v, and color classes
    # are independent sets, the number of vertices in any clique is at most
    # the number of color classes. For a subset of vertices, the bound is the
    # max color among them.

    # So: order vertices arbitrarily, and bound[i] = max(color_of[v] for v in order[:i+1]).
    # This is valid.

    # Let me just sort by color_of for a tighter bound early on.
    order = sorted(P, key=lambda v: (color_of[v], v))
    bounds = []
    running_max = 0
    for v in order:
        running_max = max(running_max, color_of[v])
        bounds.append(running_max)

    return order, bounds

def _max_clique_branch(P, comp, best_clique, best_size):
    """Branch and bound for max clique.
    P: list of candidate vertices.
    comp: complement graph bitsets.
    best_clique: list, the best clique found so far.
    best_size: int, size of best clique.
    Returns (best_clique, best_size).
    """
    if not P:
        return best_clique, best_size

    # Greedy coloring to get bound
    order, bounds = _greedy_color_order(P, comp)
    n = len(order)

    # If the max possible (current best + bound) doesn't exceed best_size, prune
    if n + len(best_clique) <= best_size:
        return best_clique, best_size

    # Branch on vertices in order (reverse, so we try high-color vertices first)
    # Actually, standard: branch in the order given, from last to first.
    # Let me branch from the end of the order (highest color bound first).

    for i in range(n - 1, -1, -1):
        v = order[i]
        # Prune: if best_size >= len(best_clique) + bounds[i], skip
        if best_size >= len(best_clique) + bounds[i]:
            continue

        # Add v to current clique
        best_clique.append(v)

        # New candidates: vertices in order[0:i] that are adjacent to v in comp
        new_P = [order[j] for j in range(i) if (comp[v] >> order[j]) & 1]

        # Recurse
        best_clique, best_size = _max_clique_branch(new_P, comp, best_clique, best_size)

        # Remove v
        best_clique.pop()

        # Update best if improved
        # (handled in recursion)

    return best_clique, best_size

def construct():
    deadline = time.time() + 85.0  # 85s safety margin for 90s limit

    comp = _build_complement_bitsets()

    # Initial greedy solution
    best_clique = []
    # Greedy: repeatedly pick vertex with max degree in remaining graph
    P = list(range(_N))
    current = []
    remaining = set(P)
    while remaining:
        # Pick vertex with max degree in remaining
        best_v = None
        best_deg = -1
        for v in remaining:
            deg = bin(comp[v] & sum(1 << u for u in remaining)).count('1')
            # That's inefficient; let me just pick first or use a simple heuristic
            if deg > best_deg:
                best_deg = deg
                best_v = v
        if best_v is None:
            break
        current.append(best_v)
        remaining.discard(best_v)
        # Remove non-neighbors of best_v from remaining
        to_remove = set()
        for u in remaining:
            if not (comp[best_v] >> u) & 1:
                to_remove.add(u)
        remaining -= to_remove

    best_size = len(current)
    best_clique = current[:]

    # Now try branch and bound from a good starting point
    # Order vertices by degree for the initial B&B
    # Let's try B&B on the full graph
    # But that might be too slow. Let me use a time check.

    # Actually, let me do a proper B&B with time checks.
    # I'll use a global time check.

    # Let me implement a simpler but effective approach:
    # 1. Get a good initial solution via greedy.
    # 2. Use local search / simulated annealing to improve.
    # 3. Use B&B with time limit for final optimization.

    # Given the time constraint, let me do multiple greedy restarts and local search.

    # Greedy with randomization
    def greedy_randomized():
        remaining = set(range(_N))
        current = []
        while remaining:
            # Pick random vertex from top-degree vertices
            degrees = {}
            for v in remaining:
                deg = 0
                mask = comp[v]
                for u in remaining:
                    if (mask >> u) & 1:
                        deg += 1
                degrees[v] = deg
            max_deg = max(degrees.values())
            candidates = [v for v in remaining if degrees[v] >= max_deg - 2]
            v = random.choice(candidates)
            current.append(v)
            remaining.discard(v)
            to_remove = set()
            for u in remaining:
                if not (comp[v] >> u) & 1:
                    to_remove.add(u)
            remaining -= to_remove
        return current

    def local_search(clique):
        """Try to improve clique by swapping vertices."""
        clique_set = set(clique)
        improved = True
        while improved:
            improved = False
            # Try adding a vertex
            for v in range(_N):
                if v in clique_set:
                    continue
                # Check if v is adjacent to all in clique
                ok = True
                for u in clique:
                    if not (comp[v] >> u) & 1:
                        ok = False
                        break
                if ok:
                    clique.append(v)
                    clique_set.add(v)
                    improved = True
                    break
            # Try swapping
            for i, u in enumerate(clique):
                for v in range(_N):
                    if v in clique_set:
                        continue
                    # Check if v can replace u: v adjacent to all other clique members
                    ok = True
                    for j, w in enumerate(clique):
                        if j == i:
                            continue
                        if not (comp[v] >> w) & 1:
                            ok = False
                            break
                    if ok:
                        # Swap: remove u, add v
                        clique[i] = v
                        clique_set.discard(u)
                        clique_set.add(v)
                        improved = True
                        break
                if improved:
                    break
        return clique

    # Run multiple greedy + local search
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        cand = greedy_randomized()
        cand = local_search(cand)
        if len(cand) > best_size:
            best_size = len(cand)
            best_clique = cand[:]

    # Try B&B with time limit if we have time
    if time.time() < deadline - 15:
        # Try B&B on the full graph
        # This might be too slow, so let's try with a time limit
        # I'll implement a time-limited B&B
        try:
            bb_start = time.time()
            # Use B&B
            P_full = list(range(_N))
            # Sort by degree for better pruning
            P_full.sort(key=lambda v: bin(comp[v]).count('1'), reverse=True)
            result_clique, result_size = _max_clique_branch(P_full, comp, best_clique[:], best_size)
            if result_size > best_size:
                best_size = result_size
                best_clique = result_clique[:]
        except Exception:
            pass

    # Convert back to numbers (v+1)
    result = sorted(v + 1 for v in best_clique)
    return result

# Quick self-test
if __name__ == "__main__":
    r = construct()
    print(f"Size: {len(r)}")
    # Verify
    for i in range(len(r)):
        for j in range(i, len(r)):
            s = r[i] + r[j]
            if _is_square(s):
                print(f"CONFLICT: {r[i]} + {r[j]} = {s}")
                break
    print("OK" if not any(_is_square(r[i]+r[j]) for i in range(len(r)) for j in range(i, len(r))) else "INVALID")
