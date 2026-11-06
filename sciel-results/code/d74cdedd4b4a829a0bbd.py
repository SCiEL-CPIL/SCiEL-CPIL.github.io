"""Dense 4-cycle-rainbow edge-coloured graph on 20 vertices, 20 colours."""
import time
import random
import itertools

N = 20
C = 20
DEADLINE = 115  # seconds, safety margin before 120s kill

def construct():
    start_time = time.time()
    best_edges = None
    best_count = 0

    # Precompute all 4-cycles (as sets of 4 vertex pairs) for fast checking
    # A 4-cycle is a set of 4 edges forming a cycle of length 4
    # We'll use a different approach: check validity incrementally

    # For each pair of vertices (u,v), track which colours are used
    # For each 4-cycle, track the set of colours used

    # Key insight: when adding edge (u,v,c), we need to check that no 4-cycle
    # containing this edge has a repeated colour.

    # Precompute: for each pair (u,v), the list of 4-cycles containing edge (u,v)
    # A 4-cycle containing (u,v) has form u-v-w-x-u where w,x are other vertices
    # So for edge (u,v), the 4-cycles are: for each pair (w,x) with w!=v,x and x!=u,w
    #   and w-x is an edge, u-x is an edge, v-w is an edge
    # But we don't know which edges exist yet, so we need a different approach.

    # Better approach: maintain adjacency and for each vertex pair, the set of colours used.
    # When adding (u,v,c), check all 4-cycles through (u,v).
    # A 4-cycle through (u,v) is: u-v-w-x-u where edges (v,w), (w,x), (x,u) exist.
    # For this to be a 4-cycle, we need w != u, w != v, x != u, x != v, x != w.

    # Let's use a smarter approach:
    # adj[u][v] = set of colours used on edge (u,v), or None if no edge
    # When adding (u,v,c):
    #   For each w in range(N) where w != u and w != v:
    #     If edge (v,w) exists:
    #       For each x in range(N) where x != u and x != v and x != w:
    #         If edge (w,x) exists and edge (x,u) exists:
    #           Check that c is not in the colours of (v,w), (w,x), (x,u)

    # This is O(N^2) per edge addition, which is fast enough.

    def check_and_add(edges_dict, adj, u, v, c):
        """Check if adding edge (u,v,c) is valid. If so, add it and return True."""
        if u == v:
            return False
        if (min(u,v), max(u,v)) in edges_dict:
            return False

        # Check all 4-cycles containing edge (u,v)
        # A 4-cycle: u-v-w-x-u
        for w in range(N):
            if w == u or w == v:
                continue
            vw_key = (min(v,w), max(v,w))
            if vw_key not in edges_dict:
                continue
            vw_colours = edges_dict[vw_key]
            if c in vw_colours:
                return False
            for x in range(N):
                if x == u or x == v or x == w:
                    continue
                wx_key = (min(w,x), max(w,x))
                xu_key = (min(x,u), max(x,u))
                if wx_key not in edges_dict or xu_key not in edges_dict:
                    continue
                wx_colours = edges_dict[wx_key]
                xu_colours = edges_dict[xu_key]
                if c in wx_colours or c in xu_colours:
                    return False
                # Also check that the other three edges don't share colours with each other
                # Wait, we only need to check that the 4-cycle has 4 distinct colours.
                # The colours of (v,w), (w,x), (x,u) are already fixed.
                # We need c != colour(v,w), c != colour(w,x), c != colour(x,u).
                # But those edges might have multiple colours? No, each edge has one colour.
                # Actually in this problem, each edge has exactly one colour.
                # So edges_dict[key] should be a single colour, not a set.

        # Let me reconsider: each edge has one colour. So:
        # edges_dict[(u,v)] = c (single colour)
        # adj[u][v] = c or None

        return True

    # Let me restart with a cleaner design
    # edges: dict from (min(u,v), max(u,v)) to colour
    # adj: 2D list, adj[u][v] = colour or -1

    def is_valid_addition(adj, u, v, c):
        """Check if adding edge (u,v) with colour c is valid (no non-rainbow 4-cycle)."""
        # For each 4-cycle containing (u,v): u-v-w-x-u
        # Need: c != col(v,w), c != col(w,x), c != col(x,u)
        # AND the existing 4-cycle (if (v,w),(w,x),(x,u) all exist) must already be rainbow
        #   i.e., col(v,w) != col(w,x) != col(x,u) != col(v,w) -- but this was already checked when those edges were added
        
        for w in range(N):
            if w == u or w == v:
                continue
            if adj[v][w] < 0:
                continue
            for x in range(N):
                if x == u or x == v or x == w:
                    continue
                if adj[w][x] < 0 or adj[x][u] < 0:
                    continue
                # 4-cycle u-v-w-x-u exists (or will exist after adding u-v)
                # Colours: c, adj[v][w], adj[w][x], adj[x][u]
                if c == adj[v][w] or c == adj[w][x] or c == adj[x][u]:
                    return False
        return True

    def greedy_build(order, adj):
        """Build a graph by adding edges in the given order, picking valid colours."""
        new_adj = [[-1]*N for _ in range(N)]
        count = 0
        for (u, v) in order:
            if u == v:
                continue
            if new_adj[u][v] >= 0:
                continue
            # Try to find a valid colour
            found = False
            for c in range(C):
                if is_valid_addition(new_adj, u, v, c):
                    new_adj[u][v] = c
                    new_adj[v][u] = c
                    count += 1
                    found = True
                    break
            if not found:
                # Try all colours more carefully? No, if no colour works, skip.
                pass
        return new_adj, count

    # Generate edge orderings
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]

    # Strategy 1: Random orders with multiple restarts
    rng = random.Random(42)
    for restart in range(5000):
        if time.time() - start_time > DEADLINE:
            break
        order = all_pairs[:]
        rng.shuffle(order)
        adj, count = greedy_build(order, None)
        if count > best_count:
            best_count = count
            best_edges = [[u, v, adj[u][v]] for (u, v) in all_pairs if adj[u][v] >= 0]
            if best_count >= 180:  # very dense, unlikely
                break

    # Strategy 2: Structured orders (e.g., by distance, or specific patterns)
    # Try ordering by vertex degree potential
    for restart in range(2000):
        if time.time() - start_time > DEADLINE:
            break
        # Random permutation of vertices, then order edges by (perm[u], perm[v])
        perm = list(range(N))
        rng.shuffle(perm)
        pos = [0]*N
        for i, v in enumerate(perm):
            pos[v] = i
        order = sorted(all_pairs, key=lambda p: (pos[p[0]], pos[p[1]]))
        adj, count = greedy_build(order, None)
        if count > best_count:
            best_count = count
            best_edges = [[u, v, adj[u][v]] for (u, v) in all_pairs if adj[u][v] >= 0]

    # Strategy 3: Local improvement - try to add more edges to best solution
    if best_edges is not None:
        adj = [[-1]*N for _ in range(N)]
        for u, v, c in best_edges:
            adj[u][v] = c
            adj[v][u] = c

        # Try to add edges that were skipped
        improved = True
        while improved:
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            for u in range(N):
                for v in range(u+1, N):
                    if adj[u][v] >= 0:
                        continue
                    for c in range(C):
                        if is_valid_addition(adj, u, v, c):
                            adj[u][v] = c
                            adj[v][u] = c
                            best_count += 1
                            improved = True
                            break

        best_edges = [[u, v, adj[u][v]] for (u, v) in all_pairs if adj[u][v] >= 0]

    # Strategy 4: Try different greedy strategies - prioritise high-degree vertices
    for restart in range(3000):
        if time.time() - start_time > DEADLINE:
            break
        # Weighted random order
        weights = [rng.random() for _ in range(N)]
        order = sorted(all_pairs, key=lambda p: (weights[p[0]] + weights[p[1]], rng.random()))
        adj, count = greedy_build(order, None)
        if count > best_count:
            best_count = count
            best_edges = [[u, v, adj[u][v]] for (u, v) in all_pairs if adj[u][v] >= 0]

    # Final local improvement
    if best_edges is not None:
        adj = [[-1]*N for _ in range(N)]
        for u, v, c in best_edges:
            adj[u][v] = c
            adj[v][u] = c

        improved = True
        while improved:
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            for u in range(N):
                for v in range(u+1, N):
                    if adj[u][v] >= 0:
                        continue
                    for c in range(C):
                        if is_valid_addition(adj, u, v, c):
                            adj[u][v] = c
                            adj[v][u] = c
                            best_count += 1
                            improved = True
                            break

        best_edges = [[u, v, adj[u][v]] for (u, v) in all_pairs if adj[u][v] >= 0]

    if best_edges is None:
        # Fallback: star
        best_edges = [[0, v, 0] for v in range(1, N)]

    return best_edges
