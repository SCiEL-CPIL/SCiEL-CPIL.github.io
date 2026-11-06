import time
import random
import sys

def construct():
    """
    Construct a Q3-free graph on 16 vertices maximizing edges.
    Uses local search with fast Q3 detection.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n = 16
    # Total possible edges
    total_edges = n * (n - 1) // 2  # 120

    # Represent graph as adjacency matrix (list of bytearrays for speed)
    # adj[i][j] = 1 if edge (i,j) exists
    adj = [bytearray(n) for _ in range(n)]

    def has_edge(i, j):
        return adj[i][j]

    def add_edge(i, j):
        adj[i][j] = 1
        adj[j][i] = 1

    def remove_edge(i, j):
        adj[i][j] = 0
        adj[j][i] = 0

    def edge_count():
        return sum(sum(row) for row in adj) // 2

    # Fast Q3 detection:
    # A 3-cube Q3 has 8 vertices and 12 edges.
    # It can be characterized as: pick a vertex v with 3 neighbors a,b,c
    # that are pairwise non-adjacent. Then the other 4 vertices must be
    # a^b, a^c, b^c, a^b^c where each pair (x,y) in the "opposite" set
    # are adjacent to exactly the right vertices.
    #
    # More precisely: if v is connected to a, b, c (pairwise non-adjacent),
    # then we need vertices d, e, f, g such that:
    #   d is adjacent to a and b but not c (and not v)
    #   e is adjacent to a and c but not b (and not v)
    #   f is adjacent to b and c but not a (and not v)
    #   g is adjacent to d, e, f but not v, a, b, c
    # And d,e,f,g form the rest of the cube.
    #
    # Actually, let's use a simpler characterization:
    # Q3 is the graph with vertices {0,1,2,3,4,5,6,7} where edge exists
    # if Hamming distance is 1.
    #
    # Key structural property: Q3 is bipartite with parts of size 4.
    # Each vertex has degree 3. The graph is 3-regular on 8 vertices.
    #
    # For detection: enumerate all 8-vertex subsets? That's C(16,8) = 12870.
    # For each subset, check if it contains a Q3 as a subgraph.
    # But that's too slow for repeated calls.
    #
    # Better: use the fact that in a Q3, every vertex has exactly 3 neighbors
    # within the Q3. So if we find 8 vertices that induce a 3-regular bipartite
    # graph with 12 edges, that's a Q3.
    #
    # Even better local check: for each vertex v, consider its neighbors.
    # If v has neighbors a, b, c that are pairwise non-adjacent, then check
    # if there exist vertices d, e, f, g completing the cube.

    def find_q3():
        """Find a Q3 subgraph if one exists. Returns (v, a, b, c, d, e, f, g) or None."""
        for v in range(n):
            # Get neighbors of v
            nbrs = [u for u in range(n) if adj[v][u] and u != v]
            if len(nbrs) < 3:
                continue
            # Try all triples of neighbors that are pairwise non-adjacent
            for i in range(len(nbrs)):
                a = nbrs[i]
                for j in range(i + 1, len(nbrs)):
                    b = nbrs[j]
                    if adj[a][b]:
                        continue
                    for k in range(j + 1, len(nbrs)):
                        c = nbrs[k]
                        if adj[a][c] or adj[b][c]:
                            continue
                        # Now check if we can complete the cube
                        # d must be adjacent to a and b, not to c, not to v
                        # e must be adjacent to a and c, not to b, not to v
                        # f must be adjacent to b and c, not to a, not to v
                        # g must be adjacent to d, e, f, not to v,a,b,c
                        # And d,e,f must be pairwise non-adjacent? No, in Q3:
                        # d and e are not adjacent (distance 2), d and f not adjacent, e and f not adjacent.
                        # g is adjacent to d, e, f.
                        #
                        # Let's search for d, e, f, g.
                        for d in range(n):
                            if d in (v, a, b, c):
                                continue
                            if not (adj[a][d] and adj[b][d]):
                                continue
                            if adj[c][d] or adj[v][d]:
                                continue
                            for e in range(n):
                                if e in (v, a, b, c, d):
                                    continue
                                if not (adj[a][e] and adj[c][e]):
                                    continue
                                if adj[b][e] or adj[v][e]:
                                    continue
                                if adj[d][e]:
                                    continue  # d and e should not be adjacent
                                for f in range(n):
                                    if f in (v, a, b, c, d, e):
                                        continue
                                    if not (adj[b][f] and adj[c][f]):
                                        continue
                                    if adj[a][f] or adj[v][f]:
                                        continue
                                    if adj[d][f] or adj[e][f]:
                                        continue
                                    # Now find g adjacent to d, e, f but not to v,a,b,c
                                    for g in range(n):
                                        if g in (v, a, b, c, d, e, f):
                                            continue
                                        if adj[v][g] or adj[a][g] or adj[b][g] or adj[c][g]:
                                            continue
                                        if adj[d][g] and adj[e][g] and adj[f][g]:
                                            return (v, a, b, c, d, e, f, g)
        return None

    # Initialize with a good starting point
    # Start with a random dense graph and remove edges that create Q3
    random.seed(42)

    # Strategy: start with all edges, then greedily remove to make Q3-free
    # But that gives few edges. Better: start sparse and add edges.
    #
    # Let's start with a known Q3-free structure and add edges.
    # K3,13 has 39 edges. Can we do better?
    #
    # Try: start with random graph with moderate density, then local search.

    # Initialize adj to all zeros
    for i in range(n):
        for j in range(n):
            adj[i][j] = 0

    # Add a base structure: complete bipartite K3,13
    # Vertices 0,1,2 on one side, 3..15 on other
    for a in range(3):
        for b in range(3, 16):
            add_edge(a, b)

    # Now try to add more edges via local search
    # First, try adding all missing edges one by one, keeping only those that don't create Q3

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Phase 1: Greedy addition
    # Shuffle and try to add each edge
    random.shuffle(all_pairs)
    for (u, v) in all_pairs:
        if adj[u][v]:
            continue
        # Try adding edge (u,v)
        add_edge(u, v)
        if find_q3() is not None:
            remove_edge(u, v)

    # Phase 2: Local search - try to swap edges or add edges
    # Try adding edges, and if that fails, try removing one edge and adding two others
    current_edges = edge_count()
    best_edges = current_edges
    best_adj = [row[:] for row in adj]

    no_improve_count = 0
    max_no_improve = 50

    while time.time() < deadline - 5:
        improved = False
        # Try adding a random missing edge
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            u, v = random.choice(all_pairs)
            if adj[u][v]:
                continue
            add_edge(u, v)
            if find_q3() is None:
                current_edges += 1
                improved = True
            else:
                remove_edge(u, v)

        # Try removing one edge and adding two others
        if not improved:
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                # Pick a random existing edge to remove
                existing = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i][j]]
                if not existing:
                    break
                rem_u, rem_v = random.choice(existing)
                remove_edge(rem_u, rem_v)
                current_edges -= 1

                # Try to add two edges
                added = 0
                for (au, av) in all_pairs:
                    if time.time() > deadline - 5:
                        break
                    if adj[au][av]:
                        continue
                    if (au, av) == (rem_u, rem_v) or (av, au) == (rem_u, rem_v):
                        continue
                    add_edge(au, av)
                    if find_q3() is None:
                        current_edges += 1
                        added += 1
                        if added == 2:
                            break
                    else:
                        remove_edge(au, av)

                if added >= 1:
                    improved = True
                else:
                    # Revert
                    add_edge(rem_u, rem_v)
                    current_edges += 1

        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [row[:] for row in adj]
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Restore best and try different strategy
                for i in range(n):
                    for j in range(n):
                        adj[i][j] = best_adj[i][j]
                current_edges = best_edges
                no_improve_count = 0
                # Perturb: remove a random edge and try to re-add
                existing = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i][j]]
                if existing:
                    ru, rv = random.choice(existing)
                    remove_edge(ru, rv)
                    current_edges -= 1

    # Restore best
    for i in range(n):
        for j in range(n):
            adj[i][j] = best_adj[i][j]

    # Final check: verify no Q3
    if find_q3() is not None:
        # Should not happen, but safety: fall back to K3,13
        return [[a, b] for a in range(3) for b in range(3, 16)]

    # Return edge list
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                result.append([i, j])
    return result
