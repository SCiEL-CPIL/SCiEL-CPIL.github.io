"""Fewest four-cycles in graphs with 31 vertices and 91 edges."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    n = 31
    q = 5
    target_edges = 91

    # Generate points of PG(2, q)
    points = []
    # Points at infinity: [x:1:0] for x in 0..q-1, and [1:0:0]
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    # Finite points: [x:y:1] for x, y in 0..q-1
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))

    if len(points) != n:
        # Fallback to simple random graph if something goes wrong with coordinates
        import random
        random.seed(42)
        edges = []
        while len(edges) < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) not in edges:
                edges.append((u, v))
        return [[u, v] for u, v in edges]

    # Build adjacency
    neighbors = [set() for _ in range(n)]
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            # Polarity condition: x1*x2 + y1*y2 - z1*z2 == 0 mod q
            if (p1[0] * p2[0] + p1[1] * p2[1] - p1[2] * p2[2]) % q == 0:
                neighbors[i].add(j)
                neighbors[j].add(i)
                edges.append((i, j))

    current_edges = len(edges)

    # Find non-edges
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if j not in neighbors[i]:
                non_edges.append((i, j))

    # We need to add (target_edges - current_edges) edges.
    # Usually this is 1.
    edges_to_add = target_edges - current_edges

    added_edges = []

    # To be safe, if we need to add more than 1, we add them one by one,
    # updating neighbors after each addition to ensure correctness for subsequent additions.
    # However, for this specific task, edges_to_add should be 1.

    for _ in range(edges_to_add):
        if time.time() - start_time > DEADLINE:
            break
        best_edge = None
        min_cost = float('inf')

        # Recalculate non-edges if we added some?
        # If edges_to_add is 1, non_edges is static.
        # If > 1, we should remove added edges from non_edges list.
        # For simplicity and given the specific case (1 edge), we use the static list.
        # If we need to add multiple, we'd need to be careful.

        for u, v in non_edges:
            # Check if this edge was already added in this loop (if edges_to_add > 1)
            if (u, v) in added_edges:
                continue

            # Cost = number of paths u-x-w-v in current graph
            cost = 0
            for w in neighbors[v]:
                # Count common neighbors of w and u
                cost += len(neighbors[w] & neighbors[u])

            if cost < min_cost:
                min_cost = cost
                best_edge = (u, v)

        if best_edge:
            u, v = best_edge
            edges.append((u, v))
            neighbors[u].add(v)
            neighbors[v].add(u)
            added_edges.append((u, v))
        else:
            break

    # Helper to count C4s containing a specific edge (u, v)
    def count_c4_through_edge(u, v):
        count = 0
        # A C4 containing edge (u,v) is u-x-y-v where x != y, x != v, y != u, x-y is an edge
        # x is a neighbor of u (x != v)
        # y is a neighbor of v (y != u)
        # x-y is an edge
        # x != y
        for x in neighbors[u]:
            if x == v:
                continue
            for y in neighbors[v]:
                if y == u or y == x:
                    continue
                if y in neighbors[x]:
                    count += 1
        return count

    # Helper to count total C4s in the graph
    def count_total_c4():
        total = 0
        for u in range(n):
            for v in neighbors[u]:
                if u < v:
                    total += count_c4_through_edge(u, v)
        # Each C4 is counted 4 times (once for each edge)
        return total // 4

    # Local search: try to swap an edge for a non-edge to reduce C4 count
    # Current C4 count
    current_c4 = count_total_c4()

    # Precompute c(e) for all edges
    edge_c4 = {}
    for u, v in edges:
        edge_c4[(u, v)] = count_c4_through_edge(u, v)
        edge_c4[(v, u)] = count_c4_through_edge(v, u)

    # Local search loop
    improved = True
    while improved and (time.time() - start_time < DEADLINE):
        improved = False
        best_swap = None
        best_delta = 0

        # Iterate over all edges to remove
        for e1 in edges:
            u1, v1 = e1
            if time.time() - start_time > DEADLINE:
                break

            # Iterate over all non-edges to add
            for e2 in non_edges:
                u2, v2 = e2

                # Skip if e2 is already in edges (should not happen if we maintain non_edges correctly, but just in case)
                if (u2, v2) in edge_c4 or (v2, u2) in edge_c4:
                    continue

                # Calculate delta C4
                # Remove e1, add e2
                # Delta = C4s gained by adding e2 in (G - e1) - C4s lost by removing e1 from G
                # C4s lost by removing e1 = edge_c4[e1]
                # C4s gained by adding e2 in (G - e1):
                # A C4 containing e2 in G-e1 is u2-x-y-v2 where x-y is an edge in G-e1.
                # This is the same as count_c4_through_edge(u2, v2) in G,
                # MINUS the number of such C4s that use edge e1.
                # A C4 u2-x-y-v2 uses e1 if e1 is one of (u2,x), (x,y), (y,v2).
                # Case 1: e1 = (u2, x) -> x = v1 or u1. But x is neighbor of u2.
                #          So if v1 is neighbor of u2, then x=v1 is a candidate.
                #          Then the C4 is u2-v1-y-v2. For this to be a C4, (v1,y) and (y,v2) must be edges.
                #          Also y != u2, v1.
                # Case 2: e1 = (y, v2) -> y = u1 or v1.
                # Case 3: e1 = (x, y) -> x,y are u1,v1 or v1,u1.
                
                # This exact calculation is complex. Let's use a simpler approximation or exact check.
                # Exact check:
                # Count paths u2-x-y-v2 in G that do NOT use e1.
                paths_not_using_e1 = 0
                for x in neighbors[u2]:
                    if x == v2:
                        continue
                    for y in neighbors[v2]:
                        if y == u2 or y == x:
                            continue
                        if y in neighbors[x]:
                            # Check if path u2-x-y-v2 uses e1
                            uses_e1 = False
                            if (u2, x) == e1 or (x, u2) == e1:
                                uses_e1 = True
                            elif (x, y) == e1 or (y, x) == e1:
                                uses_e1 = True
                            elif (y, v2) == e1 or (v2, y) == e1:
                                uses_e1 = True
                            
                            if not uses_e1:
                                paths_not_using_e1 += 1

                # The number of new C4s formed by adding e2 is paths_not_using_e1.
                # Wait, is this correct?
                # A C4 is u2-x-y-v2-u2.
                # The edges are (u2,x), (x,y), (y,v2), (v2,u2).
                # If we add e2=(u2,v2), the other 3 edges must exist in G-e1.
                # So yes, we count triples (x,y) such that (u2,x), (x,y), (y,v2) are in G-e1.
                # This is exactly paths_not_using_e1.
                
                # C4s lost = edge_c4[e1]
                # But wait, edge_c4[e1] counts C4s in G.
                # When we remove e1, we lose all C4s containing e1.
                # So Delta = paths_not_using_e1 - edge_c4[e1]
                
                delta = paths_not_using_e1 - edge_c4[e1]

                if delta < best_delta:
                    best_delta = delta
                    best_swap = (e1, e2)

        if best_swap and best_delta < 0:
            e1, e2 = best_swap
            u1, v1 = e1
            u2, v2 = e2

            # Perform swap
            # Remove e1
            edges.remove(e1)
            neighbors[u1].remove(v1)
            neighbors[v1].remove(u1)
            del edge_c4[(u1, v1)]
            del edge_c4[(v1, u1)]

            # Add e2
            edges.append(e2)
            neighbors[u2].add(v2)
            neighbors[v2].add(u2)

            # Update non_edges list
            non_edges.remove(e2)

            # Update edge_c4 for all edges
            # This is expensive O(E * (deg^2))
            # But E=91, deg~6, so 91*36 ~ 3000 ops. Fine.
            for e in edges:
                a, b = e
                edge_c4[(a, b)] = count_c4_through_edge(a, b)
                edge_c4[(b, a)] = count_c4_through_edge(b, a)

            current_c4 += best_delta
            improved = True

    return [[u, v] for u, v in edges]
# EVOLVE-BLOCK-END
