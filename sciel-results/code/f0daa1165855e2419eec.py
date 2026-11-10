"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from itertools import product

def construct():
    """
    Finds a dense triangle-free graph on 16 vertices with chromatic number >= 4.
    Uses structured blow-up of the Grötzsch graph (Mycielski of C5) to maximize edges.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin for 150s limit

    n = 16

    # Grötzsch graph: 11 vertices, 20 edges, triangle-free, chromatic number 4
    # Structure:
    # Vertices 0-4: cycle C5
    # Vertices 5-9: Mycielski middle layer
    # Vertex 10: Mycielski top
    groetzsch_edges = []
    # Cycle C5 on vertices 0-4
    for i in range(5):
        groetzsch_edges.append((i, (i + 1) % 5))
    # Middle layer: vertex 5+i connected to i and (i+1)%5
    for i in range(5):
        groetzsch_edges.append((5 + i, i))
        groetzsch_edges.append((5 + i, (i + 1) % 5))
    # Top vertex 10 connected to all middle layer
    for i in range(5):
        groetzsch_edges.append((10, 5 + i))

    # Verify Grötzsch is triangle-free and 4-chromatic
    # It is known to be triangle-free with chromatic number 4.

    # Now we blow up: replace each of the 11 vertices with an independent set of size 1 or 2.
    # Total vertices = sum of sizes = 16, so we need to distribute 16 across 11 parts.
    # 5 parts of size 2 and 6 parts of size 1: 5*2 + 6*1 = 16.
    # Or other distributions.

    # For each assignment of sizes (each 1 or 2, summing to 16), we create a graph.
    # Edges between parts: if (u,v) is an edge in Grötzsch, then all edges between part_u and part_v are present.
    # This maintains triangle-free because Grötzsch is triangle-free (no triangles in the blow-up if original has none).
    # Chromatic number: blow-up of a 4-chromatic graph with independent sets is still 4-chromatic
    # (can color each part with one of 4 colors, but need 4 because original needs 4).

    # Actually, the blow-up of a k-chromatic graph by independent sets is k-chromatic.
    # So any such blow-up is 4-chromatic.

    # We want to maximize edges. The number of edges in the blow-up is:
    # sum over edges (u,v) in Grötzsch of size(u) * size(v)
    # So we want to assign larger sizes to vertices with higher degree in Grötzsch.

    # Degrees in Grötzsch:
    # Vertices 0-4 (cycle): each has degree 3 (2 cycle neighbors + 1 middle)
    # Wait, let me recompute.
    # Vertex i (0-4): connected to (i-1)%5, (i+1)%5 in cycle, and to 5+i in middle? 
    # Actually from the edges:
    # Vertex i is connected to (i-1)%5 and (i+1)%5 (cycle), and to 5+i? No.
    # Let me trace:
    # Edges involving vertex i (0<=i<=4):
    # - (i, (i+1)%5) and ((i-1)%5, i) from cycle
    # - (5+i, i) from middle layer
    # So vertex i has degree 3.
    #
    # Vertex 5+i: connected to i and (i+1)%5. Degree 2.
    #
    # Vertex 10: connected to 5+0, ..., 5+4. Degree 5.

    # So degrees: vertices 0-4 have degree 3, vertices 5-9 have degree 2, vertex 10 has degree 5.

    # To maximize edges in blow-up, assign size 2 to highest degree vertices.
    # Vertex 10 (degree 5) should be size 2.
    # Vertices 0-4 (degree 3) should be size 2.
    # That's 1 + 5 = 6 vertices of size 2, giving 6*2 + 5*1 = 17. Too many.
    # We need exactly 16. So 5 vertices of size 2 and 6 of size 1.

    # Best: assign size 2 to vertex 10 and four of the cycle vertices (0-4).
    # Or vertex 10 and all five cycle vertices would be 6 of size 2 = 17, too many.
    # So: vertex 10 (size 2) + 4 of {0..4} (size 2) + rest size 1.
    # Or: 5 of {0..4} size 2, vertex 10 size 1.

    # Let's compute edge count for different assignments.
    # Edge count = sum_{(u,v) in E} s(u)*s(v)

    # Let's enumerate all ways to choose which 5 of the 11 vertices get size 2 (rest size 1).
    # There are C(11,5) = 462 possibilities. For each, compute edge count, verify triangle-free
    # (should be automatic), and check 4-chromatic (should be automatic). Then pick max.

    from itertools import combinations

    best_edges = []
    best_count = 0

    # Grötzsch edges as set for quick lookup
    groetzsch_edge_set = set(tuple(sorted(e)) for e in groetzsch_edges)

    # Adjacency for Grötzsch
    groetzsch_adj = [set() for _ in range(11)]
    for u, v in groetzsch_edges:
        groetzsch_adj[u].add(v)
        groetzsch_adj[v].add(u)

    def is_triangle_free_blowup(sizes):
        """Check if the blow-up is triangle-free. Since Grötzsch is triangle-free,
        any blow-up by independent sets is triangle-free. So this is always True."""
        return True

    def is_4_chromatic_blowup(sizes):
        """The blow-up of a 4-chromatic graph by independent sets is 4-chromatic.
        So this is always True for Grötzsch blow-ups."""
        return True

    def build_blowup(sizes):
        """Build the actual graph on 16 vertices from the size assignment."""
        # Assign vertex indices
        vertex_map = []  # vertex_map[original_vertex] = list of new vertex indices
        current_idx = 0
        for i in range(11):
            vertex_map.append(list(range(current_idx, current_idx + sizes[i])))
            current_idx += sizes[i]

        edges = []
        for u in range(11):
            for v in range(u + 1, 11):
                if (u, v) in groetzsch_edge_set:
                    for a in vertex_map[u]:
                        for b in vertex_map[v]:
                            edges.append([a, b])
        return edges

    def count_edges_from_sizes(sizes):
        """Compute number of edges in the blow-up without building it."""
        count = 0
        for u in range(11):
            for v in range(u + 1, 11):
                if (u, v) in groetzsch_edge_set:
                    count += sizes[u] * sizes[v]
        return count

    # Enumerate all assignments of 5 vertices to size 2
    for five_twos in combinations(range(11), 5):
        sizes = [1] * 11
        for i in five_twos:
            sizes[i] = 2

        # Verify sum is 16
        if sum(sizes) != 16:
            continue

        edge_count = count_edges_from_sizes(sizes)
        if edge_count > best_count:
            best_count = edge_count
            best_edges = build_blowup(sizes)

    # Also try: 4 vertices of size 2, 1 vertex of size 3, rest size 1? 
    # 4*2 + 1*3 + 6*1 = 8+3+6 = 17. Too many.
    # 3 vertices of size 2, 2 vertices of size 2... no, we need sum=16.
    # General: let x_i be sizes, sum x_i = 16, x_i >= 1, x_i <= 3 (say).
    # Let y_i = x_i - 1 >= 0, sum y_i = 5.
    # So we distribute 5 units among 11 vertices, each y_i <= 2 (if max size 3).
    # This is more general. Let's do it.

    # More general search: distribute 5 extra units among 11 vertices, each getting 0, 1, or 2 extra.
    # This covers all cases where sizes are 1, 2, or 3.

    def generate_size_assignments():
        """Generate all size assignments where sizes[i] >= 1, sum = 16, sizes[i] <= 3."""
        # Equivalent to: y_i = sizes[i] - 1, y_i >= 0, sum y_i = 5, y_i <= 2
        results = []
        # Use recursion
        def rec(idx, remaining, current):
            if idx == 11:
                if remaining == 0:
                    results.append(current[:])
                return
            if remaining < 0:
                return
            for add in range(min(2, remaining) + 1):
                current.append(1 + add)
                rec(idx + 1, remaining - add, current)
                current.pop()
        rec(0, 5, [])
        return results

    all_assignments = generate_size_assignments()

    for sizes in all_assignments:
        if sum(sizes) != 16:
            continue
        edge_count = count_edges_from_sizes(sizes)
        if edge_count > best_count:
            best_count = edge_count
            best_edges = build_blowup(sizes)

    # Verify the best graph is triangle-free and not 3-colorable
    if best_edges:
        # Check triangle-free
        adj = [set() for _ in range(n)]
        for u, v in best_edges:
            adj[u].add(v)
            adj[v].add(u)

        triangle_free = True
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for w in adj[u]:
                    if w > v and w in adj[v]:
                        triangle_free = False
                        break
                if not triangle_free:
                    break
            if not triangle_free:
                break

        if not triangle_free:
            # Should not happen, but fallback
            return []

        # Check 3-colorability
        def is_3_colorable():
            colors = [-1] * n
            order = sorted(range(n), key=lambda x: -len(adj[x]))

            def rec(idx):
                if idx == n:
                    return True
                v = order[idx]
                used = set()
                for u in adj[v]:
                    if colors[u] != -1:
                        used.add(colors[u])
                for c in range(3):
                    if c not in used:
                        colors[v] = c
                        if rec(idx + 1):
                            return True
                        colors[v] = -1
                return False

            return rec(0)

        if is_3_colorable():
            # Should not happen for Grötzsch blow-up, but fallback
            return []

        return best_edges

    # Fallback: return Grötzsch graph with isolated vertices
    fallback_edges = []
    for u, v in groetzsch_edges:
        fallback_edges.append([u, v])
    return fallback_edges

# EVOLVE-BLOCK-END
