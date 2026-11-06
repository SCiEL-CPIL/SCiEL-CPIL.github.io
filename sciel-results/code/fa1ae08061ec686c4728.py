import time
import random
import sys
sys.setrecursionlimit(100000)

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0

    def edge_idx(i, j, n):
        """Flat index for edge (i,j), i<j."""
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def build_matrix(n, colors):
        m = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                m[i][j] = c
                m[j][i] = c
                idx += 1
        return m

    def has_c6(n, colors):
        """Check if any color class contains a C6. Returns True if invalid."""
        m = build_matrix(n, colors)
        # For each color, build adjacency and check for 6-cycles
        for c in range(4):
            # Adjacency as bitsets
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if m[i][j] == c:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            # Check for C6: for each pair (a,b) with edge, check if there's a path of length 4 from b to a avoiding a
            # Use BFS/DFS with small graph
            # For efficiency, check all 6-subsets
            for a in range(n):
                for b in range(a + 1, n):
                    if not (adj[a] & (1 << b)):
                        continue
                    # Find path b -> x -> y -> z -> w -> a of length 4
                    # x,y,z,w distinct, not in {a,b}
                    # BFS from b, depth 4, target a
                    # visited: bitmask
                    # states: (node, depth, visited_mask)
                    # Since n<=20, we can do iterative deepening or just check all permutations
                    # Better: for each pair (a,b) edge, enumerate paths
                    # Use DFS with pruning
                    stack = [(b, 4, (1 << b))]  # (current, remaining_depth, visited)
                    found = False
                    while stack:
                        node, depth, visited = stack.pop()
                        if depth == 0:
                            if node == a:
                                found = True
                                break
                            continue
                        # Next nodes
                        candidates = adj[node] & ~visited
                        # Don't visit a until the end
                        if depth > 1:
                            candidates &= ~(1 << a)
                        while candidates:
                            lb = candidates & (-candidates)
                            nxt = lb.bit_length() - 1
                            candidates ^= lb
                            stack.append((nxt, depth - 1, visited | lb))
                    if found:
                        return True
        return False

    def backtrack_search(n, time_limit):
        """
        Backtracking search for a valid 4-coloring of K_n with no monochromatic C6.
        Uses constraint propagation to prune.
        """
        deadline = time.time() + time_limit
        num_edges = n * (n - 1) // 2
        colors = [-1] * num_edges

        # Precompute: for each edge (u,v) and color c, which other edges are "dangerous"
        # When we set (u,v)=c, a C6 could form if there exist x,y,z,w such that
        # (u,x),(x,y),(y,z),(z,w),(w,v) all = c, or (v,x),(x,y),(y,z),(z,w),(w,u) all = c.
        # More generally, for any 4-tuple of distinct vertices {x,y,z,w} disjoint from {u,v},
        # if (u,x)=c and (x,y)=c and (y,z)=c and (z,w)=c and (w,v)=c, then C6 u-x-y-z-w-v-u.
        # Similarly for the other direction and other orderings.

        # For efficiency, we'll do a check after each assignment:
        # When (u,v) is set to color c, check if there's a path of length 4 from u to v
        # in the subgraph of color c edges (including the new edge), that together with (u,v)
        # forms a C6.

        def check_new_c6(u, v, c):
            """Check if setting (u,v)=c creates a monochromatic C6. Returns True if C6 created."""
            # Look for a path of length 4 from u to v using color c edges
            # The path is u - x - y - z - w - v where x,y,z,w are distinct and not in {u,v}
            # We need edges (u,x),(x,y),(y,z),(z,w),(w,v) all color c
            # Plus (u,v) is color c, completing the cycle.
            # But we also need to make sure this is a valid C6 (all 6 vertices distinct, which they are).

            # Build adjacency for color c among vertices not in {u,v}
            # Actually, we just need to check specific paths.

            # For efficiency, enumerate possible intermediate vertices
            # x must be adjacent to u in color c
            # w must be adjacent to v in color c
            # y must be adjacent to x in color c
            # z must be adjacent to y in color c
            # And (z,w) must be color c

            # Let's be smart: precompute for each vertex and color, which vertices are connected
            # But colors are being assigned dynamically, so we check on the fly.

            # For each x (adjacent to u in color c):
            #   For each w (adjacent to v in color c):
            #     If x == w: skip (need 4 distinct intermediates)
            #     For each y (adjacent to x in color c, y != u, y != v, y != x, y != w):
            #       For each z (adjacent to y in color c, z != u, z != v, z != x, z != y, z != w):
            #         If (z,w) is color c: C6 found!

            # This is O(n^4) per check, which is expensive. Let's optimize.

            # Alternative: for the new edge (u,v) with color c,
            # check all 4-tuples (x,y,z,w) of distinct vertices not in {u,v}
            # such that (u,x)=c, (x,y)=c, (y,z)=c, (z,w)=c, (w,v)=c.

            # Precompute color-c neighbors for each vertex
            def color_neighbors(vertex, col):
                """Return list of vertices w such that (vertex,w) has color col."""
                result = []
                for w in range(n):
                    if w == vertex:
                        continue
                    i, j = (vertex, w) if vertex < w else (w, vertex)
                    if colors[edge_idx(i, j, n)] == col:
                        result.append(w)
                return result

            # Get color-c neighbors of u and v
            u_neighbors = color_neighbors(u, c)
            v_neighbors = color_neighbors(v, c)

            for x in u_neighbors:
                if x == v:
                    continue
                for w in v_neighbors:
                    if w == u or w == x:
                        continue
                    # Now find path x - y - z - w in color c
                    x_neighbors = color_neighbors(x, c)
                    for y in x_neighbors:
                        if y == u or y == v or y == x or y == w:
                            continue
                        y_neighbors = color_neighbors(y, c)
                        for z in y_neighbors:
                            if z == u or z == v or z == x or z == y or z == w:
                                continue
                            # Check if (z,w) is color c
                            i, j = (z, w) if z < w else (w, z)
                            if colors[edge_idx(i, j, n)] == c:
                                return True
            return False

        # Order edges for backtracking: use a heuristic
        # Process edges in a fixed order, but we can reorder for efficiency
        # Simple: lexicographic order
        edge_order = list(range(num_edges))

        def solve(pos):
            if time.time() > deadline:
                return False
            if pos == num_edges:
                return True

            ei = edge_order[pos]
            # Find u,v for this edge
            # Reverse the edge_idx mapping
            # i * (2*n - i - 1) // 2 + (j - i - 1) = ei
            # We can precompute this
            # For now, iterate
            found_uv = False
            for i in range(n):
                base = i * (2 * n - i - 1) // 2
                if ei < base:
                    break
                if ei >= base:
                    offset = ei - base
                    j = i + offset + 1
                    if j < n:
                        u, v = i, j
                        found_uv = True
                        break
            if not found_uv:
                # Should not happen
                return False

            for c in range(4):
                colors[ei] = c
                if not check_new_c6(u, v, c):
                    if solve(pos + 1):
                        return True
                colors[ei] = -1

            return False

        if solve(0):
            return colors[:]
        return None

    def verify(n, colors):
        """Verify the coloring is valid."""
        if len(colors) != n * (n - 1) // 2:
            return False
        for c in colors:
            if c < 0 or c > 3:
                return False
        return not has_c6(n, colors)

    # Try n=18 with backtracking
    if time.time() < hard_deadline - 10:
        res = backtrack_search(18, 100)
        if res and verify(18, res):
            return {'n': 18, 'colors': res}

    # Try n=17 with backtracking
    if time.time() < hard_deadline - 5:
        res = backtrack_search(17, 5)
        if res and verify(17, res):
            return {'n': 17, 'colors': res}

    # Fallback: try n=16 with quick backtracking
    if time.time() < hard_deadline - 2:
        res = backtrack_search(16, 2)
        if res and verify(16, res):
            return {'n': 16, 'colors': res}

    # Absolute fallback: n=1
    return {'n': 1, 'colors': []}

if __name__ == '__main__':
    result = construct()
    print(f"n={result['n']}, edges={len(result['colors'])}")
