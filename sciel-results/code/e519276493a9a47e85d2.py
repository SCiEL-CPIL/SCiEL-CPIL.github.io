import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Try to find a valid 5-coloring of K_n with no monochromatic C4.
    # We start from n=26 (known feasible via Lazebnik-Woldar) and work down.
    # Use local search with efficient delta evaluation.

    best_result = None

    for n in range(26, 10, -1):
        if elapsed() > deadline - 5:
            break

        # Initialize colors randomly or with a structured heuristic
        # Each edge (i,j) with i<j gets a color in {0,1,2,3,4}
        num_edges = n * (n - 1) // 2

        # Precompute edge index for O(1) lookup
        edge_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx[(i, j)] = idx
                idx += 1

        def get_edge_color(colors, i, j):
            if i > j:
                i, j = j, i
            return colors[edge_idx[(i, j)]]

        def set_edge_color(colors, i, j, c):
            if i > j:
                i, j = j, i
            colors[edge_idx[(i, j)]] = c

        def count_violations(colors):
            """Count number of monochromatic C4s. A C4 is four distinct vertices a,b,c,d
            such that edges ab, bc, cd, da all have the same color.
            We count each C4 once per color by checking all pairs of opposite vertices.
            Actually, a simpler approach: for each color c, for each pair of vertices (u,v),
            count how many common neighbors w have both (u,w) and (v,w) colored c.
            If this count >= 2, we have a violation. But we need to count distinct C4s.
            
            More efficient: for each color c, build adjacency. For each pair (u,v) not adjacent in color c,
            count common neighbors in color c. If >= 2, that's a C4.
            
            Let's use: for each color c, for each pair of vertices (u,v), let common_c(u,v) = number of w
            such that (u,w) and (v,w) are both color c. A C4 exists iff for some color c and some pair (u,v),
            common_c(u,v) >= 2. The number of violations is sum over all (u,v) and c of max(0, common_c(u,v)-1).
            """
            violations = 0
            for c in range(5):
                # For each pair (u,v), count common neighbors in color c
                for u in range(n):
                    for v in range(u + 1, n):
                        common = 0
                        for w in range(n):
                            if w == u or w == v:
                                continue
                            if get_edge_color(colors, u, w) == c and get_edge_color(colors, v, w) == c:
                                common += 1
                        if common >= 2:
                            violations += common - 1
            return violations

        def count_violations_fast(colors):
            """Faster violation count using per-color neighbor sets."""
            violations = 0
            for c in range(5):
                # Build neighbor sets for color c
                neighbors = [set() for _ in range(n)]
                for i in range(n):
                    for j in range(i + 1, n):
                        if get_edge_color(colors, i, j) == c:
                            neighbors[i].add(j)
                            neighbors[j].add(i)
                # For each pair (u,v), count common neighbors
                for u in range(n):
                    for v in range(u + 1, n):
                        common = len(neighbors[u] & neighbors[v])
                        if common >= 2:
                            violations += common - 1
            return violations

        # Local search
        def local_search(n, max_iter=50000):
            colors = np.random.randint(0, 5, size=num_edges)
            current_violations = count_violations_fast(colors)

            if current_violations == 0:
                return colors

            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break

                if current_violations == 0:
                    return colors

                # Pick a random edge and try to change its color
                edge_i = random.randint(0, num_edges - 1)
                # Find which (i,j) this edge is
                # This is slow; let's precompute reverse mapping
                # Instead, pick random (i,j) directly
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i

                old_color = colors[edge_idx[(i, j)]]
                new_color = random.randint(0, 4)
                if new_color == old_color:
                    continue

                # Compute delta in violations
                colors[edge_idx[(i, j)]] = new_color
                new_violations = count_violations_fast(colors)
                if new_violations < current_violations:
                    current_violations = new_violations
                else:
                    colors[edge_idx[(i, j)]] = old_color

            return colors

        # Try multiple restarts
        found = False
        for restart in range(5):
            if time.time() > deadline - 5:
                break

            result = local_search(n, max_iter=30000)
            v = count_violations_fast(result)
            if v == 0:
                # Convert to list format
                color_list = []
                for i in range(n):
                    for j in range(i + 1, n):
                        color_list.append(int(result[edge_idx[(i, j)]]))
                best_result = {'n': n, 'colors': color_list}
                found = True
                break

        if found:
            break

    if best_result is None:
        # Fallback: use the seed solution for n=11
        n = 11
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        best_result = {'n': n, 'colors': colors}

    return best_result
