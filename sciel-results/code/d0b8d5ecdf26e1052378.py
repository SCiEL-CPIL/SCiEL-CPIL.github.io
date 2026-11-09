import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # GF(4) arithmetic: represent as 2-bit integers
    # GF(4) = {0, 1, 2, 3} with 2=alpha, 3=alpha+1
    gf4_mult = {
        (0,0):0, (0,1):0, (0,2):0, (0,3):0,
        (1,0):0, (1,1):1, (1,2):2, (1,3):3,
        (2,0):0, (2,1):2, (2,2):3, (2,3):2,
        (3,0):0, (3,1):3, (3,2):2, (3,3):1,
    }
    gf4_inv = {1: 1, 2: 3, 3: 2}

    def gf4_add(a, b):
        return a ^ b

    def gf4_mul(a, b):
        return gf4_mult[(a, b)]

    def gf4_div(a, b):
        return gf4_mul(a, gf4_inv[b])

    # Construction for n=16: vertices are elements of GF(4)^2
    n = 16
    colors = []
    for i in range(n):
        ai, bi = i // 4, i % 4
        for j in range(i + 1, n):
            aj, bj = j // 4, j % 4
            d1 = gf4_add(ai, aj)
            d2 = gf4_add(bi, bj)

            if d1 == 0 and d2 == 0:
                c = 0
            elif d1 == 0:
                c = 0  # horizontal
            elif d2 == 0:
                c = 1  # vertical
            else:
                slope = gf4_div(d2, d1)
                if slope == 1:
                    c = 2
                elif slope == 2:
                    c = 3
                else:  # slope == 3
                    c = 4
            colors.append(c)

    # Verify C4-freeness
    def has_c4(color_val, n, colors):
        """Check if any color class contains a C4."""
        adj = [set() for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == color_val:
                    adj[i].add(j)
                    adj[j].add(i)
                idx += 1

        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if len(common) >= 2:
                    return True
        return False

    def verify_valid(n, colors):
        for c in range(5):
            if has_c4(c, n, colors):
                return False
        return True

    # Fallback construction for n=11
    def construct_fallback():
        n = 11
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        return n, colors

    # First, verify the GF(4) construction
    if verify_valid(n, colors):
        return {'n': n, 'colors': colors}

    # If GF(4) construction fails, try greedy randomized construction for larger n
    def greedy_randomized(target_n, max_restarts=50):
        """
        Greedy randomized construction that sequentially assigns colors to edges
        while maintaining a dynamic count of forbidden pairs.
        """
        import random

        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            random.seed(restart)
            n = target_n
            color_of = {}  # (i,j) -> color, i<j
            # For each color c, track for each pair (u,v) the number of common neighbors in color c
            # common_count[c][u][v] = number of vertices w such that (u,w) and (v,w) are both color c
            # We need this to be < 2 for all pairs

            # Precompute all edges in order
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    edges.append((i, j))

            # Shuffle edge order for randomization
            edge_order = list(range(len(edges)))
            random.shuffle(edge_order)

            # common_count[c][u] is a set of vertices v such that (u,v) has common neighbor count >= 1 in color c
            # Actually, we need: for each color c, for each pair (u,v), count of common neighbors
            # This is expensive. Let's use a different approach.

            # For each color c, maintain adj[c][v] = set of neighbors of v in color c
            # When adding edge (u,v) with color c, we need to check that
            # len(adj[c][u] & adj[c][v]) < 2
            # i.e., u and v don't already have 2 common neighbors in color c

            adj = [set() for _ in range(5)]  # adj[c][v] = set of neighbors of v in color c
            # Actually, let's use: adj[c] = list of sets, adj[c][v] = set of neighbors of v in color c
            adj = [[set() for _ in range(n)] for _ in range(5)]

            valid = True
            for idx in edge_order:
                if time.time() > deadline - 5:
                    valid = False
                    break

                i, j = edges[idx]
                # Try each color, pick the one that minimizes the increase in "risk"
                best_color = -1
                best_risk = float('inf')

                for c in range(5):
                    # Check if adding edge (i,j) in color c creates a C4
                    common = adj[c][i] & adj[c][j]
                    if len(common) >= 2:
                        # This color is forbidden
                        continue
                    # Risk: number of common neighbors (closer to 2 is worse)
                    risk = len(common)
                    # Also consider: how many vertices w are adjacent to both i and j in color c
                    # That's exactly len(common)
                    if risk < best_risk:
                        best_risk = risk
                        best_color = c

                if best_color == -1:
                    valid = False
                    break

                c = best_color
                color_of[(i, j)] = c
                adj[c][i].add(j)
                adj[c][j].add(i)

            if not valid:
                continue

            # Build the colors list in lexicographic order
            colors_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors_list.append(color_of[(i, j)])

            # Verify
            if verify_valid(n, colors_list):
                return n, colors_list

        return None

    # Try greedy randomized for n=25, 24, 23, ... down to 17
    for target_n in [25, 24, 23, 22, 21, 20, 19, 18, 17]:
        if time.time() > deadline - 10:
            break
        result = greedy_randomized(target_n, max_restarts=30)
        if result is not None:
            n_found, colors_found = result
            return {'n': n_found, 'colors': colors_found}

    # Final fallback
    n_fb, colors_fb = construct_fallback()
    return {'n': n_fb, 'colors': colors_fb}

# EVOLVE-BLOCK-END
