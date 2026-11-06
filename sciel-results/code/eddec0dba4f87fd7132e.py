"""Hadamard-based 4-coloring of K_n for R(3,3,3,3) > n.
Combines K16 Greenwood-Gleason base doubled to K32 with systematic extension."""

import time
import random
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill

    # --- Build base K16 Greenwood-Gleason colouring ---
    # GF(16) with polynomial x^4 + x + 1 (0b10011)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    # Base K16: colour of edge (i,j) = log[i^j] % 3
    base16 = {}
    for i in range(16):
        for j in range(i + 1, 16):
            base16[(i, j)] = log[i ^ j] % 3

    # --- Build K32: two copies of K16, cross edges all colour 3 ---
    n32 = 32
    adj = [[0] * n32 for _ in range(n32)]
    for i in range(n32):
        for j in range(i + 1, n32):
            c = base16[(i % 16, j % 16)] if i // 16 == j // 16 else 3
            adj[i][j] = c
            adj[j][i] = c

    # Precompute adjacency as numpy for fast verification
    adj_np = np.array(adj, dtype=np.int8)

    # Precompute per-color adjacency masks
    color_masks = [adj_np == c for c in range(4)]

    # Precompute independent set property: for each color c and each pair (a,b),
    # whether edge (a,b) has color c
    # For fast constraint checking when adding a vertex

    def is_valid_coloring(n, colors):
        """Fast verification using numpy."""
        adj_v = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_v[i, j] = c
                adj_v[j, i] = c
                idx += 1
        for c in range(4):
            mask = (adj_v == c)
            # Triangle exists if there are i<j<k with all three edges color c
            # Check: for each i, neighbors of i in color c must form independent set
            for i in range(n):
                nbrs = np.where(mask[i, i+1:])[0] + i + 1
                if len(nbrs) < 2:
                    continue
                # Check if any pair of neighbors has color-c edge
                sub = mask[np.ix_(nbrs, nbrs)]
                if np.any(sub):
                    return False
        return True

    # --- Extension machinery ---
    # For a new vertex v, color_map[i] = color of edge (v, i)
    # Constraint: for each color c, the set {i : color_map[i] == c} must be
    # an independent set in the color-c subgraph of the existing coloring.
    # i.e., no two vertices in that set can have a color-c edge between them.

    def can_add_vertex_fast(adj_np, n, color_map):
        """Check if color_map assignment for a new vertex is valid."""
        groups = {0: [], 1: [], 2: [], 3: []}
        for i in range(n):
            groups[color_map[i]].append(i)
        for c in range(4):
            grp = groups[c]
            for a_idx in range(len(grp)):
                a = grp[a_idx]
                for b_idx in range(a_idx + 1, len(grp)):
                    b = grp[b_idx]
                    if adj_np[a, b] == c:
                        return False
        return True

    def try_extend(adj_np, n, max_new, budget_s):
        """Try to add up to max_new vertices using randomized search."""
        current_adj = adj_np.copy()
        current_n = n
        start_time = time.time()

        for _ in range(max_new):
            if time.time() > start_time + budget_s:
                break
            found = False
            # Try random assignments with increasing attempts
            for attempt in range(50000):
                if time.time() > start_time + budget_s:
                    break
                color_map = [random.randint(0, 3) for _ in range(current_n)]
                if can_add_vertex_fast(current_adj, current_n, color_map):
                    # Add vertex
                    new_n = current_n + 1
                    new_adj = np.zeros((new_n, new_n), dtype=np.int8)
                    new_adj[:current_n, :current_n] = current_adj
                    for i in range(current_n):
                        new_adj[i, new_n-1] = color_map[i]
                        new_adj[new_n-1, i] = color_map[i]
                    current_adj = new_adj
                    current_n = new_n
                    found = True
                    break
            if not found:
                break

        if current_n > n:
            return current_n, current_adj
        return None

    # --- Try to extend K32 ---
    extended = None
    if time.time() < deadline - 10.0:
        extended = try_extend(adj_np, 32, max_new=1, budget_s=120.0)

    # --- Build final result ---
    if extended is not None:
        n_final, adj_final = extended
        colors = []
        for i in range(n_final):
            for j in range(i + 1, n_final):
                colors.append(int(adj_final[i, j]))
        return {'n': n_final, 'colors': colors}
    else:
        # Return K32
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(int(adj_np[i, j]))
        return {'n': 32, 'colors': colors}
