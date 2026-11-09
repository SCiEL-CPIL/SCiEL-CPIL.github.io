"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110.0
    rng = random.Random(12345)

    # ---------- GF(16) arithmetic (Greenwood-Gleason base colouring) ----------
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    def base_color(i, j):
        # 16-vertex Greenwood-Gleason colouring: colours 0,1,2, triangle-free
        if i // 16 == j // 16:
            return log[(i % 16) ^ (j % 16)] % 3
        return 3

    # ---------- initial 32-vertex colouring (two copies + bipartite colour 3) ----------
    n0 = 32
    mat0 = [[-1] * n0 for _ in range(n0)]
    for i in range(n0):
        for j in range(i + 1, n0):
            c = base_color(i, j)
            mat0[i][j] = c
            mat0[j][i] = c

    def matrix_to_colors(mat, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(mat[i][j])
        return colors

    best_n = n0
    best_mat = [row[:] for row in mat0]

    # ---------- Fast triangle-free check using NumPy ----------
    def is_triangle_free(adj, n):
        for c in range(4):
            A = adj[c]
            common = A @ A.T
            if np.any(common * A):
                return False
        return True

    # ---------- Build adjacency matrices from colour list ----------
    def build_adj(colors, n):
        adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i, j] = True
                adj[c][j, i] = True
                idx += 1
        return adj

    # ---------- Local search to extend vertex ----------
    def extend_vertex(adj, n, time_limit, rng):
        """
        Try to extend the coloring to n+1 vertices.
        adj: list of 4 (n,n) boolean numpy arrays.
        Returns: (success, new_adj, assignment)
        """
        new_n = n + 1
        best_cost = float('inf')
        best_assign = None

        for trial in range(50):
            if time.time() > time_limit:
                break

            assign = np.random.randint(0, 4, size=n)

            # Calculate initial cost
            total_cost = 0
            for c in range(4):
                mask = (assign == c)
                if np.sum(mask) > 1:
                    sub = adj[c][np.ix_(mask, mask)]
                    total_cost += np.sum(sub) // 2

            # Local search
            stuck = 0
            for it in range(3000):
                if time.time() > time_limit:
                    break

                if total_cost == 0:
                    best_cost = 0
                    best_assign = assign.copy()
                    break

                # Pick random vertex to flip
                u = rng.randint(0, n - 1)
                old_c = assign[u]

                # Find best new color
                min_new_cost = total_cost
                best_new_c = old_c

                for new_c in range(4):
                    if new_c == old_c:
                        continue

                    # Calculate cost change
                    nbrs_old_in_set = np.sum(adj[old_c][u, :] & (assign == old_c))
                    nbrs_new_in_set = np.sum(adj[new_c][u, :] & (assign == new_c))

                    delta = nbrs_new_in_set - nbrs_old_in_set
                    new_cost = total_cost + delta

                    if new_cost < min_new_cost:
                        min_new_cost = new_cost
                        best_new_c = new_c

                if best_new_c == old_c:
                    stuck += 1
                    if stuck > 50:
                        break
                else:
                    stuck = 0
                    assign[u] = best_new_c
                    total_cost = min_new_cost

            if total_cost < best_cost:
                best_cost = total_cost
                best_assign = assign.copy()

            if best_cost == 0:
                break

        if best_cost == 0 and best_assign is not None:
            # Construct new adj matrices
            new_adj = [np.zeros((new_n, new_n), dtype=bool) for _ in range(4)]
            for c in range(4):
                new_adj[c][:n, :n] = adj[c]
                for i in range(n):
                    if best_assign[i] == c:
                        new_adj[c][i, new_n - 1] = True
                        new_adj[c][new_n - 1, i] = True
            return True, new_adj, best_assign
        else:
            return False, None, None

    # ---------- Main loop: try to add vertices ----------
    current_colors = matrix_to_colors(mat0, n0)
    current_n = n0
    current_adj = build_adj(current_colors, current_n)

    # Verify base
    if not is_triangle_free(current_adj, current_n):
        return {'n': 0, 'colors': []}

    max_target = 51
    while current_n < max_target and time.time() < deadline - 5:
        time_limit = min(deadline - 5, time.time() + 8.0)
        success, new_adj, _ = extend_vertex(current_adj, current_n, time_limit, rng)
        if success:
            # Update current state
            current_n += 1
            current_adj = new_adj

            # Reconstruct colors from adj
            new_colors = []
            for i in range(current_n):
                for j in range(i + 1, current_n):
                    for c in range(4):
                        if current_adj[c][i, j]:
                            new_colors.append(c)
                            break
            current_colors = new_colors

            if current_n > best_n:
                best_n = current_n
                best_mat = [row[:] for row in current_mat] if False else None
                # Store best as colors
                best_colors = new_colors[:]

        else:
            # Failed to add a vertex; break to avoid wasting time
            break

    # Return best result
    if 'best_colors' not in locals() or best_n == n0:
        best_colors = matrix_to_colors(mat0, best_n)
    else:
        best_colors = best_colors

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
