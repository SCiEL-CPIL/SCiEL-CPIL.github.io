"""R_4(C6) - 4-colouring of K_n without monochromatic C6.
Combines matrix-power C6 counting, backtracking with forward checking,
and constraint-weighted local search.
"""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    # ---------- helpers ----------
    def edge_index(i, j, n):
        """Lexicographic edge index for (i,j), i<j."""
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def count_c6_total(adj_list, n):
        """Count total monochromatic C6s using matrix powers.
        adj_list: list of 4 numpy (n,n) int32 adjacency matrices.
        C6 count for a graph = trace(A^6) / 12 (each C6 counted 12 times).
        """
        total = 0
        for c in range(4):
            A = adj_list[c]
            A2 = A @ A
            A4 = A2 @ A2
            A6 = A4 @ A2
            total += int(np.trace(A6)) // 12
        return total

    def is_valid(adj_list, n):
        """Check no monochromatic C6."""
        for c in range(4):
            A = adj_list[c]
            A2 = A @ A
            A4 = A2 @ A2
            A6 = A4 @ A2
            if np.trace(A6) > 0:
                return False
        return True

    def extract_colors(adj_list, n):
        """Extract color list in lexicographic order."""
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                for c in range(4):
                    if adj_list[c][i, j]:
                        colors.append(c)
                        break
        return colors

    def colors_to_adj(colors, n):
        """Convert flat color list to list of 4 adjacency matrices."""
        adj = [np.zeros((n, n), dtype=np.int32) for _ in range(4)]
        for k, c in enumerate(colors):
            i = k // (n - 1)
            # need to recover i,j from lexicographic index
            # k = i*(n-1) - i*(i-1)//2 + (j-i-1)
            # solve for i: find i such that i*(n-1)-i*(i-1)//2 <= k < (i+1)*(n-1)-(i+1)*i//2
            # simpler: just iterate
            pass
        # Actually let's just build it directly
        adj = [np.zeros((n, n), dtype=np.int32) for _ in range(4)]
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[k]
                adj[c][i, j] = 1
                adj[c][j, i] = 1
                k += 1
        return adj

    # ---------- backtracking with forward checking ----------
    def backtrack_search(n, time_limit):
        """Try to build a valid 4-coloring of K_n via backtracking with forward checking.
        Returns (colors_list, adj_list) or (None, None).
        """
        E = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        adj = [np.zeros((n, n), dtype=np.int32) for _ in range(4)]
        colors = [0] * E
        edge_pos = {}
        for k, (i, j) in enumerate(edges):
            edge_pos[(i, j)] = k
            edge_pos[(j, i)] = k

        def has_path5(u, v, c, adj_c, visited_mask):
            """Check if there's a simple path of length 5 from u to v in color c graph.
            Uses DFS with bitmask. adj_c is the adjacency matrix for color c.
            """
            # BFS/DFS for path of exactly 5 edges from u to v
            # Use iterative deepening or just DFS
            def dfs(node, depth, mask):
                if depth == 5:
                    return node == v
                # neighbors of node in color c, excluding visited
                nbrs = adj_c[node]
                for nxt in range(n):
                    if nbrs[nxt] and not (mask >> nxt) & 1:
                        if dfs(nxt, depth + 1, mask | (1 << nxt)):
                            return True
                return False
            return dfs(u, 0, (1 << u))

        def creates_c6(c, u, v):
            """Check if adding edge (u,v) with color c creates a C6.
            This means there exists a path of length 5 from u to v in the current
            color-c subgraph (without the u-v edge).
            """
            return has_path5(u, v, c, adj[c], 0)

        def backtrack(idx):
            if time.time() > time_limit:
                return False
            if idx == E:
                return True
            i, j = edges[idx]
            # Try each color
            for c in range(4):
                if creates_c6(c, i, j):
                    continue
                # Assign
                adj[c][i, j] = 1
                adj[c][j, i] = 1
                colors[idx] = c
                if backtrack(idx + 1):
                    return True
                # Undo
                adj[c][i, j] = 0
                adj[c][j, i] = 0
            return False

        # Try a few random orderings
        for attempt in range(30):
            if time.time() > time_limit:
                break
            random.shuffle(edges)
            # Reset
            for c in range(4):
                adj[c] = np.zeros((n, n), dtype=np.int32)
            colors = [0] * E
            if backtrack(0):
                return colors, adj
        return None, None

    # ---------- local search (repair) ----------
    def local_search(adj, n, max_iters=50000, time_limit=None):
        """Simulated annealing / hill climbing to reduce C6 count."""
        if time_limit is None:
            time_limit = deadline
        current_cost = count_c6_total(adj, n)
        if current_cost == 0:
            return adj, 0

        best_adj = [a.copy() for a in adj]
        best_cost = current_cost
        temp = 2.0
        cooling = 0.9997
        no_improve = 0
        max_no_improve = 50000

        for it in range(max_iters):
            if time.time() > time_limit:
                break
            if no_improve > max_no_improve:
                break
            if current_cost == 0:
                return adj, 0

            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue

            # Find current color
            old_c = -1
            for c in range(4):
                if adj[c][i, j]:
                    old_c = c
                    break
            if old_c == -1:
                continue

            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue

            # Apply
            adj[old_c][i, j] = 0
            adj[old_c][j, i] = 0
            adj[new_c][i, j] = 1
            adj[new_c][j, i] = 1

            new_cost = count_c6_total(adj, n)
            delta = new_cost - current_cost

            if delta <= 0 or random.random() < np.exp(-delta / max(0.01, temp)):
                current_cost = new_cost
                no_improve = 0
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = [a.copy() for a in adj]
                    if best_cost == 0:
                        return best_adj, 0
            else:
                # Revert
                adj[new_c][i, j] = 0
                adj[new_c][j, i] = 0
                adj[old_c][i, j] = 1
                adj[old_c][j, i] = 1
                no_improve += 1

            temp *= cooling

        return best_adj, best_cost

    # ---------- main search ----------
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]

    # Strategy: for each n from 18 down to 10, try backtracking first (fast prune),
    # then local search with multiple restarts.
    for n in range(18, 9, -1):
        if time.time() > deadline - 5:
            break

        remaining = deadline - 5 - time.time()
        found = False

        # Phase 1: Backtracking with forward checking (fast for small n, may prune well)
        bt_time = min(remaining * 0.3, 30.0)
        if bt_time > 3:
            colors_bt, adj_bt = backtrack_search(n, start_time + time.time() - start_time + bt_time)
            # fix: time_limit should be absolute
            if colors_bt is not None and is_valid(adj_bt, n):
                best_n = n
                best_colors = colors_bt
                found = True
                break

        # Phase 2: Local search with random restarts
        if not found:
            ls_time = deadline - 5 - time.time()
            if ls_time > 2:
                max_restarts = max(5, int(ls_time * 4))
                for attempt in range(max_restarts):
                    if time.time() > deadline - 3:
                        break
                    # Random initialization
                    adj = [np.zeros((n, n), dtype=np.int32) for _ in range(4)]
                    for i in range(n):
                        for j in range(i + 1, n):
                            c = random.randint(0, 3)
                            adj[c][i, j] = 1
                            adj[c][j, i] = 1

                    adj, cost = local_search(adj, n, max_iters=20000, time_limit=deadline - 3)
                    if cost == 0 and is_valid(adj, n):
                        best_n = n
                        best_colors = extract_colors(adj, n)
                        found = True
                        break

        if found:
            break

    # Fallback: K9 cyclic construction
    if best_n < 9:
        best_n = 9
        best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]

    return {'n': best_n, 'colors': best_colors}
