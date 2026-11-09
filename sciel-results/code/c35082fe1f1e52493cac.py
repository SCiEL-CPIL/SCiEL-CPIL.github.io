import numpy as np
import time
import random

def count_edges(adj):
    return np.sum(adj != -1) // 2

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    random.seed(42)
    np.random.seed(42)

    # Precompute 4-cycle witnesses for each edge (u,v).
    # witness[(u,v)] = list of (x, y) such that u-v-x-y-u is a 4-cycle.
    # This removes the inner double-loop overhead from get_forbidden /
    # is_recolor_valid, which is the main speedup enabling more search.
    witness = {}
    for u in range(n):
        for v in range(u + 1, n):
            wl = []
            for x in range(n):
                if x == u or x == v:
                    continue
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    wl.append((x, y))
            witness[(u, v)] = wl
            witness[(v, u)] = wl

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_colors = list(range(n))

    def get_forbidden(u, v, adj):
        """Set of colors forbidden for edge (u,v).
        If the 3 other edges in any 4-cycle have duplicate colors, return all colors.
        """
        forbidden = set()
        for (x, y) in witness[(u, v)]:
            if adj[v, x] == -1:
                continue
            if adj[x, y] == -1:
                continue
            if adj[y, u] == -1:
                continue

            c1 = adj[v, x]
            c2 = adj[x, y]
            c3 = adj[y, u]

            if c1 == c2 or c1 == c3 or c2 == c3:
                return set(range(n))

            forbidden.add(c1)
            forbidden.add(c2)
            forbidden.add(c3)

        return forbidden

    def is_recolor_valid(u, v, new_c, adj):
        """Check if recoloring edge (u,v) to new_c keeps all 4-cycles rainbow."""
        for (x, y) in witness[(u, v)]:
            if adj[v, x] == -1:
                continue
            if adj[x, y] == -1:
                continue
            if adj[y, u] == -1:
                continue

            c1 = adj[v, x]
            c2 = adj[x, y]
            c3 = adj[y, u]

            if new_c == c1 or new_c == c2 or new_c == c3:
                return False
        return True

    def try_add_missing(adj):
        """Try to add as many missing edges as possible. Returns number added."""
        added = 0
        missing = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == -1]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            forb = get_forbidden(u, v, adj)
            if len(forb) < n:
                available = [c for c in all_colors if c not in forb]
                c = random.choice(available)
                adj[u, v] = c
                adj[v, u] = c
                added += 1
        return added

    def try_recolor(adj, limit):
        """Try to recolor up to `limit` existing edges. Returns True if any recolor happened."""
        existing = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] != -1]
        random.shuffle(existing)
        for u, v in existing[:limit]:
            if time.time() > deadline:
                break
            old_c = adj[u, v]
            colors = all_colors[:]
            random.shuffle(colors)
            for new_c in colors:
                if new_c == old_c:
                    continue
                if is_recolor_valid(u, v, new_c, adj):
                    adj[u, v] = new_c
                    adj[v, u] = new_c
                    return True
        return False

    best_count = 0
    best_adj = None

    for restart in range(100):
        if time.time() > deadline:
            break

        # Initialize with a random spanning tree
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)

        vertices = list(range(n))
        random.shuffle(vertices)
        for i in range(1, n):
            u = vertices[i]
            v = vertices[random.randint(0, i - 1)]
            c = random.randint(0, n - 1)
            adj[u, v] = c
            adj[v, u] = c

        # Greedy edge addition with forbidden-color check
        edges = all_edges[:]
        random.shuffle(edges)
        for u, v in edges:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            forb = get_forbidden(u, v, adj)
            if len(forb) < n:
                available = [c for c in all_colors if c not in forb]
                c = random.choice(available)
                adj[u, v] = c
                adj[v, u] = c

        # Local search: iteratively add missing edges and recolor
        for iteration in range(800):
            if time.time() > deadline:
                break

            improved = False

            # Try adding missing edges
            added = try_add_missing(adj)
            if added > 0:
                improved = True

            if not improved:
                # Try recoloring existing edges to unblock additions
                if try_recolor(adj, 30):
                    improved = True

            if not improved:
                break

        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj.copy()

        if best_count > 160:
            break

    # Final polish: spend remaining time aggressively improving the best solution.
    # The witness-table speedup frees time here for more add/recolor sweeps.
    if best_adj is not None and time.time() < deadline - 5:
        adj = best_adj.copy()
        for polish_iter in range(500):
            if time.time() > deadline - 2:
                break

            improved = False

            added = try_add_missing(adj)
            if added > 0:
                improved = True

            if not improved:
                if try_recolor(adj, 50):
                    improved = True

            if not improved:
                break

        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj.copy()

    # Convert to result
    res = []
    if best_adj is None:
        for v in range(1, n):
            res.append([0, v, 0])
        return res

    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j] != -1:
                res.append([i, j, int(best_adj[i, j])])
    return res
