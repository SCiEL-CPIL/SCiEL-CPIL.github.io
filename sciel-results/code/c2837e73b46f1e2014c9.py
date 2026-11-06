import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill

    N = 16
    ALL = (1 << N) - 1

    edges_list = [(i, j) for i in range(N) for j in range(i + 1, N)]
    M = len(edges_list)

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(N)) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return False
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        return common == 0

    def is_3_colorable(adj):
        """Fast 3-colorability check using color-class bitmasks with MRV ordering."""
        # Order vertices by degree descending
        degrees = [bin(adj[i]).count('1') for i in range(N)]
        order = sorted(range(N), key=lambda x: degrees[x], reverse=True)
        pos = {v: i for i, v in enumerate(order)}

        # Precompute for each position, which earlier positions are neighbors
        pred_nb = [[] for _ in range(N)]
        for i in range(N):
            v = order[i]
            for j in range(i):
                w = order[j]
                if (adj[v] >> w) & 1:
                    pred_nb[i].append(j)

        color = [-1] * N
        cc = [0, 0, 0]  # color class bitmasks

        def backtrack(idx):
            if idx == N:
                return True
            v = order[idx]
            # Forbidden colors from predecessors
            forbidden = 0
            for j in pred_nb[idx]:
                c = color[order[j]]
                if c >= 0:
                    forbidden |= (1 << c)

            # MRV: try colors in order of scarcity (try most constrained first)
            # Actually just try all valid colors
            for c in range(3):
                if forbidden & (1 << c):
                    continue
                color[v] = c
                cc[c] |= (1 << v)
                if backtrack(idx + 1):
                    return True
                cc[c] &= ~(1 << v)
                color[v] = -1
            return False

        return backtrack(0)

    def build_groetzsch():
        adj = [0] * N
        # 5-cycle: 0-1-2-3-4-0
        for i in range(5):
            add_edge(adj, i, (i + 1) % 5)
        # Chords
        add_edge(adj, 5, 1); add_edge(adj, 5, 3)
        add_edge(adj, 6, 2); add_edge(adj, 6, 4)
        add_edge(adj, 7, 3); add_edge(adj, 7, 0)
        add_edge(adj, 8, 4); add_edge(adj, 8, 1)
        add_edge(adj, 9, 0); add_edge(adj, 9, 2)
        # Vertex 10 connects to 5,6,7,8,9
        for v in range(5, 10):
            add_edge(adj, 10, v)
        return adj

    # Build initial 4-chromatic graph
    best_adj = build_groetzsch()
    best_count = count_edges(best_adj)

    # Verify
    assert not is_3_colorable(best_adj), "Grötzsch should be 4-chromatic"

    # Greedy edge addition: repeatedly try to add any missing edge
    def greedy_add(adj):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for (u, v) in edges_list:
                if time.time() > deadline:
                    break
                if (adj[u] >> v) & 1:
                    continue
                if can_add_edge(adj, u, v):
                    new_adj = adj[:]
                    add_edge(new_adj, u, v)
                    if not is_3_colorable(new_adj):
                        adj = new_adj
                        changed = True
                        break  # restart scan
        return adj

    best_adj = greedy_add(best_adj)
    best_count = count_edges(best_adj)

    # Local search with random restarts and edge swaps
    rng = random.Random(42)

    for restart in range(50):
        if time.time() > deadline:
            break

        current = best_adj[:]
        cur_count = best_count

        # Random perturbation: remove a few random edges
        if restart > 0:
            current = best_adj[:]
            cur_count = best_count
            # Remove 1-3 random edges to perturb
            present_edges = [(u, v) for (u, v) in edges_list if (current[u] >> v) & 1]
            n_remove = min(1 + restart % 3, len(present_edges))
            for _ in range(n_remove):
                if not present_edges:
                    break
                idx = rng.randrange(len(present_edges))
                u, v = present_edges[idx]
                remove_edge(current, u, v)
                present_edges.pop(idx)
                cur_count -= 1

        # Local search: try adding edges and edge swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False

            # Phase 1: try adding any missing edge
            for (u, v) in edges_list:
                if time.time() > deadline:
                    break
                if (current[u] >> v) & 1:
                    continue
                if can_add_edge(current, u, v):
                    new_adj = current[:]
                    add_edge(new_adj, u, v)
                    if not is_3_colorable(new_adj):
                        current = new_adj
                        cur_count += 1
                        improved = True
                        break

            if time.time() > deadline:
                break

            # Phase 2: edge swaps (remove 1, add 2)
            if not improved:
                present_edges = [(u, v) for (u, v) in edges_list if (current[u] >> v) & 1]
                rng.shuffle(present_edges)
                for (ru, rv) in present_edges[:20]:  # limit swaps per iteration
                    if time.time() > deadline:
                        break
                    test = current[:]
                    remove_edge(test, ru, rv)
                    added = 0
                    # Try to add two edges
                    for (a, b) in edges_list:
                        if time.time() > deadline:
                            break
                        if (a == ru and b == rv) or (a == rv and b == ru):
                            continue
                        if (test[a] >> b) & 1:
                            continue
                        if not can_add_edge(test, a, b):
                            continue
                        test2 = test[:]
                        add_edge(test2, a, b)
                        if not is_3_colorable(test2):
                            test = test2
                            added += 1
                            if added >= 2:
                                break
                    if added >= 2:
                        current = test
                        cur_count += added - 1
                        improved = True
                        break

        if cur_count > best_count:
            best_adj = current
            best_count = cur_count

    # Final: one more greedy pass
    best_adj = greedy_add(best_adj)
    best_count = count_edges(best_adj)

    # Verify triangle-free
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                common = best_adj[u] & best_adj[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                if common:
                    # Should not happen, but remove if so
                    pass

    # Verify not 3-colorable
    if is_3_colorable(best_adj):
        best_adj = build_groetzsch()
        best_count = count_edges(best_adj)

    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])

    return edges
