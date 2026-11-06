import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    n = 20
    C = 20

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m = len(all_edges)
    edge_index = {}
    for i, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    # Precompute for each edge (u,v) the list of (a, b) where a~u, b~v,
    # a!=b, a!=v, b!=u, and a~b. These form 4-cycles a-u-v-b-a.
    # When edge (u,v) is added with color c, we need:
    #   c != color(u,a), c != color(v,b), c != color(a,b)
    #   and color(u,a), color(v,b), color(a,b) must already be mutually distinct.
    conflict_pairs = [[] for _ in range(m)]
    for i, (u, v) in enumerate(all_edges):
        for a in range(n):
            if a == u or a == v:
                continue
            if (a, u) not in edge_index and (u, a) not in edge_index:
                continue
            for b in range(n):
                if b == u or b == v or b == a:
                    continue
                if (b, v) not in edge_index and (v, b) not in edge_index:
                    continue
                if (a, b) not in edge_index and (b, a) not in edge_index:
                    continue
                conflict_pairs[i].append((a, b))

    def fast_check(adj, neighbors):
        """Verify all 4-cycles are rainbow using bitsets."""
        for u in range(n):
            for v in neighbors[u]:
                if v <= u:
                    continue
                c_uv = adj[u][v]
                for w in neighbors[v]:
                    if w == u or w <= u:
                        continue
                    c_vw = adj[v][w]
                    if c_vw == c_uv:
                        return False
                    for x in neighbors[w]:
                        if x == v or x <= v:
                            continue
                        if x not in neighbors[u]:
                            continue
                        c_wx = adj[w][x]
                        if c_wx == c_uv or c_wx == c_vw:
                            return False
                        c_xu = adj[x][u]
                        if c_xu == c_uv or c_xu == c_vw or c_xu == c_wx:
                            return False
        return True

    def try_add_edge(u, v, c, adj, neighbors, forbid_count, edge_set):
        """Try to add edge (u,v) with color c. Returns True if successful."""
        if (u, v) in edge_set or (v, u) in edge_set:
            return False
        i = edge_index[(u, v) if u < v else (v, u)]
        # Check feasibility: for each conflict pair (a,b), the three existing colors
        # must be mutually distinct and none equal to c.
        for (a, b) in conflict_pairs[i]:
            ca = adj[u][a]
            cb = adj[v][b]
            cab = adj[a][b]
            if ca == cb or cb == cab or ca == cab:
                return False
            if c == ca or c == cb or c == cab:
                return False
        # All checks passed; add the edge
        adj[u][v] = c
        adj[v][u] = c
        neighbors[u].add(v)
        neighbors[v].add(u)
        edge_set.add((u, v))
        edge_set.add((v, u))
        # Update forbid_count for all other edges
        for (x, y) in all_edges:
            if x == u and y == v:
                continue
            if x == v and y == u:
                continue
            j = edge_index[(x, y) if x < y else (y, x)]
            for (a, b) in conflict_pairs[j]:
                # Check if this conflict pair involves the new edge
                if (a == u and b == v) or (a == v and b == u):
                    forbid_count[j] += 1
                    break
                if (a == u and b == x) or (a == x and b == u):
                    # a~x means a is neighbor of x, but we need a~u for the new edge
                    # Actually conflict_pairs[j] has (a,b) where a~x, b~y
                    # The new edge is (u,v). For (u,v) to be a conflict pair of (x,y),
                    # we need u~x and v~y (or u~y and v~x)
                    pass
        # Simpler: just update forbid_count by checking if new edge creates new conflicts
        # For each edge (x,y) != (u,v), check if (u,v) appears in conflict_pairs of (x,y)
        # i.e., if u~x and v~y (with u!=v, x!=y, etc.)
        for x in neighbors[u]:
            if x == v:
                continue
            for y in neighbors[v]:
                if y == u or y == x:
                    continue
                # Edge (x,y) now has a new conflict pair involving (u,v)
                # Only if (x,y) exists
                if (x, y) in edge_set or (y, x) in edge_set:
                    j = edge_index[(x, y) if x < y else (y, x)]
                    forbid_count[j] += 1
        return True

    def greedy_build(seed, init_edges=None, rng=None):
        """Build a valid edge-colored graph greedily using forbid_count."""
        if rng is None:
            rng = random.Random(seed)
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edge_set = set()
        edges = []

        if init_edges:
            for u, v, c in init_edges:
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edge_set.add((u, v))
                edge_set.add((v, u))
                edges.append([u, v, c])

        # Initialize forbid_count
        forbid_count = [0] * m
        for i, (u, v) in enumerate(all_edges):
            if (u, v) in edge_set:
                continue
            for (a, b) in conflict_pairs[i]:
                ca = adj[u][a]
                cb = adj[v][b]
                cab = adj[a][b]
                if ca == cb or cb == cab or ca == cab:
                    forbid_count[i] += 1

        remaining = [(u, v) for (u, v) in all_edges if (u, v) not in edge_set and (v, u) not in edge_set]
        rng.shuffle(remaining)

        for (u, v) in remaining:
            i = edge_index[(u, v) if u < v else (v, u)]
            if forbid_count[i] > 0:
                continue
            # Find valid colors
            forbidden = set()
            for (a, b) in conflict_pairs[i]:
                forbidden.add(adj[u][a])
                forbidden.add(adj[v][b])
                forbidden.add(adj[a][b])
            candidates = [c for c in range(C) if c not in forbidden]
            if not candidates:
                continue
            c = rng.choice(candidates)
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            edge_set.add((u, v))
            edge_set.add((v, u))
            edges.append([u, v, c])
            # Update forbid_count for affected edges
            for x in neighbors[u]:
                if x == v:
                    continue
                for y in neighbors[v]:
                    if y == u or y == x:
                        continue
                    if (x, y) in edge_set or (y, x) in edge_set:
                        j = edge_index[(x, y) if x < y else (y, x)]
                        forbid_count[j] += 1

        return edges, adj, neighbors, edge_set, forbid_count

    def local_search(edges, adj, neighbors, edge_set, forbid_count, seed, time_limit):
        """Improve by recoloring edges and adding new ones."""
        rng = random.Random(seed)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to add edges
            remaining = [(u, v) for (u, v) in all_edges if (u, v) not in edge_set and (v, u) not in edge_set]
            rng.shuffle(remaining)
            for (u, v) in remaining:
                i = edge_index[(u, v) if u < v else (v, u)]
                if forbid_count[i] > 0:
                    continue
                forbidden = set()
                for (a, b) in conflict_pairs[i]:
                    forbidden.add(adj[u][a])
                    forbidden.add(adj[v][b])
                    forbidden.add(adj[a][b])
                candidates = [c for c in range(C) if c not in forbidden]
                if not candidates:
                    continue
                c = rng.choice(candidates)
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edge_set.add((u, v))
                edge_set.add((v, u))
                edges.append([u, v, c])
                for x in neighbors[u]:
                    if x == v:
                        continue
                    for y in neighbors[v]:
                        if y == u or y == x:
                            continue
                        if (x, y) in edge_set or (y, x) in edge_set:
                            j = edge_index[(x, y) if x < y else (y, x)]
                            forbid_count[j] += 1
                improved = True

            if not improved:
                # Try recoloring to unlock additions
                for idx in range(len(edges)):
                    u, v, old_c = edges[idx]
                    i = edge_index[(u, v) if u < v else (v, u)]
                    # Check if recoloring is possible
                    for new_c in range(C):
                        if new_c == old_c:
                            continue
                        valid = True
                        for (a, b) in conflict_pairs[i]:
                            ca = adj[u][a]
                            cb = adj[v][b]
                            cab = adj[a][b]
                            if ca == cb or cb == cab or ca == cab:
                                valid = False
                                break
                            if new_c == ca or new_c == cb or new_c == cab:
                                valid = False
                                break
                        if not valid:
                            continue
                        # Try recoloring and see if it unlocks an edge
                        adj[u][v] = new_c
                        adj[v][u] = new_c
                        edges[idx] = [u, v, new_c]
                        # Update forbid_count
                        for x in neighbors[u]:
                            if x == v:
                                continue
                            for y in neighbors[v]:
                                if y == u or y == x:
                                    continue
                                if (x, y) in edge_set or (y, x) in edge_set:
                                    j = edge_index[(x, y) if x < y else (y, x)]
                                    # Remove old forbid, add new
                                    pass
                        # Try to add one edge
                        remaining2 = [(wu, wv) for (wu, wv) in all_edges if (wu, wv) not in edge_set and (wv, wu) not in edge_set]
                        rng.shuffle(remaining2)
                        added = False
                        for (wu, wv) in remaining2:
                            j = edge_index[(wu, wv) if wu < wv else (wv, wu)]
                            if forbid_count[j] > 0:
                                continue
                            forbidden2 = set()
                            for (a, b) in conflict_pairs[j]:
                                forbidden2.add(adj[wu][a])
                                forbidden2.add(adj[wv][b])
                                forbidden2.add(adj[a][b])
                            candidates2 = [c for c in range(C) if c not in forbidden2]
                            if not candidates2:
                                continue
                            nc = rng.choice(candidates2)
                            adj[wu][wv] = nc
                            adj[wv][wu] = nc
                            neighbors[wu].add(wv)
                            neighbors[wv].add(wu)
                            edge_set.add((wu, wv))
                            edge_set.add((wv, wu))
                            edges.append([wu, wv, nc])
                            for x in neighbors[wu]:
                                if x == wv:
                                    continue
                                for y in neighbors[wv]:
                                    if y == wu or y == x:
                                        continue
                                    if (x, y) in edge_set or (y, x) in edge_set:
                                        k = edge_index[(x, y) if x < y else (y, x)]
                                        forbid_count[k] += 1
                            added = True
                            break
                        if added:
                            improved = True
                            break
                        else:
                            # Revert recoloring
                            adj[u][v] = old_c
                            adj[v][u] = old_c
                            edges[idx] = [u, v, old_c]
                    if improved:
                        break

        return edges

    best_count = 0
    best_edges = []

    # Phase 1: Bipartite seeding + greedy
    # Use a random bipartition and seed with a complete bipartite subgraph
    for trial in range(50):
        if time.time() > deadline - 60:
            break
        rng = random.Random(trial * 12345 + 678)
        # Random bipartition
        part_a = set(rng.sample(range(n), n // 2))
        part_b = set(range(n)) - part_a
        # Seed with some bipartite edges
        init_edges = []
        a_list = list(part_a)
        b_list = list(part_b)
        for a in a_list:
            for b in b_list:
                if rng.random() < 0.3:
                    init_edges.append([a, b, rng.randrange(C)])
        # Check if init is valid (should be since bipartite has no 4-cycles within it... actually it does)
        # Just try greedy build with this init
        edges, adj, neighbors, edge_set, forbid_count = greedy_build(trial, init_edges, rng)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]

    # Phase 2: Plain greedy with many restarts
    for i in range(200):
        if time.time() > deadline - 40:
            break
        edges, adj, neighbors, edge_set, forbid_count = greedy_build(i * 999 + 1)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]

    # Phase 3: Local search on best
    if best_edges and time.time() < deadline - 20:
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edge_set = set()
        for u, v, c in best_edges:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            edge_set.add((u, v))
            edge_set.add((v, u))
        forbid_count = [0] * m
        for i, (u, v) in enumerate(all_edges):
            if (u, v) in edge_set:
                continue
            for (a, b) in conflict_pairs[i]:
                ca = adj[u][a]
                cb = adj[v][b]
                cab = adj[a][b]
                if ca == cb or cb == cab or ca == cab:
                    forbid_count[i] += 1
        improved_edges = local_search(best_edges[:], adj, neighbors, edge_set, forbid_count, 42, deadline - 10)
        if len(improved_edges) > best_count:
            best_count = len(improved_edges)
            best_edges = improved_edges[:]

    # Final verification
    if best_edges:
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        for u, v, c in best_edges:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
        if not fast_check(adj, neighbors):
            best_edges = []
            for i in range(50):
                candidate, adj2, neighbors2, edge_set2, forbid_count2 = greedy_build(5000 + i)
                if fast_check(adj2, neighbors2):
                    best_edges = candidate
                    break
            if not best_edges:
                best_edges = [[0, v, v % C] for v in range(1, n)]

    if not best_edges:
        best_edges = [[0, v, v % C] for v in range(1, n)]

    return best_edges
