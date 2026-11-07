"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        edge_set.add((v, u))

    degree = [len(adj[v]) for v in range(n)]

    # ---- Bron-Kerbosch to find maximum clique ----
    max_clique = []

    def bronk(R, P_set, X_set):
        nonlocal max_clique
        if time.time() > deadline:
            return
        if not P_set and not X_set:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        U = P_set | X_set
        if not U:
            return
        pivot = None
        max_inter = -1
        for u in U:
            inter = len(P_set & adj[u])
            if inter > max_inter:
                max_inter = inter
                pivot = u
        candidates = list(P_set - adj[pivot])
        for v in candidates:
            bronk(R | {v}, P_set & adj[v], X_set & adj[v])
            P_set.discard(v)
            X_set.add(v)
            if time.time() > deadline:
                return

    all_vertices = set(range(n))
    bronk(set(), all_vertices, set())

    if len(max_clique) < 2:
        max_clique = [edges[0][0], edges[0][1]]

    # ---- Path finding with BFS, avoiding forbidden + used interior ----
    def find_shortest_path(src, dst, forbidden, used):
        if src == dst:
            return [src]
        blocked = forbidden | used
        visited = {src}
        queue = deque([(src, [src])])
        while queue:
            node, path = queue.popleft()
            if node == dst:
                return path
            for nb in adj[node]:
                if nb in blocked or nb in visited:
                    continue
                visited.add(nb)
                queue.append((nb, path + [nb]))
        return None

    def try_route(branches, order_hint=None):
        """Try to find vertex-disjoint interior paths for all pairs of branches.
        Returns list of paths (in combinations order) or None."""
        k = len(branches)
        if k < 2:
            return None
        pairs = list(combinations(range(k), 2))
        branch_set = set(branches)
        used_interior = set()
        paths = {}

        # Build difficulty-sorted list of pairs
        def diff(i, j):
            bi, bj = branches[i], branches[j]
            if (bi, bj) in edge_set:
                return 0  # direct edge, no interior needed
            common = (adj[bi] & adj[bj]) - branch_set - used_interior
            return len(common)

        remaining = list(pairs)
        max_rounds = 200
        for _ in range(max_rounds):
            if time.time() > deadline - 3:
                return None
            if not remaining:
                break
            # Sort by difficulty (fewest options first)
            remaining.sort(key=lambda p: diff(p[0], p[1]))
            i, j = remaining[0]
            bi, bj = branches[i], branches[j]
            forbidden = branch_set - {bi, bj}

            if (bi, bj) in edge_set:
                # Direct edge: path is just [bi, bj], no interior
                paths[(i, j)] = [bi, bj]
                remaining.remove((i, j))
                continue

            path = find_shortest_path(bi, bj, forbidden, used_interior)
            if path is not None:
                interior = set(path[1:-1])
                if not (interior & used_interior):
                    paths[(i, j)] = path
                    used_interior.update(interior)
                    remaining.remove((i, j))
                else:
                    remaining.pop(0)
            else:
                remaining.pop(0)

        if len(paths) != len(pairs):
            return None
        result = []
        for i, j in pairs:
            result.append(list(paths[(i, j)]))
        return result

    # ---- Start from max clique, try to extend greedily ----
    current = list(max_clique)
    best_branches = list(current)
    best_paths = try_route(current)
    if best_paths is None:
        # Fallback: just use the clique with direct edges
        best_paths = []
        for u, v in combinations(current, 2):
            best_paths.append([u, v])
    best_size = len(current)

    # Candidate pool: high-degree vertices not in current clique
    candidate_pool = sorted(range(n), key=lambda v: degree[v], reverse=True)

    # Greedy extension: try adding vertices one at a time
    extended = True
    while extended:
        extended = False
        if time.time() > deadline - 10:
            break
        for v in candidate_pool:
            if v in current:
                continue
            if time.time() > deadline - 5:
                break
            new_branches = current + [v]
            paths = try_route(new_branches)
            if paths is not None:
                current = new_branches
                best_branches = list(current)
                best_paths = paths
                best_size = len(current)
                extended = True
                break

    # ---- Also try: start from a different seed (top-degree pair) and extend ----
    # This handles cases where the max clique doesn't extend well but a different set does
    top_deg = sorted(range(n), key=lambda v: degree[v], reverse=True)
    for seed_idx in range(min(5, len(top_deg))):
        if time.time() > deadline - 10:
            break
        seed_v = top_deg[seed_idx]
        seed = [seed_v]
        for w in top_deg:
            if w == seed_v:
                continue
            if (seed_v, w) in edge_set:
                seed.append(w)
                break
        if len(seed) < 2:
            continue
        seed = list(seed)
        # Extend greedily
        cur = list(seed)
        ext = True
        while ext:
            ext = False
            if time.time() > deadline - 5:
                break
            for v in candidate_pool:
                if v in cur:
                    continue
                new_b = cur + [v]
                paths = try_route(new_b)
                if paths is not None:
                    cur = new_b
                    ext = True
                    break
        if len(cur) > best_size:
            best_branches = list(cur)
            best_paths = try_route(cur)
            best_size = len(cur)
        elif len(cur) == best_size and best_paths is None:
            best_paths = try_route(cur)
            if best_paths is not None:
                best_branches = list(cur)

    # ---- Final verification and type conversion ----
    if best_paths is None or len(best_paths) != len(list(combinations(best_branches, 2))):
        # Ultimate fallback: two connected vertices
        u, v = edges[0]
        best_branches = [int(u), int(v)]
        best_paths = [[int(u), int(v)]]

    # Ensure types
    branches_out = [int(b) for b in best_branches]
    paths_out = []
    for p in best_paths:
        paths_out.append([int(x) for x in p])

    # Final sanity check
    if len(branches_out) < 2:
        u, v = edges[0]
        branches_out = [int(u), int(v)]
        paths_out = [[int(u), int(v)]]

    return {'branches': branches_out, 'paths': paths_out}

# EVOLVE-BLOCK-END
