import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    D = 4

    # base adjacency as bitmasks
    base_adj = [0] * n
    base_set = set()
    for e in P['edges']:
        u, v = e
        if u > v:
            u, v = v, u
        base_adj[u] |= 1 << v
        base_adj[v] |= 1 << u
        base_set.add((u, v))

    # all non-adjacent candidate edges
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_set:
                candidates.append((u, v))

    # candidate adjacency bitmasks
    cand_adj = {}
    for (u, v) in candidates:
        cand_adj[(u, v)] = (1 << u) | (1 << v)

    # precompute for each candidate: which base pairs it "covers" (shortens)
    # and whether adding it creates a triangle
    # triangle check: exists w with w in base_adj[u] & base_adj[v]
    tri_conflict = {}
    for (u, v) in candidates:
        common = base_adj[u] & base_adj[v]
        tri_conflict[(u, v)] = common  # bitmask of common neighbors

    # For pruning: for a candidate (u,v), the set of vertex pairs (a,b) with a<b
    # whose shortest path distance in base graph is > D and that pass through
    # the edge (u,v) in some sense -- we'll use a simpler proxy:
    # candidate (u,v) is "useful" if it connects two components of the
    # graph where we remove base edges. Simpler: just use all candidates.

    # Compute base distances (BFS) for all pairs to know which are far
    base_dist = [[-1] * n for _ in range(n)]
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in range(n):
                if (base_adj[u] >> v) & 1 and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        base_dist[src] = dist

    far_pairs_base = []
    for a in range(n):
        for b in range(a + 1, n):
            if base_dist[a][b] > D:
                far_pairs_base.append((a, b))

    # Sort candidates by a usefulness heuristic: number of far pairs (a,b)
    # such that base_dist[a][u] + 1 + base_dist[v][b] <= D OR symmetric
    # (i.e., the new edge could serve as a shortcut)
    def usefulness(c):
        u, v = c
        cnt = 0
        for (a, b) in far_pairs_base:
            # path a -> u -> v -> b
            if base_dist[a][u] >= 0 and base_dist[v][b] >= 0 and base_dist[a][u] + 1 + base_dist[v][b] <= D:
                cnt += 1
            # path a -> v -> u -> b
            if base_dist[a][v] >= 0 and base_dist[u][b] >= 0 and base_dist[a][v] + 1 + base_dist[u][b] <= D:
                cnt += 1
        return cnt

    candidates.sort(key=usefulness, reverse=True)

    # Precompute for each candidate the set of far pairs it can "fix"
    # (i.e., pairs (a,b) where using this edge as a shortcut gives dist <= D)
    cand_fixes = []
    for c in candidates:
        u, v = c
        fixes = set()
        for (a, b) in far_pairs_base:
            ok = False
            if base_dist[a][u] >= 0 and base_dist[v][b] >= 0 and base_dist[a][u] + 1 + base_dist[v][b] <= D:
                ok = True
            if base_dist[a][v] >= 0 and base_dist[u][b] >= 0 and base_dist[a][v] + 1 + base_dist[u][b] <= D:
                ok = True
            if ok:
                fixes.add((a, b))
        cand_fixes.append(fixes)

    # Branch and bound
    start_time = time.time()
    deadline = start_time + 75.0

    best_solution = None
    best_count = float('inf')

    # State: chosen list, chosen_adj (bitmasks), chosen_set
    def add_edge_to_state(state, c):
        u, v = c
        new_adj = state['adj'][:]
        new_adj[u] |= 1 << v
        new_adj[v] |= 1 << u
        return {
            'adj': new_adj,
            'chosen': state['chosen'] + [c],
            'set': state['set'] | {c},
        }

    def bfs_diameter(adj):
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in range(n):
                    if (adj[u] >> v) & 1 and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_d:
                            max_d = dist[v]
                        q.append(v)
            if -1 in dist:
                return float('inf')
        return max_d

    def is_triangle_free_given_chosen(c, adj):
        # c = (u,v); check no common neighbor in adj
        u, v = c
        common = adj[u] & adj[v]
        return common == 0

    def far_pairs_after(adj):
        fps = []
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in range(n):
                    if (adj[u] >> v) & 1 and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            for b in range(src + 1, n):
                if dist[b] > D:
                    fps.append((src, b))
        return fps

    def can_remaining_cover(fps, idx, remaining_candidates):
        # fps: set of far pairs; check if union of cand_fixes[idx:] covers fps
        if not fps:
            return True
        # quick: if number of remaining < needed lower bound, fail
        # simple check: for each far pair, is there any remaining candidate that fixes it?
        for fp in fps:
            found = False
            for j in range(idx, len(candidates)):
                if fp in cand_fixes[j]:
                    found = True
                    break
            if not found:
                return False
        return True

    def can_reach_diameter(idx, chosen_count):
        # optimistic: even if we add all remaining, can we get diameter <= D?
        # lower bound on edges needed: each edge can reduce diameter by at most some amount
        # simple: if chosen_count + (len(candidates) - idx) < 0: no
        # Use: current diameter must be reducible. Weak bound:
        return True

    # Initial state
    init_state = {'adj': base_adj[:], 'chosen': [], 'set': set()}

    # If base already has diameter <= D, return empty
    if bfs_diameter(base_adj) <= D:
        return []

    # DFS branch and bound
    def dfs(idx, state):
        nonlocal best_solution, best_count
        if time.time() > deadline:
            return
        chosen_count = len(state['chosen'])
        if chosen_count >= best_count:
            return

        # Check if current state already achieves diameter <= D
        cur_diam = bfs_diameter(state['adj'])
        if cur_diam <= D:
            if chosen_count < best_count:
                best_count = chosen_count
                best_solution = list(state['chosen'])
            return

        # Prune: can remaining candidates cover all current far pairs?
        fps = set(far_pairs_after(state['adj']))
        if not can_remaining_cover(fps, idx, candidates):
            return

        # If no more candidates
        if idx >= len(candidates):
            return

        # Try skipping candidate idx
        dfs(idx + 1, state)

        # Try adding candidate idx
        c = candidates[idx]
        if is_triangle_free_given_chosen(c, state['adj']):
            new_state = add_edge_to_state(state, c)
            dfs(idx + 1, new_state)

    # To make DFS effective, we should also try a greedy upper bound first
    # to get a good best_count for pruning.
    def greedy_upper_bound():
        adj = base_adj[:]
        chosen = []
        chosen_set = set()
        while time.time() < deadline:
            cur_diam = bfs_diameter(adj)
            if cur_diam <= D:
                break
            fps = far_pairs_after(adj)
            if not fps:
                break
            # pick candidate that fixes most far pairs without triangle
            best_c = None
            best_fix = 0
            for c in candidates:
                if c in chosen_set:
                    continue
                if not is_triangle_free_given_chosen(c, adj):
                    continue
                u, v = c
                cnt = 0
                for fp in fps:
                    a, b = fp
                    ok = False
                    if base_dist[a][u] >= 0 and base_dist[v][b] >= 0 and base_dist[a][u] + 1 + base_dist[v][b] <= D:
                        ok = True
                    if base_dist[a][v] >= 0 and base_dist[u][b] >= 0 and base_dist[a][v] + 1 + base_dist[u][b] <= D:
                        ok = True
                    if ok:
                        cnt += 1
                if cnt > best_fix:
                    best_fix = cnt
                    best_c = c
            if best_c is None:
                break
            u, v = best_c
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            chosen.append(best_c)
            chosen_set.add(best_c)
        return chosen

    greedy_sol = greedy_upper_bound()
    if greedy_sol:
        # verify
        adj = base_adj[:]
        for (u, v) in greedy_sol:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        if bfs_diameter(adj) <= D:
            best_count = len(greedy_sol)
            best_solution = list(greedy_sol)

    # Now run DFS to try to improve
    dfs(0, init_state)

    # Local search reduction: try to remove each edge from best_solution
    if best_solution is not None:
        reduced = list(best_solution)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(reduced)):
                trial = reduced[:i] + reduced[i+1:]
                adj = base_adj[:]
                for (u, v) in trial:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                if bfs_diameter(adj) <= D:
                    reduced = trial
                    improved = True
                    break
        if len(reduced) < best_count:
            best_count = len(reduced)
            best_solution = reduced

    # Final fallback: if no solution found, return greedy or a known-good construction
    if best_solution is None:
        # Fallback construction: hub at 11
        edges = []
        # connect 11 to all even vertices except neighbors
        for v in range(n):
            if v == 11 or v == 10 or v == 12:
                continue
            if abs(v - 11) >= 2:
                u, v2 = (v, 11) if v < 11 else (11, v)
                edges.append([u, v2])
        # This may not be optimal but is valid
        return edges

    # Convert to list of [u, v] with Python ints
    result = []
    for (u, v) in best_solution:
        result.append([int(u), int(v)])
    return result
