"""
Longest simple path in a pinned G(100, 1.8/100) graph.

Strategy:
  1. Build a strong initial path using vertex-splitting + Hopcroft-Karp matching
     (gives a set of vertex-disjoint paths covering many vertices).
  2. Stitch the matching-derived paths together into a single long path.
  3. Refine with iterative insertion, end-extension, and IDDFS with Warnsdorff
     ordering to escape local traps.
  4. Local search: 2-opt segment reversal + insertion to add unvisited vertices.
"""
import random
import time
from collections import deque

N = 100
EDGES = [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]

ADJ = [set() for _ in range(N)]
DEG = [0] * N
for u, v in EDGES:
    ADJ[u].add(v)
    ADJ[v].add(u)
    DEG[u] += 1
    DEG[v] += 1


def find_components():
    visited = set()
    comps = []
    for s in range(N):
        if s not in visited:
            comp = set()
            stack = [s]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.add(u)
                for v in ADJ[u]:
                    if v not in visited:
                        stack.append(v)
            comps.append(comp)
    return comps


def matching_seed(comp_vertices, deadline):
    """
    Use vertex-splitting + Hopcroft-Karp to find a maximum matching.
    Extract vertex-disjoint paths from the matching.
    Returns a list of paths (each a list of vertices).
    """
    vertices = sorted(comp_vertices)
    n = len(vertices)
    idx = {v: i for i, v in enumerate(vertices)}

    # Bipartite: left = v_out, right = v_in.
    # Edge u_out -> v_in if (u,v) in original graph.
    left_adj = [set() for _ in range(n)]
    for u in vertices:
        for v in ADJ[u]:
            if v in idx:
                left_adj[idx[u]].add(idx[v])

    pair_u = [-1] * n
    pair_v = [-1] * n
    dist = [0] * n

    def bfs():
        q = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = float('inf')
        found = False
        while q:
            u = q.popleft()
            for v in left_adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == float('inf'):
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found

    def dfs(u):
        for v in left_adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = float('inf')
        return False

    while time.time() < deadline:
        if not bfs():
            break
        for u in range(n):
            if pair_u[u] == -1:
                dfs(u)

    # Build next_node: vertices[u] -> vertices[pair_u[u]]
    next_node = {}
    prev_node = {}
    for u in range(n):
        if pair_u[u] != -1:
            next_node[vertices[u]] = vertices[pair_u[u]]
        if pair_v[u] != -1:
            prev_node[vertices[u]] = vertices[pair_u[u]]  # Wait, pair_v[v]=u means u_out->v_in, so prev of v is u

    # Actually: pair_u[u] = v means edge u->v (u_out to v_in). So next[u]=v.
    # pair_v[v] = u means the same edge, so prev[v]=u.
    next_node = {}
    prev_node = {}
    for u in range(n):
        if pair_u[u] != -1:
            next_node[vertices[u]] = vertices[pair_u[u]]
            prev_node[vertices[pair_u[u]]] = vertices[u]

    # Find path starts: vertices with no incoming edge (not in prev_node)
    visited = set()
    paths = []
    for start in vertices:
        if start not in prev_node and start not in visited:
            path = [start]
            visited.add(start)
            cur = start
            while cur in next_node:
                cur = next_node[cur]
                if cur in visited:
                    break
                path.append(cur)
                visited.add(cur)
            if len(path) > 1:
                paths.append(path)
            elif len(path) == 1:
                paths.append(path)

    # Also pick up any unvisited vertices as singleton paths
    for v in vertices:
        if v not in visited:
            paths.append([v])
            visited.add(v)

    return paths


def stitch_paths(paths, deadline):
    """
    Stitch multiple vertex-disjoint paths into one long path by finding
    edges between endpoints of different paths.
    """
    if not paths:
        return []
    if len(paths) == 1:
        return paths[0][:]

    # Sort by length descending
    paths = sorted(paths, key=len, reverse=True)
    used = [False] * len(paths)
    main = paths[0][:]
    used[0] = True

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(1, len(paths)):
            if used[i]:
                continue
            p = paths[i]
            # Try to attach p to front or back of main
            # Attach to back: main[-1] -- p[0]
            if main and main[-1] in ADJ[p[0]]:
                main.extend(p)
                used[i] = True
                improved = True
                continue
            # Attach to back: main[-1] -- p[-1] (reverse p)
            if main and main[-1] in ADJ[p[-1]]:
                main.extend(reversed(p))
                used[i] = True
                improved = True
                continue
            # Attach to front: p[0] -- main[0]
            if main and p[0] in ADJ[main[0]]:
                main = p + main
                used[i] = True
                improved = True
                continue
            # Attach to front: p[-1] -- main[0] (reverse p)
            if main and p[-1] in ADJ[main[0]]:
                main = list(reversed(p)) + main
                used[i] = True
                improved = True
                continue
            # Try inserting p between two consecutive vertices in main
            for j in range(len(main) - 1):
                if main[j] in ADJ[p[0]] and p[-1] in ADJ[j + 1]:
                    # Can't directly insert; need to find a way
                    pass
            # Try to find an edge from any vertex in main to any vertex in p
            # and reverse/splice
            for j in range(len(main)):
                for k in range(len(p)):
                    if main[j] in ADJ[p[k]]:
                        # Splice: main[0..j] + reversed(p[k:]) + p[:k] reversed? No.
                        # This is getting complex. Let's just try endpoint connections.
                        pass
    return main


def greedy_warnsdorff(start, rng):
    seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        v = min(cand, key=lambda w: (len(ADJ[w] - seen), rng.random()))
        path.append(v)
        seen.add(v)
    return path


def greedy_max_degree(start, rng):
    seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        v = max(cand, key=lambda w: (DEG[w], rng.random()))
        path.append(v)
        seen.add(v)
    return path


def greedy_random(start, rng):
    seen = {start}
    path = [start]
    while True:
        cand = list(ADJ[path[-1]] - seen)
        if not cand:
            break
        v = rng.choice(cand)
        path.append(v)
        seen.add(v)
    return path


def greedy_mixed(start, rng):
    seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        weights = rng.random() * 2
        v = max(cand, key=lambda w: (-len(ADJ[w] - seen), DEG[w] * weights, rng.random()))
        path.append(v)
        seen.add(v)
    return path


def greedy_forced(start, rng):
    """Prefer the unique unvisited neighbor (forced move); tie-break by max degree."""
    seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        if len(cand) == 1:
            v = next(iter(cand))
        else:
            v = max(cand, key=lambda w: (DEG[w], rng.random()))
        path.append(v)
        seen.add(v)
    return path


def extend_both_ends(path, seen):
    path = list(path)
    seen = set(seen)
    improved = True
    while improved:
        improved = False
        while True:
            cand = ADJ[path[0]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            improved = True
        while True:
            cand = ADJ[path[-1]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.append(v)
            seen.add(v)
            improved = True
    return path


def insert_unvisited(path, seen, deadline):
    """Repeatedly try to insert unvisited vertices into the path."""
    path = list(path)
    seen = set(seen)
    unvisited = set(range(N)) - seen
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u in list(unvisited):
            # Try inserting at any position
            for i in range(len(path) - 1):
                if path[i] in ADJ[u] and path[i + 1] in ADJ[u]:
                    path.insert(i + 1, u)
                    seen.add(u)
                    unvisited.discard(u)
                    improved = True
                    break
            if improved:
                break
        # Extend ends
        while True:
            cand = ADJ[path[0]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            improved = True
        while True:
            cand = ADJ[path[-1]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.append(v)
            seen.add(v)
            improved = True
    return path


def iddfs_refine(start, seen, path, best, deadline, max_depth):
    """
    Iterative deepening DFS with Warnsdorff ordering to escape local traps.
    """
    if time.time() > deadline:
        return
    if len(path) > len(best):
        best.clear()
        best.extend(path)
    if len(path) >= max_depth:
        return
    cand = ADJ[path[-1]] - seen
    if not cand:
        return
    # Warnsdorff ordering
    cand = sorted(cand, key=lambda w: (len(ADJ[w] - seen), w))
    for v in cand:
        path.append(v)
        seen.add(v)
        iddfs_refine(start, seen, path, best, deadline, max_depth)
        path.pop()
        seen.remove(v)
        if time.time() > deadline:
            return
        if len(best) >= max_depth:
            return


def two_opt_improve(path, seen, deadline):
    """
    Iteratively apply: (a) insert unvisited vertices, (b) extend ends,
    (c) 2-opt segment reversals that create new insertion opportunities.
    Each pass is O(n^2) in path length; loop until no improvement or deadline.
    """
    path = list(path)
    seen = set(seen)
    unvisited = set(range(N)) - seen
    improved = True
    while improved and time.time() < deadline:
        improved = False

        # (a) Insert unvisited vertices at any position
        for u in list(unvisited):
            for i in range(len(path) - 1):
                if path[i] in ADJ[u] and path[i + 1] in ADJ[u]:
                    path.insert(i + 1, u)
                    seen.add(u)
                    unvisited.discard(u)
                    improved = True
                    break
            if improved:
                break

        # (b) Extend both ends greedily (Warnsdorff)
        while True:
            cand = ADJ[path[0]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            improved = True
        while True:
            cand = ADJ[path[-1]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.append(v)
            seen.add(v)
            improved = True

        # (c) 2-opt: reverse a segment to create new adjacencies
        n = len(path)
        found_rev = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                # After reversing path[i+1..j], the new connections are:
                #   path[i] - path[j]  and  path[i+1] - path[j+1] (if j+1 < n)
                if path[i] in ADJ[path[j]]:
                    if j + 1 < n:
                        if path[i + 1] in ADJ[path[j + 1]]:
                            path[i + 1:j + 1] = path[i + 1:j + 1][::-1]
                            improved = True
                            found_rev = True
                            break
                    else:
                        path[i + 1:j + 1] = path[i + 1:j + 1][::-1]
                        improved = True
                        found_rev = True
                        break
            if found_rev:
                break
    return path


def construct():
    t0 = time.time()
    deadline = t0 + 80  # Safety margin before 90s kill

    comps = find_components()
    comps.sort(key=len, reverse=True)

    best_path = []
    rng = random.Random(42)

    for comp in comps:
        if time.time() > deadline:
            break
        comp_size = len(comp)
        comp_list = sorted(comp)
        target = comp_size

        # Phase 1: Matching-based seed
        if time.time() < deadline - 10:
            paths = matching_seed(comp_list, time.time() + 5)
            stitched = stitch_paths(paths, time.time() + 5)
            if stitched:
                stitched = extend_both_ends(stitched, set(stitched))
                stitched = insert_unvisited(stitched, set(stitched), time.time() + 5)
                if len(stitched) > len(best_path):
                    best_path = stitched[:]

        # Phase 2: Diverse greedy starts
        for start in comp_list:
            if time.time() > deadline - 15:
                break
            for func in [greedy_warnsdorff, greedy_max_degree, greedy_random, greedy_mixed, greedy_forced]:
                if time.time() > deadline - 15:
                    break
                p = func(start, rng)
                p = extend_both_ends(p, set(p))
                p = insert_unvisited(p, set(p), min(time.time() + 2, deadline - 15))
                if len(p) > len(best_path):
                    best_path = p[:]

        # Phase 3: IDDFS refinement
        if len(best_path) < target and time.time() < deadline - 5:
            # Try from endpoints of current best
            starts_to_try = set()
            if best_path:
                starts_to_try.add(best_path[0])
                starts_to_try.add(best_path[-1])
            # Add a few low-degree vertices
            for v in comp_list:
                if DEG[v] <= 2:
                    starts_to_try.add(v)
                if len(starts_to_try) >= 10:
                    break

            for start in starts_to_try:
                if time.time() > deadline - 5:
                    break
                budget = min(3.0, (deadline - 5 - time.time()) * 0.4)
                dl = time.time() + budget
                local_best = list(best_path)
                iddfs_refine(start, {start}, [start], local_best, dl, target)
                if len(local_best) > len(best_path):
                    best_path = local_best
                    if len(best_path) >= target:
                        break

        # Phase 4: 2-opt + insertion (with multiple restarts from perturbed starts)
        if len(best_path) < target and time.time() < deadline - 2:
            p = two_opt_improve(best_path, set(best_path), deadline - 2)
            if len(p) > len(best_path):
                best_path = p

            # Additional 2-opt passes from different starting paths
            if len(best_path) < target and time.time() < deadline - 2:
                for start in comp_list[:15]:
                    if time.time() > deadline - 2:
                        break
                    for func in [greedy_forced, greedy_warnsdorff]:
                        if time.time() > deadline - 2:
                            break
                        p2 = func(start, rng)
                        p2 = extend_both_ends(p2, set(p2))
                        p2 = two_opt_improve(p2, set(p2), min(time.time() + 3, deadline - 2))
                        if len(p2) > len(best_path):
                            best_path = p2

    # Final safety
    if not best_path:
        best_path = [0]

    # Verify validity
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in ADJ[best_path[i]]:
            return [0]
    if len(set(best_path)) != len(best_path):
        seen = set()
        fixed = []
        for v in best_path:
            if v not in seen:
                fixed.append(v)
                seen.add(v)
        best_path = fixed

    return best_path
