import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def eulerian_decompose(edges):
        """
        Make graph Eulerian by pairing odd vertices with shortest paths,
        find Eulerian circuit, split into simple paths at repeated vertices.
        Returns list of simple paths (vertex lists).
        """
        if not edges:
            return []
        adj = build_adj(edges)
        degree = {v: len(adj[v]) for v in adj}
        odd = [v for v in degree if degree[v] % 2 == 1]

        # Build full graph including virtual edges for pairing
        # We'll work on a multigraph represented by edge list
        # Pair odd vertices greedily by BFS shortest path
        import heapq
        def bfs_shortest(src):
            # returns dict dist and parent
            dist = {src: 0}
            parent = {}
            q = [src]
            for u in q:
                for nb in adj[u]:
                    if nb not in dist:
                        dist[nb] = dist[u] + 1
                        parent[nb] = u
                        q.append(nb)
            return dist, parent

        virtual_edges = []
        # pair odd vertices
        i = 0
        while i < len(odd):
            if i + 1 >= len(odd):
                break
            a, b = odd[i], odd[i + 1]
            dist, parent = bfs_shortest(a)
            if b in dist:
                # reconstruct path
                path = [b]
                cur = b
                while cur != a:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                for k in range(len(path) - 1):
                    virtual_edges.append(tuple(sorted((path[k], path[k+1]))))
            i += 2

        # Now build multigraph adjacency with edge counts
        # Use a list of (u,v) edges, track used
        all_edges = list(edges) + virtual_edges
        # Build adjacency as list of (neighbor, edge_id)
        m = len(all_edges)
        m_adj = defaultdict(list)
        for eid, (u, v) in enumerate(all_edges):
            m_adj[u].append((v, eid))
            m_adj[v].append((u, eid))

        used = [False] * m
        # Hierholzer to find Eulerian circuit (graph should be Eulerian now)
        # But virtual edges make it Eulerian; real edges may not all be in one circuit
        # We'll do: while there are unused edges, find a circuit/trail
        circuits = []
        remaining_real = set(range(len(edges)))

        def find_circuit(start):
            # returns list of edge ids forming a circuit starting/ending at start
            stack = [(start, None, None)]  # (vertex, incoming_edge_id, direction)
            circuit = []
            # Hierholzer
            v = start
            path = [v]
            edge_path = []
            while True:
                # find unused edge from v
                taken = False
                for nb, eid in m_adj[v]:
                    if not used[eid]:
                        used[eid] = True
                        edge_path.append(eid)
                        path.append(nb)
                        v = nb
                        taken = True
                        break
                if not taken:
                    # backtrack
                    if len(path) == 1:
                        break
                    v = path.pop()
                    edge_path.pop()
                else:
                    continue
            # This naive backtracking is wrong for Hierholzer. Let me use proper stack.
            return path, edge_path

        # Proper Hierholzer using stack
        def hierholzer(start):
            # returns list of edge ids in circuit order
            stack = [start]
            circuit = []
            # We need to track position; use iterative with adjacency index
            idx = defaultdict(int)
            while stack:
                v = stack[-1]
                # advance idx[v] until unused edge
                while idx[v] < len(m_adj[v]) and used[m_adj[v][idx[v]][1]]:
                    idx[v] += 1
                if idx[v] < len(m_adj[v]):
                    nb, eid = m_adj[v][idx[v]]
                    idx[v] += 1
                    used[eid] = True
                    stack.append(nb)
                else:
                    circuit.append(stack.pop())
            circuit.reverse()
            return circuit

        # Find all circuits covering all edges
        circuits = []
        for eid in range(m):
            if not used[eid]:
                u, v = all_edges[eid]
                circuit = hierholzer(u)
                circuits.append(circuit)

        # Now split each circuit into simple paths at repeated vertices
        paths = []
        for circuit in circuits:
            # circuit is list of vertices, first==last for circuit
            # Split into simple paths: whenever a vertex repeats, cut
            current = [circuit[0]]
            seen = {circuit[0]}
            for k in range(1, len(circuit)):
                vtx = circuit[k]
                if vtx in seen:
                    if len(current) >= 2:
                        paths.append(current)
                    current = [vtx]
                    seen = {vtx}
                else:
                    current.append(vtx)
                    seen.add(vtx)
            if len(current) >= 2:
                paths.append(current)

        return paths

    def verify_and_fix(paths, edges):
        """Ensure paths are simple, use valid edges, edge-disjoint, cover all edges."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        used = set()
        valid_paths = []
        for p in paths:
            # check simple
            if len(p) != len(set(p)):
                # try to salvage: take prefix until repeat
                seen = set()
                fixed = []
                for v in p:
                    if v in seen:
                        break
                    seen.add(v)
                    fixed.append(v)
                p = fixed
            if len(p) < 2:
                continue
            # check edges valid and unused
            ok = True
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i+1])))
                if e not in edge_set or e in used:
                    ok = False
                    break
            if ok:
                for i in range(len(p) - 1):
                    used.add(tuple(sorted((p[i], p[i+1]))))
                valid_paths.append(p)
        # Add any missing edges as single-edge paths
        for e in edge_set:
            if e not in used:
                valid_paths.append([e[0], e[1]])
                used.add(e)
        return valid_paths

    def merge_paths(paths, edges):
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        improved = True
        while improved:
            improved = False
            new_current = []
            used = [False] * len(current)
            for i in range(len(current)):
                if used[i]:
                    continue
                p1 = current[i]
                merged_with = -1
                for j in range(len(current)):
                    if i == j or used[j]:
                        continue
                    p2 = current[j]
                    # try p1 + p2[1:]
                    if p1[-1] == p2[0]:
                        merged = p1 + p2[1:]
                        if len(merged) == len(set(merged)) and all(
                            tuple(sorted((merged[k], merged[k+1]))) in edge_set
                            for k in range(len(merged) - 1)
                        ):
                            new_current.append(merged)
                            used[j] = True
                            merged_with = j
                            improved = True
                            break
                    if p2[-1] == p1[0]:
                        merged = p2 + p1[1:]
                        if len(merged) == len(set(merged)) and all(
                            tuple(sorted((merged[k], merged[k+1]))) in edge_set
                            for k in range(len(merged) - 1)
                        ):
                            new_current.append(merged)
                            used[j] = True
                            merged_with = j
                            improved = True
                            break
                if merged_with == -1:
                    new_current.append(p1)
            current = new_current
        return current

    def decompose_one(edges, deadline):
        if not edges:
            return []
        best = None
        best_count = len(edges)

        # Strategy 1: Eulerian
        if time.time() < deadline - 5:
            paths = eulerian_decompose(edges)
            paths = verify_and_fix(paths, edges)
            paths = merge_paths(paths, edges)
            if len(paths) < best_count:
                best_count = len(paths)
                best = paths

        # Strategy 2: greedy with multiple seeds, local merge
        for seed in range(20):
            if time.time() > deadline - 3:
                break
            rng = random.Random(seed * 31 + 7)
            remaining = set(tuple(sorted(e)) for e in edges)
            adj = build_adj(list(remaining))
            degree = {v: len(adj[v]) for v in adj}
            paths = []
            while remaining:
                if time.time() > deadline - 2:
                    break
                # pick start: prefer odd
                odd_verts = [v for v in adj if degree.get(v, 0) % 2 == 1 and adj[v]]
                if odd_verts:
                    start = rng.choice(odd_verts)
                else:
                    cands = [v for v in adj if adj[v]]
                    if not cands:
                        break
                    start = rng.choice(cands)
                # greedy extend with Warnsdorff
                path = [start]
                visited = {start}
                for _ in range(200):
                    cur = path[-1]
                    nbs = [nb for nb in adj[cur] if nb not in visited]
                    if not nbs:
                        break
                    def score(nb):
                        return sum(1 for x in adj[nb] if x not in visited)
                    if rng.random() < 0.8:
                        nbs.sort(key=score)
                        top = min(3, len(nbs))
                        choice = rng.choice(nbs[:top])
                    else:
                        choice = rng.choice(nbs)
                    path.append(choice)
                    visited.add(choice)
                if len(path) < 2:
                    if adj[start]:
                        nb = rng.choice(list(adj[start]))
                        path = [start, nb]
                    else:
                        e = next(iter(remaining))
                        path = [e[0], e[1]]
                path_edges = set()
                for i in range(len(path) - 1):
                    e = tuple(sorted((path[i], path[i+1])))
                    if e in remaining:
                        path_edges.add(e)
                if not path_edges:
                    e = next(iter(remaining))
                    path = [e[0], e[1]]
                    path_edges = {e}
                for e in path_edges:
                    remaining.discard(e)
                    u, v = e
                    adj[u].discard(v)
                    adj[v].discard(u)
                    degree[u] = degree.get(u, 0) - 1
                    degree[v] = degree.get(v, 0) - 1
                for v in list(adj.keys()):
                    if not adj[v]:
                        del adj[v]
                paths.append(path)
            paths = verify_and_fix(paths, edges)
            paths = merge_paths(paths, edges)
            if len(paths) < best_count:
                best_count = len(paths)
                best = paths

        if best is None:
            return [[u, v] for u, v in edges]
        return best

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            break
        edges = instance(*spec)
        # allocate time per graph
        per = (deadline - time.time()) / (3 - len(results))
        paths = decompose_one(edges, time.time() + per)
        results.append(paths)

    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])

    return results
