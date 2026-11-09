import time
import sys

def construct():
    from random import Random
    import numpy as np
    from scipy.optimize import linprog

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycles_in_2matching(adj, n, edge_set):
        """Find all cycles in a 2-matching (max degree 2 subgraph)."""
        # Build adjacency for the 2-matching
        adj2 = {i: [] for i in range(n)}
        for (u, v) in edge_set:
            adj2[u].append(v)
            adj2[v].append(u)
        
        visited = [False] * n
        cycles = []
        
        for start in range(n):
            if visited[start] or len(adj2[start]) == 0:
                continue
            # Follow the path
            path = [start]
            visited[start] = True
            cur = start
            prev = -1
            while True:
                nxt = None
                for v in adj2[cur]:
                    if v != prev and not visited[v]:
                        nxt = v
                        break
                if nxt is None:
                    # Check if we can close a cycle
                    if cur != start:
                        # Check if start is in adj2[cur]
                        if start in adj2[cur]:
                            cycles.append(path)
                    break
                path.append(nxt)
                visited[nxt] = True
                prev = cur
                cur = nxt
                if cur == start:
                    cycles.append(path[:-1])
                    break
        return cycles

    def decompose(edges, n):
        remaining = set()
        for (u, v) in edges:
            remaining.add((min(u, v), max(u, v)))
        
        edge_list = sorted(remaining)
        m = len(edge_list)
        
        if m == 0:
            return []
        
        # Build edge index map
        edge_idx = {e: i for i, e in enumerate(edge_list)}
        
        # Build incidence: for each vertex, list of (edge_idx, other_vertex)
        vertex_edges = {i: [] for i in range(n)}
        for i, (u, v) in enumerate(edge_list):
            vertex_edges[u].append((i, v))
            vertex_edges[v].append((i, u))
        
        # LP formulation:
        # Variables: x_e for each edge (0 or 1)
        # Maximize: sum x_e (maximize edges in 2-matching)
        # Subject to: for each vertex v, sum_{e incident to v} x_e <= 2
        # This gives a maximum 2-matching
        # Then we extract cycles from it
        
        deadline = time.time() + 110
        
        # c: minimize -sum(x_e)
        c = -np.ones(m)
        
        # A_ub: degree constraints, each row sums to <= 2
        A_ub = np.zeros((n, m))
        for i, (u, v) in enumerate(edge_list):
            A_ub[u, i] = 1
            A_ub[v, i] = 1
        b_ub = np.full(n, 2)
        
        bounds = [(0, 1)] * m
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                # Get the 2-matching edges
                x = res.x
                selected_edges = set()
                for i, e in enumerate(edge_list):
                    if x[i] > 0.5:
                        selected_edges.add(e)
                
                # Find cycles in the 2-matching
                cycles = find_cycles_in_2matching(vertex_edges, n, selected_edges)
                
                pieces = []
                used_edges = set()
                for cyc in cycles:
                    if len(cyc) >= 3:
                        # Verify all edges exist
                        valid = True
                        for i in range(len(cyc)):
                            a = cyc[i]
                            b = cyc[(i + 1) % len(cyc)]
                            e = (min(a, b), max(a, b))
                            if e not in selected_edges:
                                valid = False
                                break
                        if valid:
                            pieces.append(list(cyc))
                            for i in range(len(cyc)):
                                a = cyc[i]
                                b = cyc[(i + 1) % len(cyc)]
                                e = (min(a, b), max(a, b))
                                used_edges.add(e)
                
                # Remaining edges in selected_edges but not in cycles become single edges
                for e in selected_edges:
                    if e not in used_edges:
                        u, v = e
                        pieces.append([u, v])
                
                # Edges not in 2-matching are single edges
                for e in edge_list:
                    if e not in selected_edges:
                        u, v = e
                        pieces.append([u, v])
                
                return pieces
        except Exception:
            pass
        
        # Fallback: greedy longest cycle
        pieces = []
        cur_adj = {i: [] for i in range(n)}
        for (u, v) in remaining:
            cur_adj[u].append(v)
            cur_adj[v].append(u)
        
        while True:
            if time.time() > deadline:
                break
            # Find longest cycle using DFS
            best_cycle = None
            best_len = 0
            for start in range(n):
                if not cur_adj[start]:
                    continue
                visited = [False] * n
                in_stack = [False] * n
                depth = [-1] * n
                parent = [-1] * n
                cycles_found = []
                
                def dfs(u):
                    visited[u] = True
                    in_stack[u] = True
                    for v in cur_adj[u]:
                        if not visited[v]:
                            parent[v] = u
                            depth[v] = depth[u] + 1
                            dfs(v)
                        elif in_stack[v]:
                            if depth[u] - depth[v] >= 2:
                                cycle = []
                                cur = u
                                while cur != v:
                                    cycle.append(cur)
                                    cur = parent[cur]
                                cycle.append(v)
                                cycle.reverse()
                                cycles_found.append(cycle)
                    in_stack[u] = False
                
                depth[start] = 0
                dfs(start)
                for cyc in cycles_found:
                    L = len(cyc)
                    if L > best_len:
                        best_len = L
                        best_cycle = cyc
            
            if best_cycle is None:
                break
            k = len(best_cycle)
            if k < 3 or len(set(best_cycle)) != k:
                break
            ok = True
            for i in range(k):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % k]
                e = (min(a, b), max(a, b))
                if e not in remaining:
                    ok = False
                    break
            if not ok:
                break
            for i in range(k):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % k]
                e = (min(a, b), max(a, b))
                remaining.discard(e)
                cur_adj[a].remove(b)
                cur_adj[b].remove(a)
            pieces.append(list(best_cycle))
        
        for (u, v) in sorted(remaining):
            pieces.append([u, v])
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n = spec[0]
        edges = instance(*spec)
        results.append(decompose(edges, n))
    return results
