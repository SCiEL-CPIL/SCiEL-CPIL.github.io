"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def path_cover_lp(edges, time_limit):
        """
        Model edge-disjoint path cover as a min-cost flow / LP problem.
        Each undirected edge {u,v} becomes two directed arcs (u,v) and (v,u) with cost 1, capacity [0,1].
        We want to select a subset of arcs such that:
          - For each undirected edge, at most one of its two arcs is used (x_uv + x_vu <= 1)
          - Flow conservation: for each vertex, outflow - inflow = b_v where b_v is the imbalance
          - The number of paths = (1/2) * sum_v |b_v|
        
        To minimize path count, we minimize sum of |b_v| over all vertices.
        This is equivalent to minimizing the L1 norm of the imbalance vector.
        
        LP formulation:
          Variables: x_{uv} in [0,1] for each directed arc
          Objective: minimize sum_v t_v where t_v >= b_v and t_v >= -b_v
          Constraints:
            x_{uv} + x_{vu} <= 1 for each undirected edge
            t_v - (outdeg_v - indeg_v) >= 0
            t_v + (outdeg_v - indeg_v) >= 0
            t_v >= 0
        """
        import numpy as np
        try:
            from scipy.optimize import linprog
        except ImportError:
            return [[u, v] for u, v in edges]
        
        if not edges:
            return []
        
        n = max(max(u, v) for u, v in edges) + 1
        m = len(edges)
        
        # Variables: 0..2m-1 are arc variables, 2m..2m+n-1 are t_v
        num_vars = 2 * m + n
        
        # Objective: minimize sum of t_v
        c = np.zeros(num_vars)
        for v in range(n):
            c[2 * m + v] = 1.0
        
        # Inequality constraints (A_ub @ x <= b_ub)
        A_rows = []
        b_ub_list = []
        
        # Constraint 1: x_uv + x_vu <= 1 for each undirected edge
        for i in range(m):
            row = np.zeros(num_vars)
            row[2 * i] = 1.0
            row[2 * i + 1] = 1.0
            A_rows.append(row)
            b_ub_list.append(1.0)
        
        # Constraint 2: For each vertex v, t_v - (outdeg_v - indeg_v) >= 0
        # i.e., -(outdeg_v - indeg_v) - t_v <= 0
        # outdeg_v - indeg_v = sum of x_{vw} for w in N(v) - sum of x_{wv} for w in N(v)
        for v in range(n):
            # Row: -(outdeg_v - indeg_v) - t_v <= 0
            row = np.zeros(num_vars)
            row[2 * m + v] = -1.0  # coefficient of t_v
            for i in range(m):
                u, w = edges[i]
                # Arc 2*i: u -> w
                if u == v:
                    row[2 * i] -= 1.0  # contributes to outdeg
                if w == v:
                    row[2 * i] += 1.0  # contributes to indeg
                # Arc 2*i+1: w -> u
                if w == v:
                    row[2 * i + 1] -= 1.0  # contributes to outdeg
                if u == v:
                    row[2 * i + 1] += 1.0  # contributes to indeg
            A_rows.append(row)
            b_ub_list.append(0.0)
            
            # Row: (outdeg_v - indeg_v) - t_v <= 0
            row2 = np.zeros(num_vars)
            row2[2 * m + v] = -1.0  # coefficient of t_v
            for i in range(m):
                u, w = edges[i]
                # Arc 2*i: u -> w
                if u == v:
                    row2[2 * i] += 1.0
                if w == v:
                    row2[2 * i] -= 1.0
                # Arc 2*i+1: w -> u
                if w == v:
                    row2[2 * i + 1] += 1.0
                if u == v:
                    row2[2 * i + 1] -= 1.0
            A_rows.append(row2)
            b_ub_list.append(0.0)
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_ub_list)
        
        # Bounds: arc variables in [0,1], t_v in [0, inf]
        lb = np.zeros(num_vars)
        ub = np.full(num_vars, np.inf)
        for i in range(2 * m):
            ub[i] = 1.0
        
        try:
            if time.time() > time_limit:
                return [[u, v] for u, v in edges]
            
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=list(zip(lb, ub)), method='highs')
            
            if not res.success:
                return [[u, v] for u, v in edges]
            
            x = res.x
            directed = []
            for i in range(m):
                u, v = edges[i]
                xv = x[2 * i]
                xu = x[2 * i + 1]
                if xv > 0.5:
                    directed.append((u, v))
                elif xu > 0.5:
                    directed.append((v, u))
                else:
                    # If both are near 0 or both near 1 (shouldn't happen), pick one
                    directed.append((u, v))
            
            # Decompose into paths
            adj = defaultdict(list)
            indeg = defaultdict(int)
            outdeg = defaultdict(int)
            for u, v in directed:
                adj[u].append(v)
                outdeg[u] += 1
                indeg[v] += 1
            
            paths = []
            visited_arcs = set()
            
            # Find path starts: vertices where outdeg > indeg
            starts = []
            for v in range(n):
                if outdeg[v] > indeg[v]:
                    starts.append(v)
            
            # Also need to handle cycles (where outdeg == indeg for all vertices in a component)
            # First, extract paths starting from imbalanced vertices
            for start in starts:
                path = [start]
                cur = start
                while True:
                    found = False
                    for nbr in adj[cur]:
                        arc = (cur, nbr)
                        if arc not in visited_arcs:
                            visited_arcs.add(arc)
                            path.append(nbr)
                            cur = nbr
                            found = True
                            break
                    if not found:
                        break
                if len(path) >= 2:
                    paths.append(path)
            
            # Now handle remaining arcs (cycles)
            remaining_arcs = set(directed) - visited_arcs
            while remaining_arcs:
                start_arc = next(iter(remaining_arcs))
                path = [start_arc[0], start_arc[1]]
                visited_arcs.add(start_arc)
                remaining_arcs.discard(start_arc)
                cur = start_arc[1]
                while True:
                    found = False
                    for nbr in adj[cur]:
                        arc = (cur, nbr)
                        if arc in remaining_arcs:
                            remaining_arcs.discard(arc)
                            visited_arcs.add(arc)
                            path.append(nbr)
                            cur = nbr
                            found = True
                            break
                    if not found:
                        break
                if len(path) >= 2:
                    paths.append(path)
            
            return paths
        
        except Exception:
            return [[u, v] for u, v in edges]

    def greedy_path_cover(edges, time_limit):
        """Fallback: repeatedly extract longest simple path from remaining edges."""
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        remaining = set(edges)
        paths = []
        
        while remaining:
            if time.time() > time_limit:
                for u, v in remaining:
                    paths.append([u, v])
                break
            
            best_path = None
            verts = list(adj.keys())
            verts.sort(key=lambda v: len(adj[v]), reverse=True)
            
            for start in verts:
                if not any((start, v) in remaining or (v, start) in remaining for v in adj[start]):
                    continue
                best = _dfs_longest(start, remaining, adj, time_limit)
                if best and (best_path is None or len(best) > len(best_path)):
                    best_path = best
                if best_path and len(best_path) > 2 * len(remaining) / max(1, len(verts)):
                    break
            
            if best_path is None or len(best_path) < 2:
                for u, v in remaining:
                    paths.append([u, v])
                break
            
            paths.append(best_path)
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i + 1]
                e = tuple(sorted((u, v)))
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
        
        return paths

    def _dfs_longest(start, remaining, adj, time_limit):
        best = [start]
        
        def dfs(node, path):
            nonlocal best
            if time.time() > time_limit:
                return
            if len(path) > len(best):
                best = path[:]
            for nbr in list(adj[node]):
                e = tuple(sorted((node, nbr)))
                if e in remaining and nbr not in path:
                    path.append(nbr)
                    dfs(nbr, path)
                    path.pop()
                    if time.time() > time_limit:
                        return
        
        dfs(start, [start])
        return best

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        elapsed = time.time() - t0
        remaining_time = DEADLINE - elapsed
        if remaining_time < 5:
            # Not enough time; use trivial
            results.append([[u, v] for u, v in edges])
            continue
        
        # Try LP approach first
        per_graph_limit = min(remaining_time - 2, 45.0)
        paths = path_cover_lp(edges, time.time() + per_graph_limit)
        
        # Validate: check that paths are valid (simple, use edges, edge-disjoint, cover all)
        if not _validate_paths(paths, edges):
            # Fallback to greedy
            paths = greedy_path_cover(edges, time.time() + min(remaining_time - 2, 45.0))
        
        results.append(paths)
    
    # Fill in any missing graphs
    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results[:3]

def _validate_paths(paths, edges):
    """Validate that paths form a valid edge-disjoint path cover."""
    if not paths:
        return len(edges) == 0
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    used_edges = set()
    
    for path in paths:
        if len(path) < 2:
            return False
        # Check simple (no repeated vertices)
        if len(set(path)) != len(path):
            return False
        # Check edges exist
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            e = tuple(sorted((u, v)))
            if e not in edge_set:
                return False
            if e in used_edges:
                return False
            used_edges.add(e)
    
    # Check all edges covered
    if used_edges != edge_set:
        return False
    
    return True
# EVOLVE-BLOCK-END
