"""Path decomposition of three fixed connected graphs (Gallai's conjecture).

Uses min-cost flow (via scipy.optimize.linprog) to directly minimize the number
of edge-disjoint paths covering all edges.  Fallback to constructive Eulerian
approach if LP fails or times out.
"""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import deque

def construct():
    start_time = time.time()
    DEADLINE = start_time + 130  # 20s safety margin before 150s hard limit

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _lp_path_cover(edges, n, time_limit=30):
        """Solve min-cost flow LP to find minimum number of edge-disjoint paths.
        
        Variables: x_{uv} in {0,1} for each directed arc (u,v), meaning arc used.
        For each undirected edge {u,v}, we have arcs (u,v) and (v,u), at most one used.
        
        Objective: minimize sum of x_{uv} over all arcs / 2? No.
        Actually: minimize number of path starts.
        
        Better formulation:
        - Let f_{uv} be flow on arc (u,v), 0 <= f <= 1
        - For each undirected edge {u,v}: f_{uv} + f_{vu} <= 1 (at most one direction)
        - For each undirected edge {u,v}: f_{uv} + f_{vu} = 1 (must use exactly one)
        - Flow conservation with imbalances: 
          sum_{v: (u,v) in E} f_{uv} - sum_{v: (v,u) in E} f_{vu} = b_u
          where b_u is +1 if u is a path start, -1 if path end, 0 otherwise
        - b_u must be integer, sum b_u = 0
        - Number of paths = (sum of positive b_u) = (number of path starts)
        
        To minimize paths: minimize sum of b_u over u where b_u > 0.
        This is equivalent to: minimize (1/2) * sum_u |b_u|
        
        Since the constraint matrix is TU, we can solve the LP relaxation.
        
        Simpler: enumerate possible b values. b_u in {-1, 0, 1} for each u,
        sum b_u = 0, and the flow must be feasible.
        
        Even simpler: the minimum number of paths is max(1, |V_odd|/2) where
        V_odd are odd-degree vertices. But we want to actually construct the paths.
        
        Let me use a different approach: min-cost max-flow.
        - Source S, sink T
        - For each vertex u, node u with supply/demand
        - For each undirected edge {u,v}, arcs (u,v) and (v,u) with cap 1, cost 0
        - We want to find a set of paths that covers all edges.
        
        Actually, the cleanest LP:
        Variables: x_{uv} for each directed arc (u,v), binary.
        Constraints:
          1. For each undirected edge {u,v}: x_{uv} + x_{vu} = 1
          2. For each vertex u: outdeg(u) - indeg(u) = b_u, where b_u in {-1,0,1}
          3. Minimize sum_u max(b_u, 0) = number of path starts
        
        Since sum b_u = 0 automatically from constraint 2 (sum of LHS = 0).
        
        Let me just try: fix b_u for odd vertices to be +-1, even to 0.
        Then check if feasible flow exists. The number of paths = |V_odd|/2.
        
        But we need to actually extract the paths from the flow solution.
        """
        try:
            from scipy.optimize import linprog
        except ImportError:
            return None
        
        # Build directed arcs
        arc_idx = {}  # (u,v) -> index
        arcs = []
        for (u, v) in edges:
            arc_idx[(u, v)] = len(arcs)
            arcs.append((u, v))
            arc_idx[(v, u)] = len(arcs)
            arcs.append((v, u))
        
        m = len(arcs)  # number of arcs = 2 * |E|
        
        # Objective: minimize sum of b_u for u where b_u > 0
        # b_u = outdeg(u) - indeg(u) = sum_{v} x_{uv} - sum_{v} x_{vu}
        # We want to minimize sum_u max(b_u, 0)
        # This is non-linear. Instead, introduce auxiliary variables.
        
        # Alternative: since we know the minimum is max(1, |V_odd|/2),
        # let's just set b_u appropriately and solve for feasibility.
        
        degree = {i: 0 for i in range(n)}
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        odd = [v for v in range(n) if degree[v] % 2 == 1]
        
        if not odd:
            # All even: one Eulerian circuit
            b = {i: 0 for i in range(n)}
        else:
            # Assign +1 to half the odd vertices, -1 to the other half
            # To minimize cost, we want to pair them optimally, but for LP
            # feasibility, any assignment works as long as flow exists.
            # Use greedy: first half get +1, second half get -1
            half = len(odd) // 2
            b = {i: 0 for i in range(n)}
            for i in range(half):
                b[odd[i]] = 1
            for i in range(half, len(odd)):
                b[odd[i]] = -1
        
        # LP variables: x_j for each arc j, 0 <= x_j <= 1
        # Constraints:
        # For each vertex u: sum_{v: (u,v) in arcs} x_{uv} - sum_{v: (v,u) in arcs} x_{vu} = b[u]
        # For each undirected edge {u,v}: x_{uv} + x_{vu} = 1
        
        # Build A_eq, b_eq
        A_rows = []
        b_vec = []
        
        # Vertex balance constraints
        for u in range(n):
            row = [0.0] * m
            for v in range(n):
                if (u, v) in arc_idx:
                    row[arc_idx[(u, v)]] = 1.0
                if (v, u) in arc_idx:
                    row[arc_idx[(v, u)]] = -1.0
            A_rows.append(row)
            b_vec.append(float(b[u]))
        
        # Edge usage constraints: for each undirected edge, exactly one direction
        for (u, v) in edges:
            row = [0.0] * m
            row[arc_idx[(u, v)]] = 1.0
            row[arc_idx[(v, u)]] = 1.0
            A_rows.append(row)
            b_vec.append(1.0)
        
        # Objective: minimize 0 (feasibility problem)
        c = [0.0] * m
        
        bounds = [(0, 1)] * m
        
        t0 = time.time()
        result = linprog(c, A_eq=A_rows, b_eq=b_vec, bounds=bounds, method='highs')
        t1 = time.time()
        
        if result.status != 0 or not result.success:
            return None
        
        if t1 - t0 > time_limit:
            return None
        
        # Extract solution: x_j for each arc
        x = result.x
        
        # Reconstruct paths from the flow
        # Build adjacency of used arcs
        used_arcs = []
        for j, (u, v) in enumerate(arcs):
            if x[j] > 0.5:  # binary, so > 0.5 means 1
                used_arcs.append((u, v))
        
        # Now decompose into paths
        # Build in-degree and out-degree from used_arcs
        out_adj = {i: [] for i in range(n)}
        in_count = {i: 0 for i in range(n)}
        out_count = {i: 0 for i in range(n)}
        for u, v in used_arcs:
            out_adj[u].append(v)
            out_count[u] += 1
            in_count[v] += 1
        
        # Path starts: vertices with out_count > in_count (b_u = 1)
        # Path ends: vertices with in_count > out_count (b_u = -1)
        starts = [u for u in range(n) if out_count[u] > in_count[u]]
        ends = [u for u in range(n) if in_count[u] > out_count[u]]
        
        # Trace paths
        paths = []
        for s in starts:
            path = [s]
            cur = s
            while out_count[cur] > in_count[cur] or (out_count[cur] == in_count[cur] and out_adj[cur]):
                # Find next vertex
                if not out_adj[cur]:
                    break
                nxt = out_adj[cur][0]
                out_adj[cur].pop(0)
                path.append(nxt)
                cur = nxt
                # Check if we've reached an end
                if in_count[cur] > out_count[cur]:
                    break
            if len(path) >= 2:
                paths.append(path)
        
        # Verify
        if not _is_valid_paths(paths, edges, n):
            return None
        
        return paths
    
    def _is_valid_paths(paths, edges, n):
        """Check: simple paths, edges exist, edge-disjoint, cover all edges."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        used = set()
        for p in paths:
            if len(p) < 2:
                return False
            seen = set()
            for v in p:
                if v in seen:
                    return False
                seen.add(v)
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i + 1])))
                if e not in edge_set:
                    return False
                if e in used:
                    return False
                used.add(e)
        return used == edge_set
    
    def _eulerian_path_cover(edges, n):
        """Constructive approach: pair odd vertices, add dummy edges, Eulerian circuit, break."""
        degree = {i: 0 for i in range(n)}
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        odd = [v for v in range(n) if degree[v] % 2 == 1]
        
        # Pair odd vertices greedily (consecutive)
        pairs = []
        for i in range(0, len(odd), 2):
            if i + 1 < len(odd):
                pairs.append((odd[i], odd[i + 1]))
        
        # Build augmented graph
        adj = {i: [] for i in range(n)}
        edge_list = []
        eid = 0
        for (u, v) in edges:
            edge_list.append((u, v, eid, False))
            adj[u].append((v, eid))
            adj[v].append((u, eid))
            eid += 1
        for (u, v) in pairs:
            edge_list.append((u, v, eid, True))
            adj[u].append((v, eid))
            adj[v].append((u, eid))
            eid += 1
        
        # Hierholzer
        used = [False] * eid
        stack = [0]
        circuit = []
        while stack:
            v = stack[-1]
            advanced = False
            for idx in range(len(adj[v])):
                nb, e = adj[v][idx]
                if not used[e]:
                    used[e] = True
                    stack.append(nb)
                    advanced = True
                    break
            if not advanced:
                circuit.append(stack.pop())
        circuit.reverse()
        
        # Split at dummy edges
        dummy_set = set()
        for (u, v, e, is_d) in edge_list:
            if is_d:
                dummy_set.add(frozenset((u, v)))
        
        paths = []
        cur = [circuit[0]]
        for i in range(len(circuit) - 1):
            a, b = circuit[i], circuit[i + 1]
            cur.append(b)
            if frozenset((a, b)) in dummy_set:
                if len(cur) >= 2:
                    paths.append(cur)
                cur = [b]
        if len(cur) >= 2:
            paths.append(cur)
        
        if _is_valid_paths(paths, edges, n):
            return paths
        return None
    
    def _greedy_path_cover(edges, n):
        """Greedy fallback: repeatedly find maximal simple paths."""
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        remaining = set(tuple(sorted(e)) for e in edges)
        new_paths = []
        while remaining:
            sub = {i: set() for i in range(n)}
            for (u, v) in remaining:
                sub[u].add(v)
                sub[v].add(u)
            active = [v for v in range(n) if sub[v]]
            if not active:
                break
            start = max(active, key=lambda v: len(sub[v]))
            path = [start]
            visited = {start}
            cur = start
            while True:
                cands = [nb for nb in sub[cur] if nb not in visited]
                if not cands:
                    break
                nxt = min(cands, key=lambda nb: len(sub[nb]))
                path.append(nxt)
                visited.add(nxt)
                cur = nxt
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i + 1])))
                remaining.discard(e)
            new_paths.append(path)
        return new_paths
    
    def path_cover(edges, n):
        if not edges:
            return []
        
        # Try LP first
        lp_result = _lp_path_cover(edges, n, time_limit=20)
        if lp_result is not None:
            return lp_result
        
        # Try Eulerian constructive
        euler_result = _eulerian_path_cover(edges, n)
        if euler_result is not None:
            return euler_result
        
        # Greedy fallback
        return _greedy_path_cover(edges, n)
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            while len(results) < 3:
                edges = instance(*spec)
                results.append([list(e) for e in edges])
            break
        edges = instance(*spec)
        n = spec[0]
        paths = path_cover(edges, n)
        results.append(paths)
    return results
# EVOLVE-BLOCK-END
