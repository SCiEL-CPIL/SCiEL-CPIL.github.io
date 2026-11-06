import time
from collections import defaultdict
from random import Random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 108  # Safety margin

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_degrees(edges):
        degs = defaultdict(int)
        for u, v in edges:
            degs[u] += 1
            degs[v] += 1
        return degs

    def greedy_decompose(edges, rng, max_attempts=300):
        """Greedy path decomposition with multiple restarts."""
        if not edges:
            return []
            
        degs = get_degrees(edges)
        odd_vertices = [v for v, d in degs.items() if d % 2 != 0]
        K = max(1, len(odd_vertices) // 2)
        
        best_paths = None
        best_count = float('inf')
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            current_adj = defaultdict(list)
            for u, v in edges:
                current_adj[u].append(v)
                current_adj[v].append(u)
                
            remaining_edges = set(edges)
            paths = []
            
            while remaining_edges:
                active_vertices = set()
                for u, v in remaining_edges:
                    active_vertices.add(u)
                    active_vertices.add(v)
                
                if not active_vertices:
                    break
                    
                # Try different start strategies
                if attempt % 3 == 0:
                    # Start from a leaf or low-degree vertex
                    start = min(active_vertices, key=lambda v: len([e for e in remaining_edges if v in e]))
                elif attempt % 3 == 1:
                    # Start from high-degree vertex
                    start = max(active_vertices, key=lambda v: len([e for e in remaining_edges if v in e]))
                else:
                    start = rng.choice(list(active_vertices))
                
                path = [start]
                current = start
                visited = {start}
                
                while True:
                    avail = [nb for nb in current_adj[current] 
                             if tuple(sorted((current, nb))) in remaining_edges 
                             and nb not in visited]
                    
                    if not avail:
                        break
                    
                    # Heuristic: prefer neighbors with higher remaining degree
                    avail.sort(key=lambda x: len([e for e in remaining_edges if x in e]), reverse=True)
                    
                    # Add randomness
                    if len(avail) > 1 and rng.random() < 0.15:
                        next_v = rng.choice(avail[:min(3, len(avail))])
                    else:
                        next_v = avail[0]
                    
                    edge = tuple(sorted((current, next_v)))
                    remaining_edges.remove(edge)
                    current_adj[current].remove(next_v)
                    current_adj[next_v].remove(current)
                    
                    path.append(next_v)
                    visited.add(next_v)
                    current = next_v
                    
                paths.append(path)
                
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = paths
                if best_count == K:
                    break
                    
        if best_paths is None:
            return [list(e) for e in edges]
        return best_paths

    def milp_decompose(edges, time_limit=30):
        """
        Use MILP to find minimum path decomposition.
        Variables: x_{u,v} in {0,1} for each directed edge (u,v) - is this arc used?
        y_v in {0,1} for each vertex - is v a path endpoint (start or end)?
        
        Constraints:
        - Each undirected edge {u,v} is used in exactly one direction: x_{u,v} + x_{v,u} = 1
        - For each vertex v: 
          sum_in(v) - sum_out(v) = 0 if y_v = 0 (v is internal to a path)
          sum_in(v) - sum_out(v) = 1 if y_v is start
          sum_in(v) - sum_out(v) = -1 if y_v is end
          Since we don't distinguish start/end, we use:
          |sum_in(v) - sum_out(v)| <= y_v * 1
          Actually, let's use: sum_in(v) - sum_out(v) = s_v - t_v where s_v, t_v are start/end indicators
          
        Better formulation:
        - x_{u,v} for directed arcs
        - For each vertex v, let in_v = sum_{u} x_{u,v}, out_v = sum_{u} x_{v,u}
        - A vertex is a path endpoint if in_v != out_v
        - Specifically: if in_v > out_v, it's an endpoint (start of path in reverse or end)
        - if out_v > in_v, it's an endpoint
        - if in_v == out_v, it's internal
        
        The number of paths = (number of vertices where in_v != out_v) / 2
        
        Minimize: sum over all vertices of |in_v - out_v| / 2
        Which equals: sum of max(in_v - out_v, 0) over all vertices
        
        Let d_v = in_v - out_v. Then sum of positive d_v = number of paths.
        Minimize sum_v max(d_v, 0)
        
        Linearize: introduce p_v >= 0, p_v >= d_v, minimize sum p_v
        """
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
        except ImportError:
            return None
            
        n = max(max(u, v) for u, v in edges) + 1 if edges else 0
        if n == 0:
            return []
            
        # Build directed arc list
        arcs = []
        arc_to_idx = {}
        for u, v in edges:
            arcs.append((u, v))
            arc_to_idx[(u, v)] = len(arcs) - 1
            arcs.append((v, u))
            arc_to_idx[(v, u)] = len(arcs) - 1
            
        m = len(arcs)  # number of directed arcs
        
        # Variables: x_0...x_{m-1} for arcs, then p_0...p_{n-1} for positive imbalance
        # Total variables: m + n
        total_vars = m + n
        
        # Objective: minimize sum of p_v
        c = np.zeros(total_vars)
        for v in range(n):
            c[m + v] = 1.0  # p_v coefficient
        
        # Constraints:
        # 1. For each undirected edge {u,v}: x_{u,v} + x_{v,u} = 1
        # 2. For each vertex v: p_v >= in_v - out_v, i.e., p_v - in_v + out_v >= 0
        #    i.e., p_v + sum_{u: (u,v) in arcs} x_{u,v} - sum_{u: (v,u) in arcs} x_{v,u} >= 0
        # 3. 0 <= x <= 1, 0 <= p_v
        
        # Build A matrix for linear constraints
        # Constraint 1: equality constraints for each undirected edge
        # Constraint 2: inequality constraints for each vertex
        
        num_undirected = len(edges)
        num_constraints = num_undirected + n
        
        A = np.zeros((num_constraints, total_vars))
        lb = np.zeros(num_constraints)
        ub = np.zeros(num_constraints)
        
        # Constraints 1: x_{u,v} + x_{v,u} = 1 for each undirected edge
        for i, (u, v) in enumerate(edges):
            A[i, arc_to_idx[(u, v)]] = 1
            A[i, arc_to_idx[(v, u)]] = 1
            lb[i] = 1
            ub[i] = 1
            
        # Constraints 2: p_v - in_v + out_v >= 0 for each vertex v
        for v in range(n):
            row = num_undirected + v
            A[row, m + v] = 1.0  # p_v
            for (a, b) in arcs:
                if b == v:  # arc (a,v) contributes to in_v
                    A[row, arc_to_idx[(a, v)]] = -1.0
                if a == v:  # arc (v,b) contributes to out_v
                    A[row, arc_to_idx[(v, b)]] = 1.0
            lb[row] = 0
            ub[row] = np.inf
            
        # Variable bounds
        var_lb = np.zeros(total_vars)
        var_ub = np.ones(total_vars)  # x in [0,1], p in [0, inf] but we cap it
        for v in range(n):
            var_ub[m + v] = len(edges)  # p_v can be at most num_edges
        
        constraints = LinearConstraint(A, lb, ub)
        bounds = Bounds(var_lb, var_ub)
        
        # Integrality: x variables are binary, p variables are continuous
        integrality = np.zeros(total_vars)
        for i in range(m):
            integrality[i] = 1  # binary
        
        # Time limit in seconds
        result = milp(c, constraints=constraints, integrality=integrality, 
                      bounds=bounds, options={'time_limit': time_limit})
        
        if not result.success:
            return None
            
        # Extract solution
        x = result.x[:m]
        
        # Build paths from the directed arcs
        adj = defaultdict(list)
        for i, (u, v) in enumerate(arcs):
            if x[i] > 0.5:
                adj[u].append(v)
                
        # Find all paths
        paths = []
        used_arcs = set()
        
        # Find start vertices (out > in)
        in_deg = np.zeros(n)
        out_deg = np.zeros(n)
        for i, (u, v) in enumerate(arcs):
            if x[i] > 0.5:
                out_deg[u] += 1
                in_deg[v] += 1
                
        starts = [v for v in range(n) if out_deg[v] > in_deg[v]]
        
        for start in starts:
            path = [start]
            current = start
            visited = {start}
            while True:
                found = False
                for nb in adj[current]:
                    if nb not in visited:
                        path.append(nb)
                        visited.add(nb)
                        current = nb
                        found = True
                        break
                if not found:
                    break
            paths.append(path)
        
        # Check if we covered all edges
        if len(paths) == 0:
            return None
            
        return paths

    def decompose_graph(edges):
        if not edges:
            return []
            
        degs = get_degrees(edges)
        odd_vertices = [v for v, d in degs.items() if d % 2 != 0]
        K = max(1, len(odd_vertices) // 2)
        
        if K == 1:
            # Eulerian circuit
            adj = defaultdict(list)
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            start = next(iter(adj))
            path = [start]
            stack = [start]
            remaining = set(edges)
            while stack:
                v = stack[-1]
                next_v = None
                while adj[v]:
                    u = adj[v].pop()
                    e = tuple(sorted((v, u)))
                    if e in remaining:
                        remaining.remove(e)
                        adj[u].remove(v)
                        next_v = u
                        break
                if next_v is not None:
                    stack.append(next_v)
                    path.append(next_v)
                else:
                    stack.pop()
            return [path]
            
        # Try MILP first if graph is small enough
        if len(edges) <= 80 and time.time() < deadline - 40:
            milp_result = milp_decompose(edges, time_limit=min(25, max(5, deadline - time.time() - 35)))
            if milp_result and len(milp_result) >= K:
                # Verify coverage
                used = set()
                valid = True
                for path in milp_result:
                    for i in range(len(path) - 1):
                        e = tuple(sorted((path[i], path[i+1])))
                        if e not in edges or e in used:
                            valid = False
                            break
                        used.add(e)
                    if not valid:
                        break
                if valid and len(used) == len(edges):
                    return milp_result
        
        # Fallback to greedy
        rng = Random(42)
        remaining_time = deadline - time.time()
        max_attempts = max(50, int(remaining_time * 50))
        return greedy_decompose(edges, rng, max_attempts)

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for spec in specs:
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
        
    return results
