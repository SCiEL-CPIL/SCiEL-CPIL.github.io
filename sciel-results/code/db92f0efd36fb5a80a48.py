"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    def instance(n, d, seed):
        from random import Random
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def strong_color_ilp(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        
        adj_set = set()
        for u, nbrs in adj.items():
            for v in nbrs:
                adj_set.add((u, v))
                adj_set.add((v, u))
        
        # Build conflict graph
        edge_adj = [set() for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    edge_adj[i].add(j)
                    edge_adj[j].add(i)
                    continue
                if (u1, u2) in adj_set or (u1, v2) in adj_set or (v1, u2) in adj_set or (v1, v2) in adj_set:
                    edge_adj[i].add(j)
                    edge_adj[j].add(i)
        
        # Get an upper bound using greedy DSATUR
        def greedy_dsatur():
            color = [-1] * m
            neighbor_colors = [set() for _ in range(m)]
            uncolored = list(range(m))
            num_colors = 0
            
            while uncolored:
                best_v = None
                best_sat = -1
                best_deg = -1
                for v in uncolored:
                    sat = len(neighbor_colors[v])
                    deg = len(edge_adj[v])
                    if sat > best_sat or (sat == best_sat and deg > best_deg):
                        best_sat = sat
                        best_deg = deg
                        best_v = v
                
                used = neighbor_colors[best_v]
                c = 0
                while c in used:
                    c += 1
                color[best_v] = c
                if c + 1 > num_colors:
                    num_colors = c + 1
                
                for nb in edge_adj[best_v]:
                    if nb in uncolored:
                        neighbor_colors[nb].add(c)
                
                uncolored.remove(best_v)
            return color, num_colors
        
        greedy_sol, ub = greedy_dsatur()
        
        # Try ILP with upper bound
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np
            
            c_max = ub
            n_vars = m * c_max
            n_z = c_max
            
            # Objective: minimize sum z_c
            # Variables: x_{e,c} (m*c_max) then z_c (c_max)
            c_obj = np.zeros(n_vars + n_z)
            c_obj[m * c_max:] = 1.0
            
            # Constraints:
            # 1. sum_c x_{e,c} = 1 for each e
            # 2. x_{e,c} + x_{f,c} <= 1 for each conflict (e,f) and c
            # 3. x_{e,c} <= z_c for each e, c
            
            A_rows = []
            b_lower = []
            b_upper = []
            
            # Constraint 1: Assignment
            for e in range(m):
                row = np.zeros(n_vars + n_z)
                for c in range(c_max):
                    row[e * c_max + c] = 1.0
                A_rows.append(row)
                b_lower.append(1.0)
                b_upper.append(1.0)
            
            # Constraint 2: Conflicts
            for e in range(m):
                for f in edge_adj[e]:
                    if e < f:
                        for c in range(c_max):
                            row = np.zeros(n_vars + n_z)
                            row[e * c_max + c] = 1.0
                            row[f * c_max + c] = 1.0
                            A_rows.append(row)
                            b_lower.append(0.0)
                            b_upper.append(1.0)
            
            # Constraint 3: Link z
            for e in range(m):
                for c in range(c_max):
                    row = np.zeros(n_vars + n_z)
                    row[e * c_max + c] = 1.0
                    row[m * c_max + c] = -1.0
                    A_rows.append(row)
                    b_lower.append(0.0)
                    b_upper.append(np.inf)
            
            A = np.array(A_rows)
            b_low = np.array(b_lower)
            b_up = np.array(b_upper)
            
            # Bounds
            integrality = np.ones(n_vars + n_z)
            lb = np.zeros(n_vars + n_z)
            ub_bounds = np.ones(n_vars + n_z)
            
            bounds = Bounds(lb, ub_bounds)
            cons = LinearConstraint(A, b_low, b_up)
            
            res = milp(c_obj, constraints=cons, integrality=integrality, bounds=bounds)
            
            if res.success and res.x is not None:
                x = res.x[:m * c_max].reshape(m, c_max)
                ilp_color = []
                for e in range(m):
                    found = False
                    for c in range(c_max):
                        if x[e, c] > 0.5:
                            ilp_color.append(c)
                            found = True
                            break
                    if not found:
                        ilp_color.append(greedy_sol[e])
                return ilp_color
            else:
                return greedy_sol
        except Exception:
            return greedy_sol

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        colors = strong_color_ilp(edges, adj)
        out.append(colors)
    return out
# EVOLVE-BLOCK-END
