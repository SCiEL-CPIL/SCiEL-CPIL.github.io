def construct():
    import time
    import random
    import numpy as np
    from scipy.optimize import linprog
    
    t0 = time.time()
    DEADLINE = 135.0
    
    def instance_graph(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def solve_lp_orientation(edges, n):
        """Solve LP to find optimal orientation minimizing sum |b(v)|."""
        m = len(edges)
        if m == 0:
            return [1] * m
        
        # Map edges to indices
        edge_idx = {e: i for i, e in enumerate(edges)}
        
        # For each vertex, compute the linear expression for b(v) in terms of x
        # b(v) = sum_{w>v, {v,w} in E} (2*x_{vw} - 1) + sum_{u<v, {u,v} in E} (1 - 2*x_{uv})
        #     = 2*sum_{w>v} x_{vw} - count_{w>v} + count_{u<v} - 2*sum_{u<v} x_{uv}
        
        # Variables: x_0..x_{m-1} (for edges), then b^+_0..b^+_{n-1}, b^-_0..b^-_{n-1}
        # Total variables: m + 2*n
        
        # Objective: minimize sum_v (b^+(v) + b^-(v))
        c = np.zeros(m + 2*n)
        for v in range(n):
            c[m + v] = 1.0      # b^+(v)
            c[m + n + v] = 1.0  # b^-(v)
        
        # Constraints: for each vertex v: b^+(v) - b^-(v) = expr_v(x)
        # Rearranged: b^+(v) - b^-(v) - expr_v(x) = 0
        # In standard form for linprog (inequalities A_ub @ x <= b_ub and equalities A_eq @ x == b_eq):
        
        A_eq = np.zeros((n, m + 2*n))
        b_eq = np.zeros(n)
        
        for v in range(n):
            # expr_v(x) = 2*sum_{w>v} x_{vw} - count_{w>v} + count_{u<v} - 2*sum_{u<v} x_{uv}
            coeff = np.zeros(m)
            const = 0.0
            for i, (u, w) in enumerate(edges):
                if u == v:
                    # edge {v, w} with w > v (since edges are sorted, u < w)
                    # x_i = 1 means v->w, contributes +1 to b(v)
                    # x_i = 0 means w->v, contributes -1 to b(v)
                    # contribution: 2*x_i - 1
                    coeff[i] += 2.0
                    const += -1.0
                elif w == v:
                    # edge {u, v} with u < v
                    # x_i = 1 means u->v, contributes -1 to b(v)
                    # x_i = 0 means v->u, contributes +1 to b(v)
                    # contribution: 1 - 2*x_i
                    coeff[i] += -2.0
                    const += 1.0
            
            # b^+(v) - b^-(v) = expr_v(x) = coeff @ x + const
            # b^+(v) - b^-(v) - coeff @ x = const
            # In A_eq form: -coeff @ x + b^+(v) - b^-(v) = const
            A_eq[v, :m] = -coeff
            A_eq[v, m + v] = 1.0    # b^+(v)
            A_eq[v, m + n + v] = -1.0  # b^-(v)
            b_eq[v] = const
        
        # Bounds: x_i in [0,1], b^+ >= 0, b^- >= 0
        bounds = [(0, 1)] * m + [(0, None)] * (2*n)
        
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if result.success:
            x = result.x[:m]
            # Round to get orientation
            orientation = [1 if xi >= 0.5 else -1 for xi in x]
            return orientation
        else:
            return [1] * m
    
    def count_paths_fast(edges, orientation):
        """Fast count of paths from orientation."""
        n = max(max(e) for e in edges) + 1 if edges else 0
        bal = [0] * n
        for i, (u, v) in enumerate(edges):
            if orientation[i] == 1:
                bal[u] += 1
                bal[v] -= 1
            else:
                bal[v] += 1
                bal[u] -= 1
        return sum(abs(b) for b in bal) // 2
    
    def local_search(edges, n, orientation, max_iter=100000):
        """Fast local search with O(1) flip updates."""
        m = len(edges)
        # Precompute for each edge, the two endpoints
        bal = [0] * n
        for i, (u, v) in enumerate(edges):
            if orientation[i] == 1:
                bal[u] += 1
                bal[v] -= 1
            else:
                bal[v] += 1
                bal[u] -= 1
        current_cost = sum(abs(b) for b in bal)
        
        for _ in range(max_iter):
            improved = False
            for i in range(m):
                u, v = edges[i]
                # Try flipping edge i
                if orientation[i] == 1:
                    # Currently u->v: bal[u]+=1, bal[v]-=1
                    # After flip: v->u: bal[u]-=1, bal[v]+=1
                    # Change: bal[u] -= 2, bal[v] += 2
                    old_abs_u = abs(bal[u])
                    old_abs_v = abs(bal[v])
                    new_abs_u = abs(bal[u] - 2)
                    new_abs_v = abs(bal[v] + 2)
                    delta = new_abs_u + new_abs_v - old_abs_u - old_abs_v
                    if delta < 0:
                        bal[u] -= 2
                        bal[v] += 2
                        orientation[i] = -1
                        current_cost += delta
                        improved = True
                        break
                else:
                    # Currently v->u: bal[v]+=1, bal[u]-=1
                    # After flip: u->v: bal[u]+=1, bal[v]-=1
                    # Change: bal[u] += 2, bal[v] -= 2
                    old_abs_u = abs(bal[u])
                    old_abs_v = abs(bal[v])
                    new_abs_u = abs(bal[u] + 2)
                    new_abs_v = abs(bal[v] - 2)
                    delta = new_abs_u + new_abs_v - old_abs_u - old_abs_v
                    if delta < 0:
                        bal[u] += 2
                        bal[v] -= 2
                        orientation[i] = 1
                        current_cost += delta
                        improved = True
                        break
            if not improved:
                break
        
        return orientation, current_cost // 2
    
    def two_opt_search(edges, n, orientation, max_iter=50000):
        """2-opt: try flipping pairs of edges."""
        m = len(edges)
        bal = [0] * n
        for i, (u, v) in enumerate(edges):
            if orientation[i] == 1:
                bal[u] += 1
                bal[v] -= 1
            else:
                bal[v] += 1
                bal[u] -= 1
        current_cost = sum(abs(b) for b in bal)
        
        rng = random.Random(42)
        
        for _ in range(max_iter):
            improved = False
            # Try random pairs
            for _ in range(50):
                i1 = rng.randint(0, m - 1)
                i2 = rng.randint(0, m - 1)
                if i1 == i2:
                    continue
                u1, v1 = edges[i1]
                u2, v2 = edges[i2]
                
                # Compute delta for flipping both
                old_cost = current_cost
                # Flip edge 1
                if orientation[i1] == 1:
                    bal[u1] -= 2
                    bal[v1] += 2
                else:
                    bal[u1] += 2
                    bal[v1] -= 2
                # Flip edge 2
                if orientation[i2] == 1:
                    bal[u2] -= 2
                    bal[v2] += 2
                else:
                    bal[u2] += 2
                    bal[v2] -= 2
                
                new_cost = sum(abs(b) for b in bal)
                
                if new_cost < current_cost:
                    orientation[i1] *= -1
                    orientation[i2] *= -1
                    current_cost = new_cost
                    improved = True
                    break
                else:
                    # Undo
                    if orientation[i1] == 1:
                        bal[u1] += 2
                        bal[v1] -= 2
                    else:
                        bal[u1] -= 2
                        bal[v1] += 2
                    if orientation[i2] == 1:
                        bal[u2] += 2
                        bal[v2] -= 2
                    else:
                        bal[u2] -= 2
                        bal[v2] += 2
            
            if not improved:
                # Try single flips
                for i in range(m):
                    u, v = edges[i]
                    if orientation[i] == 1:
                        old_abs_u = abs(bal[u])
                        old_abs_v = abs(bal[v])
                        new_abs_u = abs(bal[u] - 2)
                        new_abs_v = abs(bal[v] + 2)
                        delta = new_abs_u + new_abs_v - old_abs_u - old_abs_v
                        if delta < 0:
                            bal[u] -= 2
                            bal[v] += 2
                            orientation[i] = -1
                            current_cost += delta
                            improved = True
                            break
                    else:
                        old_abs_u = abs(bal[u])
                        old_abs_v = abs(bal[v])
                        new_abs_u = abs(bal[u] + 2)
                        new_abs_v = abs(bal[v] - 2)
                        delta = new_abs_u + new_abs_v - old_abs_u - old_abs_v
                        if delta < 0:
                            bal[u] += 2
                            bal[v] -= 2
                            orientation[i] = 1
                            current_cost += delta
                            improved = True
                            break
                if not improved:
                    break
        
        return orientation, current_cost // 2
    
    def reconstruct_paths(edges, orientation, n):
        """Reconstruct paths from orientation."""
        out_adj = [[] for _ in range(n)]
        in_adj = [[] for _ in range(n)]
        
        for i, (u, v) in enumerate(edges):
            if orientation[i] == 1:
                out_adj[u].append(v)
                in_adj[v].append(u)
            else:
                out_adj[v].append(u)
                in_adj[u].append(v)
        
        starts = []
        for v in range(n):
            if len(out_adj[v]) > len(in_adj[v]):
                starts.append(v)
        
        paths = []
        visited_edges = set()
        
        for start in starts:
            if not out_adj[start]:
                continue
            path = [start]
            current = start
            while out_adj[current]:
                next_v = None
                for v in out_adj[current]:
                    edge_key = (min(current, v), max(current, v))
                    if edge_key not in visited_edges:
                        next_v = v
                        break
                if next_v is None:
                    break
                visited_edges.add((min(current, next_v), max(current, next_v)))
                path.append(next_v)
                current = next_v
            if len(path) > 1:
                paths.append(path)
        
        # Handle remaining edges
        for i, (u, v) in enumerate(edges):
            if (u, v) not in visited_edges:
                paths.append([u, v])
        
        return paths
    
    def solve_instance(n, p, seed):
        edges = instance_graph(n, p, seed)
        if not edges:
            return []
        
        m = len(edges)
        
        # Try LP first
        best_orientation = None
        best_count = m
        
        try:
            orientation = solve_lp_orientation(edges, n)
            count = count_paths_fast(edges, orientation)
            if count < best_count:
                best_count = count
                best_orientation = orientation[:]
        except Exception:
            pass
        
        # Local search from LP solution or random
        if best_orientation is None:
            best_orientation = [1] * m
        
        orientation, count = local_search(edges, n, best_orientation[:], max_iter=100000)
        if count < best_count:
            best_count = count
            best_orientation = orientation[:]
        
        if time.time() > DEADLINE - 30:
            return reconstruct_paths(edges, best_orientation, n)
        
        # 2-opt
        orientation, count = two_opt_search(edges, n, best_orientation[:], max_iter=50000)
        if count < best_count:
            best_count = count
            best_orientation = orientation[:]
        
        return reconstruct_paths(edges, best_orientation, n)
    
    results = []
    for n, p, seed in [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]:
        if time.time() > DEADLINE:
            # Emergency fallback
            edges = instance_graph(n, p, seed)
            if not edges:
                results.append([])
                continue
            results.append([[u, v] for u, v in edges])
        else:
            results.append(solve_instance(n, p, seed))
    
    return results
