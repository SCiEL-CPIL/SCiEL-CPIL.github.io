"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 15
    # All possible edges in a simple undirected graph on n vertices
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_edges)
    
    # Map edge to index
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute paths s-u-v-t
    # A path is defined by an edge (u, v) and neighbors s of u (s!=v) and t of v (t!=u) with s!=t.
    # In the MILP, we don't know the neighbors yet, so we consider all potential (s, u, v, t) tuples
    # such that (s,u), (u,v), (v,t) are potential edges, s!=t, s!=v, t!=u.
    
    # We need to enforce deg(s) != deg(t) for all such paths that exist in the final graph.
    # This is tricky because the existence of the path depends on the edges (s,u) and (v,t).
    # If (s,u) is not in the graph, the path doesn't exist.
    # So the constraint is: IF x_{s,u}=1 AND x_{u,v}=1 AND x_{v,t}=1 AND s!=t, THEN deg(s) != deg(t).
    
    # deg(i) = sum_{j != i} x_{i,j}
    
    # Linearization of "IF A=1 AND B=1 AND C=1 THEN deg(s) < deg(t) OR deg(t) < deg(s)"
    # This is a disjunction. We can use a binary variable z_{s,t,u,v} to select the direction.
    # z=0 -> deg(s) < deg(t)
    # z=1 -> deg(t) < deg(s)
    
    # Constraints for deg(s) < deg(t):
    # deg(s) <= deg(t) - 1 + M * (1 - z)
    # deg(t) >= deg(s) + 1 - M * z  (This is redundant if we just use the first one with z? No.)
    
    # Actually, standard big-M for "if condition then (A < B OR B < A)" is complex.
    # Let's use:
    # deg(s) - deg(t) <= M * (1 - z) - 1 + (M+1) * w  ... this is getting messy.
    
    # Simpler approach for "deg(s) != deg(t)" under condition C:
    # deg(s) - deg(t) <= M * (1 - C) - 1 + (M+1) * z
    # deg(t) - deg(s) <= M * (1 - C) - 1 + (M+1) * (1 - z)
    
    # Where C = x_{s,u} * x_{u,v} * x_{v,t}. This product of binaries is not linear.
    # We need to linearize C.
    # Let c_{s,u,v,t} be a binary variable such that c=1 iff x_{s,u}=1 and x_{u,v}=1 and x_{v,t}=1.
    # c <= x_{s,u}, c <= x_{u,v}, c <= x_{v,t}
    # c >= x_{s,u} + x_{u,v} + x_{v,t} - 2
    
    # Then the inequality constraints become:
    # deg(s) - deg(t) <= M * (1 - c) - 1 + (M+1) * z
    # deg(t) - deg(s) <= M * (1 - c) - 1 + (M+1) * (1 - z)
    
    # M can be n (max degree is 14).
    
    # This creates a huge number of variables and constraints.
    # Number of potential paths:
    # For each edge (u,v), ~13 choices for s, ~13 for t. Total ~ 105 * 13 * 13 ~ 17000 paths.
    # Each path introduces 1 c-variable, 1 z-variable, and 4-5 constraints.
    # Total variables: 105 (edges) + 17000 (c) + 17000 (z) ~ 34000 variables.
    # This might be too slow for a standard LP solver in 55s if not optimized.
    # However, scipy.optimize.milp is available.
    
    # Let's check if scipy is available and try to build the MILP.
    # If it's too slow, we fall back to a heuristic.
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Variables:
        # 0..m-1: x_e (edge indicators)
        # m..m+P-1: c_p (path existence indicators)
        # m+P..m+2P-1: z_p (direction indicators)
        
        # Identify all potential paths (s, u, v, t)
        # s~u, u~v, v~t, s!=t, s!=v, t!=u
        # Note: s and t are vertices, u and v are vertices.
        # Edges are (s,u), (u,v), (v,t).
        
        potential_paths = []
        for u in range(n):
            for v in range(u + 1, n):
                edge_uv_idx = edge_to_idx[(u, v)]
                # s neighbors of u (excluding v)
                for s in range(n):
                    if s == u or s == v: continue
                    edge_su_idx = edge_to_idx[(min(s,u), max(s,u))]
                    for t in range(n):
                        if t == u or t == v or t == s: continue
                        edge_vt_idx = edge_to_idx[(min(v,t), max(v,t))]
                        potential_paths.append((s, u, v, t, edge_su_idx, edge_uv_idx, edge_vt_idx))
        
        P = len(potential_paths)
        num_vars = m + 2 * P
        
        # Objective: Maximize sum(x_e)
        # min -sum(x_e)
        c_obj = np.zeros(num_vars)
        for i in range(m):
            c_obj[i] = -1.0
        
        # Constraints:
        # 1. Path existence linearization:
        # c_p <= x_su, c_p <= x_uv, c_p <= x_vt
        # c_p >= x_su + x_uv + x_vt - 2
        
        # 2. Degree inequality:
        # deg(s) - deg(t) <= M * (1 - c_p) - 1 + (M+1) * z_p
        # deg(t) - deg(s) <= M * (1 - c_p) - 1 + (M+1) * (1 - z_p)
        # M = 14 (max degree)
        
        M = 14
        
        # We will build the constraint matrix A_ub and b_ub.
        # This might be large. P ~ 17000. Rows ~ 4 * P + 2 * P = 6 * P ~ 100,000.
        # Columns ~ 34,000.
        # Dense matrix is 100k * 34k = 3.4 billion entries. Too big for dense.
        # scipy.optimize.milp uses HiGHS which supports sparse matrices.
        
        from scipy.sparse import lil_matrix
        
        A = lil_matrix((6 * P, num_vars))
        b = np.zeros(6 * P)
        
        row = 0
        for idx, (s, u, v, t, e_su, e_uv, e_vt) in enumerate(potential_paths):
            c_idx = m + idx
            z_idx = m + P + idx
            
            # c_p <= x_su
            A[row, c_idx] = 1
            A[row, e_su] = -1
            b[row] = 0
            row += 1
            
            # c_p <= x_uv
            A[row, c_idx] = 1
            A[row, e_uv] = -1
            b[row] = 0
            row += 1
            
            # c_p <= x_vt
            A[row, c_idx] = 1
            A[row, e_vt] = -1
            b[row] = 0
            row += 1
            
            # c_p >= x_su + x_uv + x_vt - 2  =>  -c_p + x_su + x_uv + x_vt <= 2
            A[row, c_idx] = -1
            A[row, e_su] = 1
            A[row, e_uv] = 1
            A[row, e_vt] = 1
            b[row] = 2
            row += 1
            
            # deg(s) - deg(t) <= M * (1 - c) - 1 + (M+1) * z
            # deg(s) - deg(t) + M*c - (M+1)*z <= M - 1
            # deg(s) = sum_{j!=s} x_{s,j}
            # deg(t) = sum_{j!=t} x_{t,j}
            
            # This row will have non-zeros for all edges incident to s or t, plus c and z.
            
            # Coeff for c: M
            A[row, c_idx] = M
            
            # Coeff for z: -(M+1)
            A[row, z_idx] = -(M + 1)
            
            # Coeff for edges
            for j in range(n):
                if j == s: continue
                e_idx = edge_to_idx[(min(s, j), max(s, j))]
                A[row, e_idx] += 1 # +1 for deg(s)
                
                if j == t: continue
                e_idx_t = edge_to_idx[(min(t, j), max(t, j))]
                A[row, e_idx_t] -= 1 # -1 for deg(t)
            
            b[row] = M - 1
            row += 1
            
            # deg(t) - deg(s) <= M * (1 - c) - 1 + (M+1) * (1 - z)
            # deg(t) - deg(s) + M*c - (M+1)*(1-z) <= M - 1
            # deg(t) - deg(s) + M*c + (M+1)*z <= 2*M
            
            A[row, c_idx] = M
            A[row, z_idx] = M + 1
            
            for j in range(n):
                if j == s: continue
                e_idx = edge_to_idx[(min(s, j), max(s, j))]
                A[row, e_idx] -= 1 # -1 for deg(s)
                
                if j == t: continue
                e_idx_t = edge_to_idx[(min(t, j), max(t, j))]
                A[row, e_idx_t] += 1 # +1 for deg(t)
            
            b[row] = 2 * M
            row += 1
            
            if time.time() > deadline - 5:
                # Fallback to heuristic if construction takes too long
                raise TimeoutError("MILP construction too slow")
        
        A = A.tocsr()
        
        # Bounds
        bounds = Bounds(lb=np.zeros(num_vars), ub=np.ones(num_vars))
        
        # Integrality
        integrality = np.ones(num_vars)
        
        res = milp(c=c_obj, constraints=LinearConstraint(A, -np.inf, b), bounds=bounds, integrality=integrality)
        
        if res.success:
            x_vals = res.x[:m]
            edges = []
            for i in range(m):
                if x_vals[i] > 0.5:
                    edges.append(list(all_edges[i]))
            return edges
        else:
            # Fallback
            pass
            
    except Exception as e:
        pass
        
    # Fallback Heuristic: Greedy + Local Search
    # This is the same logic as the original but faster
    
    def get_degrees_and_adj(edges):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        return deg, adj

    def count_violations(edges):
        if not edges:
            return 0
        deg, adj = get_degrees_and_adj(edges)
        violations = 0
        for u in range(n):
            for v in adj[u]:
                deg_s_counts = {}
                for s in adj[u]:
                    if s != v:
                        d = deg[s]
                        deg_s_counts[d] = deg_s_counts.get(d, 0) + 1
                if not deg_s_counts:
                    continue
                for t in adj[v]:
                    if t != u:
                        dt = deg[t]
                        if dt in deg_s_counts:
                            violations += deg_s_counts[dt]
        return violations

    def is_valid(edges):
        return count_violations(edges) == 0

    best_edges = []
    
    # Greedy initialization
    cur_edges = []
    cur_set = set()
    
    # Try to add edges in a random order
    shuffled_edges = all_edges[:]
    random.shuffle(shuffled_edges)
    
    for e in shuffled_edges:
        if time.time() > deadline - 5:
            break
        new_edges = cur_edges + [e]
        if is_valid(new_edges):
            cur_edges = new_edges
            cur_set.add(e)
            
    best_edges = cur_edges[:]
    
    # Local Search: Try to swap or add
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Try adding missing edges
        missing = [e for e in all_edges if e not in cur_set]
        random.shuffle(missing)
        for e in missing:
            if time.time() > deadline - 2:
                break
            new_edges = cur_edges + [e]
            if is_valid(new_edges):
                cur_edges = new_edges
                cur_set.add(e)
                improved = True
                if len(cur_edges) > len(best_edges):
                    best_edges = cur_edges[:]
        
        # Try swaps
        if not improved:
            # Simple swap: remove one, add one
            # We'll just do a few random swaps
            for _ in range(100):
                if time.time() > deadline - 2:
                    break
                if not cur_edges:
                    break
                e_remove = random.choice(cur_edges)
                missing_now = [e for e in all_edges if e not in cur_set]
                if not missing_now:
                    break
                e_add = random.choice(missing_now)
                
                new_set = set(cur_edges)
                new_set.remove(e_remove)
                new_set.add(e_add)
                new_edges_list = list(new_set)
                
                if is_valid(new_edges_list):
                    # Accept swap if it leads to same size (neutral) or if we can add more later?
                    # For now, accept neutral swaps to explore.
                    # But we want to maximize size. So only accept if size increases?
                    # Swap keeps size same. So it's for escaping local optima.
                    cur_edges = new_edges_list
                    cur_set = new_set
                    improved = True
                    break # Restart loop to try adding again
                    
    # Final polish
    final_missing = [e for e in all_edges if e not in cur_set]
    random.shuffle(final_missing)
    for e in final_missing:
        if time.time() > deadline - 1:
            break
        new_edges = cur_edges + [e]
        if is_valid(new_edges):
            cur_edges = new_edges
            cur_set.add(e)
            if len(cur_edges) > len(best_edges):
                best_edges = cur_edges[:]

    return [list(e) for e in best_edges]

# EVOLVE-BLOCK-END
