import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 20
    # List of all edges (u, v) with u < v
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            edges.append((u, v))
    
    m = len(edges)
    # Map edge to index
    edge_index = {e: i for i, e in enumerate(edges)}
    
    # Auxiliary variables y_{u,v,w} for each pair (u,v) and w not in {u,v}
    # We will index these as a flat array.
    # Let's create a list of (u, v, w) tuples for the auxiliary variables
    aux_vars = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(n):
                if w != u and w != v:
                    aux_vars.append((u, v, w))
    
    k = len(aux_vars)
    
    # Total variables: m (edges) + k (aux)
    total_vars = m + k
    
    # Objective: maximize sum of x_e
    # milp minimizes, so we minimize -sum(x_e)
    c = np.zeros(total_vars)
    for i in range(m):
        c[i] = -1.0
    
    # Integrality: all variables are binary
    integrality = np.ones(total_vars)
    
    # Bounds: all variables in [0, 1]
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    bounds = Bounds(lb, ub)
    
    # Constraints
    # 1. Linearization constraints for y_{u,v,w} <= x_{u,w}
    # 2. Linearization constraints for y_{u,v,w} <= x_{v,w}
    # 3. Linearization constraints for y_{u,v,w} >= x_{u,w} + x_{v,w} - 1
    # 4. Common neighbor constraints: sum_w y_{u,v,w} <= 2
    
    A_rows = []
    lb_rows = []
    ub_rows = []
    
    # We will build the constraint matrix A such that lb <= A @ x <= ub
    
    # Helper to get edge index
    def get_edge_idx(u, v):
        if u > v:
            u, v = v, u
        return edge_index[(u, v)]
    
    # Process each auxiliary variable
    for idx, (u, v, w) in enumerate(aux_vars):
        y_idx = m + idx
        x_uw_idx = get_edge_idx(u, w)
        x_vw_idx = get_edge_idx(v, w)
        
        # y <= x_uw  =>  y - x_uw <= 0
        row = np.zeros(total_vars)
        row[y_idx] = 1.0
        row[x_uw_idx] = -1.0
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(0.0)
        
        # y <= x_vw  =>  y - x_vw <= 0
        row = np.zeros(total_vars)
        row[y_idx] = 1.0
        row[x_vw_idx] = -1.0
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(0.0)
        
        # y >= x_uw + x_vw - 1  =>  -y + x_uw + x_vw <= 1
        row = np.zeros(total_vars)
        row[y_idx] = -1.0
        row[x_uw_idx] = 1.0
        row[x_vw_idx] = 1.0
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(1.0)
        
    # Common neighbor constraints
    # For each pair (u, v), sum_{w not in {u,v}} y_{u,v,w} <= 2
    # Group aux_vars by (u, v)
    from collections import defaultdict
    groups = defaultdict(list)
    for idx, (u, v, w) in enumerate(aux_vars):
        groups[(u, v)].append(m + idx)
        
    for (u, v), y_indices in groups.items():
        row = np.zeros(total_vars)
        for y_idx in y_indices:
            row[y_idx] = 1.0
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(2.0)
        
    A = np.array(A_rows)
    lb_con = np.array(lb_rows)
    ub_con = np.array(ub_rows)
    
    cons = LinearConstraint(A, lb_con, ub_con)
    
    res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
    
    if not res.success:
        # Fallback to a known good construction if ILP fails (e.g. timeout)
        # The seed program gives 30 edges. We can try to return that or a slightly better heuristic.
        # But typically ILP on this size should succeed.
        # Let's return the seed solution as a safe fallback.
        edges_list = []
        for k in range(5):
            edges_list += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges_list

    # Extract solution
    x = res.x[:m]
    # Round to nearest integer to handle floating point precision
    x = np.round(x).astype(int)
    
    final_edges = []
    for i in range(m):
        if x[i] == 1:
            u, v = edges[i]
            final_edges.append([u, v])
            
    return final_edges
