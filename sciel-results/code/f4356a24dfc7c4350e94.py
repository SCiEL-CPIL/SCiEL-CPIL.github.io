import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 20
    # Indices for edges u < v
    edge_indices = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m_edges = len(edge_indices) # 190
    
    # Map (u, v) with u < v to an index 0..m_edges-1
    edge_idx = {}
    for i, (u, v) in enumerate(edge_indices):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i # Symmetry helper, though we only use u<v in vars

    # Variables:
    # 0 .. m_edges-1 : A_u_v (1 if edge exists)
    # m_edges .. m_edges + m_z - 1 : Z_u_v_w (1 if w is common neighbor of u, v)
    
    # Generate Z indices
    # For each pair (u, v) with u < v, and each w != u, v
    z_indices = []
    z_idx_map = {} # (u, v, w) -> index
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(n):
                if w != u and w != v:
                    z_indices.append((u, v, w))
                    z_idx_map[(u, v, w)] = len(z_indices) - 1
    
    m_z = len(z_indices)
    total_vars = m_edges + m_z
    
    # Objective: Maximize sum of A_u_v
    # milp minimizes c @ x, so we use -1 for A variables, 0 for Z variables
    c = np.zeros(total_vars)
    c[:m_edges] = -1.0
    
    # Bounds: all variables are in [0, 1]
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    bounds = Bounds(lb, ub)
    
    # Integrality: all variables are integers
    integrality = np.ones(total_vars)
    
    # Constraints
    A_cons = []
    lb_cons = []
    ub_cons = []
    
    # 1. z_u_v_w <= A_u_w
    # 2. z_u_v_w <= A_v_w
    # 3. z_u_v_w >= A_u_w + A_v_w - 1  =>  -z_u_v_w <= -A_u_w - A_v_w + 1
    # 4. sum_w z_u_v_w <= 2
    
    for (u, v, w) in z_indices:
        idx_z = m_edges + z_idx_map[(u, v, w)]
        
        # Get edge indices for (u, w) and (v, w)
        # Ensure u < w and v < w for lookup
        idx_uw = edge_idx[(min(u, w), max(u, w))]
        idx_vw = edge_idx[(min(v, w), max(v, w))]
        
        # Constraint 1: z - A_uw <= 0
        row = np.zeros(total_vars)
        row[idx_z] = 1.0
        row[idx_uw] = -1.0
        A_cons.append(row)
        lb_cons.append(-np.inf)
        ub_cons.append(0.0)
        
        # Constraint 2: z - A_vw <= 0
        row = np.zeros(total_vars)
        row[idx_z] = 1.0
        row[idx_vw] = -1.0
        A_cons.append(row)
        lb_cons.append(-np.inf)
        ub_cons.append(0.0)
        
        # Constraint 3: -z + A_uw + A_vw >= 1  =>  -z + A_uw + A_vw >= 1
        # Or: z - A_uw - A_vw <= -1
        row = np.zeros(total_vars)
        row[idx_z] = 1.0
        row[idx_uw] = -1.0
        row[idx_vw] = -1.0
        A_cons.append(row)
        lb_cons.append(-np.inf)
        ub_cons.append(-1.0)
        
    # Constraint 4: For each pair (u, v), sum of z_u_v_w <= 2
    for u in range(n):
        for v in range(u + 1, n):
            row = np.zeros(total_vars)
            for w in range(n):
                if w != u and w != v:
                    idx_z = m_edges + z_idx_map[(u, v, w)]
                    row[idx_z] = 1.0
            A_cons.append(row)
            lb_cons.append(-np.inf)
            ub_cons.append(2.0)
            
    A_cons = np.array(A_cons)
    lb_cons = np.array(lb_cons)
    ub_cons = np.array(ub_cons)
    
    constraints = LinearConstraint(A_cons, lb_cons, ub_cons)
    
    # Solve
    res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if not res.success:
        # Fallback to seed if MILP fails (should not happen)
        A = np.zeros((n, n), dtype=int)
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    A[u, v] = 1
                    A[v, u] = 1
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v] == 1:
                    edges.append([u, v])
        return edges

    x = res.x
    # Extract edges
    edges = []
    for i, (u, v) in enumerate(edge_indices):
        if x[i] > 0.5:
            edges.append([u, v])
            
    return edges
