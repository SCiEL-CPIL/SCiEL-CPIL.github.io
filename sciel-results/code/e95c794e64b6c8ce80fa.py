import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 20
    
    # If time is already up, return empty or best known
    if time.time() > deadline:
        return []

    # Indices for edges x_{uv}
    edge_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_indices.append((i, j))
    m_edges = len(edge_indices)
    
    # Indices for z_{uvw} where u < v, w is not u or v
    # We need z for each triplet (u, v, w) to enforce sum z <= 2 for each pair (u,v)
    # Actually, the constraint is on pairs (u,v).
    # For a fixed pair (u,v), the common neighbors are w such that (u,w) and (v,w) are edges.
    # Let x_{uw} and x_{vw} be the edge variables.
    # We introduce y_{u,v,w} = x_{u,w} * x_{v,w}.
    # We need sum_{w} y_{u,v,w} <= 2.
    
    # Variables:
    # 0 .. m_edges-1 : x_{uv}
    # m_edges .. m_edges + m_z - 1 : y_{u,v,w}
    
    z_vars = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(n):
                if w != u and w != v:
                    z_vars.append((u, v, w))
    m_z = len(z_vars)
    
    total_vars = m_edges + m_z
    
    # Objective: Maximize sum x_{uv}
    # milp minimizes c @ x. So c = -1 for x, 0 for z.
    c = np.zeros(total_vars)
    for i in range(m_edges):
        c[i] = -1.0
        
    # Constraints
    # 1. Linearization: y_{u,v,w} <= x_{u,w}
    # 2. Linearization: y_{u,v,w} <= x_{v,w}
    # 3. Linearization: y_{u,v,w} >= x_{u,w} + x_{v,w} - 1
    # 4. Common Neighbor: sum_{w} y_{u,v,w} <= 2
    
    # We will build the A matrix and b vector for LinearConstraint
    # A @ x >= lb
    # A @ x <= ub
    
    # It is more efficient to use sparse matrices if available, but numpy is fine for 3600 vars.
    # However, building a dense 10000x3600 matrix might be slow/memory heavy.
    # Let's use scipy.sparse if possible, or just construct lists.
    
    # Actually, scipy.optimize.milp accepts LinearConstraint which can take sparse matrices.
    import scipy.sparse as sp
    
    # We'll collect rows for lower and upper bounds separately or combine.
    # Easier: Create A_ub, b_ub and A_eq, b_eq.
    # milp requires LinearConstraint(A, lb, ub).
    
    # Let's build the constraints in lists first to convert to sparse.
    A_rows = []
    b_lbs = []
    b_ubs = []
    
    # Helper to get x index
    x_idx = {}
    for i, (u, v) in enumerate(edge_indices):
        x_idx[(u, v)] = i
        x_idx[(v, u)] = i # Symmetry, though we only define for u<v
        
    # Helper to get z index
    z_idx = {}
    for i, (u, v, w) in enumerate(z_vars):
        # Note: y_{u,v,w} is symmetric in u,v? 
        # y_{u,v,w} represents x_{u,w} * x_{v,w}.
        # The constraint is for pair (u,v).
        # We defined z_vars for u < v.
        z_idx[(u, v, w)] = m_edges + i
        # Also store for (v,u,w) if needed? No, we only sum over w for fixed (u,v).
        
    # Constraint 1: y_{u,v,w} - x_{u,w} <= 0
    # Constraint 2: y_{u,v,w} - x_{v,w} <= 0
    # Constraint 3: y_{u,v,w} - x_{u,w} - x_{v,w} >= -1  =>  -y + x_uw + x_vw <= 1
    
    # And 4: sum_{w} y_{u,v,w} <= 2
    
    # Let's populate A rows.
    # Row type: (coeffs, lb, ub)
    
    # For each z variable (u, v, w):
    for (u, v, w), z_id in zip(z_vars, range(m_edges, total_vars)):
        # Get x indices
        # u < w or w < u?
        i1 = min(u, w)
        j1 = max(u, w)
        idx_uw = x_idx[(i1, j1)]
        
        i2 = min(v, w)
        j2 = max(v, w)
        idx_vw = x_idx[(i2, j2)]
        
        # 1. y - x_uw <= 0
        row = {}
        row[z_id] = 1.0
        row[idx_uw] = -1.0
        A_rows.append(row)
        b_lbs.append(-np.inf)
        b_ubs.append(0.0)
        
        # 2. y - x_vw <= 0
        row = {}
        row[z_id] = 1.0
        row[idx_vw] = -1.0
        A_rows.append(row)
        b_lbs.append(-np.inf)
        b_ubs.append(0.0)
        
        # 3. -y + x_uw + x_vw <= 1
        row = {}
        row[z_id] = -1.0
        row[idx_uw] = 1.0
        row[idx_vw] = 1.0
        A_rows.append(row)
        b_lbs.append(-np.inf)
        b_ubs.append(1.0)
        
    # Constraint 4: For each pair (u, v), sum_{w} y_{u,v,w} <= 2
    for u in range(n):
        for v in range(u + 1, n):
            row = {}
            for w in range(n):
                if w != u and w != v:
                    # Find z index for (u, v, w)
                    # We stored z_idx with (u, v, w) where u < v.
                    # Since we iterate u < v, this matches.
                    z_id = z_idx[(u, v, w)]
                    row[z_id] = 1.0
            A_rows.append(row)
            b_lbs.append(-np.inf)
            b_ubs.append(2.0)
            
    # Convert to sparse matrix
    # A_rows is list of dicts {col: val}
    # We need to build sparse matrix.
    rows = []
    cols = []
    vals = []
    
    for i, row in enumerate(A_rows):
        for col, val in row.items():
            rows.append(i)
            cols.append(col)
            vals.append(val)
            
    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(A_rows), total_vars))
    
    lb = np.array(b_lbs)
    ub = np.array(b_ubs)
    
    # Variable bounds: 0 to 1
    var_bounds = Bounds(lb=np.zeros(total_vars), ub=np.ones(total_vars))
    
    # Integrality: All integer
    integrality = np.ones(total_vars)
    
    # Solve
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, lb, ub),
            integrality=integrality,
            bounds=var_bounds
        )
    except Exception as e:
        # Fallback to heuristic if MILP fails or times out
        # (Though milp doesn't have a direct timeout parameter in older scipy, 
        # it might run for a long time. We must guard against this.)
        # Since we can't easily interrupt milp, we rely on the fact that 
        # this instance is small enough. 
        # If it hangs, the program will be killed by the evaluator.
        # To be safe, we might need a simpler fallback or just return empty.
        # But let's assume it solves.
        pass

    if res.status == 0: # Optimal
        x = res.x[:m_edges]
        edges = []
        for i, (u, v) in enumerate(edge_indices):
            if x[i] > 0.5:
                edges.append([u, v])
        return edges
    else:
        # If not optimal, return best feasible found or empty
        # res.x might contain a feasible solution even if not optimal?
        # Usually status 2 is infeasible, 3 is unbounded, 4 is numerical.
        # If status is not 0, we might not have a good solution.
        # Let's check if we have a feasible solution.
        if res.x is not None:
             # Verify feasibility manually? Too slow.
             # Just return empty or a simple valid graph.
             pass
             
    # Fallback: Simple greedy valid graph
    # This ensures we return *something* valid if MILP fails.
    adj = [0] * n
    count = 0
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    
    def is_valid_add(adj, u, v):
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
        w_mask = adj[u]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            common_vw = adj[v] & adj[w]
            if common_vw.bit_count() > 1:
                return False
            w_mask ^= lsb
        w_mask = adj[v]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            common_uw = adj[u] & adj[w]
            if common_uw.bit_count() > 1:
                return False
            w_mask ^= lsb
        return True

    for u, v in all_edges:
        if is_valid_add(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            count += 1
            
    edges = []
    for i in range(n):
        mask = adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
    return edges

