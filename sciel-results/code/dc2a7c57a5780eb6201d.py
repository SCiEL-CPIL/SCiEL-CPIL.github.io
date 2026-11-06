"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 9

# EVOLVE-BLOCK-START

def _greedy_seed():
    """Greedy snake: always extend with the lowest bit flip that keeps the path induced."""
    path, used = [0], {0}
    while True:
        cur, nxt = path[-1], None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                continue
            nxt = v
            break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)


def _validate_snake(path):
    """Check that path is a valid induced path in Q_N."""
    n = len(path)
    if n < 2:
        return n - 1
    # Check distinct
    if len(set(path)) != n:
        return -1
    # Check consecutive differ by exactly 1 bit
    for i in range(n - 1):
        if (path[i] ^ path[i+1]).bit_count() != 1:
            return -1
    # Check non-consecutive differ by at least 2 bits
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return -1
    return n - 1


def _milp_solve(deadline):
    """
    Solve the snake-in-the-box problem as a MILP using scipy.optimize.milp.
    
    Variables:
      x_v for v in [0, 2^N): 1 if vertex v is included in the snake
      y_uv for each edge (u,v) in Q_N: 1 if edge is in the path
    
    Objective: maximize sum(y_uv) = number of edges
    
    Constraints:
      1. Degree: for each vertex v, sum of incident y <= 2*x_v
         (endpoints have degree 1, internal have degree 2)
      2. Connectivity/subtour elimination: MTZ constraints
         For each vertex v, assign a position p_v in [0, L-1] where L = max possible length
         p_v <= p_u + L*(1 - y_uv) for each directed edge (u->v)
         p_v >= p_u + 1 - L*(1 - y_uv) for each directed edge (u->v)
    
    This is a large MILP (512 vertices, 2304 edges), so we use time limits.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    import sys
    
    nV = 1 << N  # 512 vertices
    nE = N * (1 << (N-1))  # 2304 edges
    
    # Build edge list
    edges = []
    for v in range(nV):
        for b in range(N):
            u = v ^ (1 << b)
            if v < u:
                edges.append((v, u))
    nE = len(edges)
    
    # Variable indexing:
    # x[0..nV-1]: vertex inclusion
    # x[nV..nV+nE-1]: edge selection
    # x[nV+nE..nV+nE+nV-1]: position variables (MTZ)
    
    total_vars = nV + nE + nV
    L = nV - 1  # max position value
    
    # Objective: maximize sum of edge variables -> minimize -sum
    c = np.zeros(total_vars)
    c[nV:nV+nE] = -1.0  # maximize edges
    
    # Bounds
    lb = np.zeros(total_vars)
    ub = np.zeros(total_vars)
    ub[:nV] = 1.0       # x_v in {0,1}
    ub[nV:nV+nE] = 1.0  # y_e in {0,1}
    ub[nV+nE:] = float(L)  # p_v in [0, L]
    
    # Integer constraints
    integrality = np.ones(total_vars)
    integrality[nV+nE:] = 0  # positions can be continuous
    
    constraints = []
    
    # Degree constraints: for each vertex v, sum of incident edges <= 2 * x_v
    # i.e., sum(incident y) - 2*x_v <= 0
    # And also >= 0 (at least 0 edges, which is trivial)
    
    # Build incidence
    vertex_edges = [[] for _ in range(nV)]
    for ei, (u, v) in enumerate(edges):
        vertex_edges[u].append(ei)
        vertex_edges[v].append(ei)
    
    # Degree constraint: for each v, sum_{e incident to v} y_e - 2*x_v <= 0
    for v in range(nV):
        row = np.zeros(total_vars)
        row[v] = -2.0  # -2 * x_v
        for ei in vertex_edges[v]:
            row[nV + ei] = 1.0
        constraints.append(LinearConstraint(row, -np.inf, 0))
    
    # Also: if a vertex is included, it must have at least 1 edge (unless it's an endpoint with degree 1)
    # Actually for a path: endpoints have degree 1, internal have degree 2.
    # Sum of degrees = 2 * (#edges), and #vertices = #edges + 1
    # So we need: for each v, sum(incident y) >= x_v (at least 1 edge if included)
    for v in range(nV):
        row = np.zeros(total_vars)
        row[v] = -1.0  # -x_v
        for ei in vertex_edges[v]:
            row[nV + ei] = 1.0
        constraints.append(LinearConstraint(row, 0, np.inf))
    
    # MTZ subtour elimination constraints:
    # For each edge (u,v): p_u - p_v + L * y_uv <= L - 1  (if y_uv=1, then p_u <= p_v + L - 1... wait)
    # Standard MTZ: for each directed arc (i,j): p_i - p_j + M*(1-y_ij) <= 0
    # i.e., p_i <= p_j if y_ij = 1, and p_i <= p_j + M if y_ij = 0
    # But we need: if edge (u,v) is in path, then positions differ by exactly 1.
    # This is tricky with undirected edges.
    
    # Alternative: use the fact that in a path, for each edge (u,v), |p_u - p_v| = 1
    # We can linearize: p_u - p_v <= 1 - (1-y_e)*L and p_v - p_u <= 1 - (1-y_e)*L
    # i.e., if y_e = 1: |p_u - p_v| <= 1
    #       if y_e = 0: |p_u - p_v| <= L (trivially true)
    
    for ei, (u, v) in enumerate(edges):
        # p_u - p_v <= 1 - L*(1 - y_e) = 1 - L + L*y_e
        row = np.zeros(total_vars)
        row[nV+nE+u] = 1.0
        row[nV+nE+v] = -1.0
        row[nV+ei] = L
        constraints.append(LinearConstraint(row, -np.inf, 1 - L))
        
        # p_v - p_u <= 1 - L*(1 - y_e) = 1 - L + L*y_e
        row = np.zeros(total_vars)
        row[nV+nE+u] = -1.0
        row[nV+nE+v] = 1.0
        row[nV+ei] = L
        constraints.append(LinearConstraint(row, -np.inf, 1 - L))
    
    # Time limit
    elapsed = time.time()
    time_limit = max(1, int(deadline - elapsed - 5))  # 5s safety margin
    if time_limit < 1:
        return None
    
    try:
        res = milp(c, constraints=constraints, integrality=integrality, 
                   bounds=Bounds(lb, ub), options={"time_limit": time_limit, "mip_gap": 0.0})
        
        if res.status == 0 or res.status == 1:  # optimal or best solution found
            x = res.x
            # Extract path from edge variables
            edge_vars = x[nV:nV+nE]
            selected_edges = [edges[ei] for ei in range(nE) if edge_vars[ei] > 0.5]
            
            if not selected_edges:
                return None
            
            # Build adjacency from selected edges
            adj = {}
            for u, v in selected_edges:
                adj.setdefault(u, []).append(v)
                adj.setdefault(v, []).append(u)
            
            # Find an endpoint (degree 1)
            start = None
            for v, nbrs in adj.items():
                if len(nbrs) == 1:
                    start = v
                    break
            if start is None:
                return None
            
            # Walk the path
            path = [start]
            visited = {start}
            while True:
                cur = path[-1]
                nxt = None
                for nb in adj.get(cur, []):
                    if nb not in visited:
                        nxt = nb
                        break
                if nxt is None:
                    break
                path.append(nxt)
                visited.add(nxt)
            
            # Validate
            if _validate_snake(path) == len(path) - 1:
                return path
    except Exception:
        pass
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    # First, get the greedy seed as fallback
    seed_path = _greedy_seed()
    best_path = seed_path
    best_score = len(seed_path) - 1
    
    # Try MILP with time limit
    try:
        result = _milp_solve(deadline)
        if result is not None:
            score = _validate_snake(result)
            if score > best_score:
                best_path = result
                best_score = score
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        return best_path
    
    # If MILP didn't improve, try a local search refinement on the seed
    # Simple: try to extend the path by trying different starting points
    # For now, return best found
    return best_path

# EVOLVE-BLOCK-END
