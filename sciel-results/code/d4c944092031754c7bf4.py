import numpy as np
import time
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import random

def _is_valid_vectorized(centers, sides, tol=1e-7):
    """
    Vectorized validity check.
    centers: (N, 2)
    sides: (N,)
    """
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
        
    # Boundary check
    if np.any(centers[:, 0] - sides/2 < -tol): return False
    if np.any(centers[:, 0] + sides/2 > 1 + tol): return False
    if np.any(centers[:, 1] - sides/2 < -tol): return False
    if np.any(centers[:, 1] + sides/2 > 1 + tol): return False
    
    # Overlap check
    dx = np.abs(centers[:, 0][:, np.newaxis] - centers[:, 0][np.newaxis, :])
    dy = np.abs(centers[:, 1][:, np.newaxis] - centers[:, 1][np.newaxis, :])
    sep = (sides[:, np.newaxis] + sides[np.newaxis, :]) / 2.0 - tol
    mask = ~np.eye(n, dtype=bool)
    
    if np.any((dx < sep) & (dy < sep) & mask):
        return False
        
    return True

def _initial_packing_grid():
    """
    Returns the known good packing with sum 3.0.
    2 squares of 0.5, 8 squares of 0.25.
    """
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.75, 0.25]
    sides[1] = 0.5
    
    for k in range(8):
        row = k // 4
        col = k % 4
        cy = 0.625 + row * 0.25
        cx = 0.125 + col * 0.25
        centers[2+k] = [cx, cy]
        sides[2+k] = 0.25
        
    return centers, sides

def _get_topology(centers, sides):
    """
    Extracts a valid topology from a given packing.
    Returns a list of relations for each pair (i, j) with i < j.
    Relation: 0 (left), 1 (right), 2 (below), 3 (above).
    """
    n = len(sides)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    relations = []
    for i, j in pairs:
        cx_i, cy_i = centers[i]
        cx_j, cy_j = centers[j]
        s_i, s_j = sides[i], sides[j]
        
        # Check which non-overlap condition is satisfied.
        # We need to pick one that is true.
        # If they touch, multiple might be true. We pick the first one found.
        
        # i left of j: cx_i + s_i/2 <= cx_j - s_j/2
        if cx_i + s_i/2 <= cx_j - s_j/2 + 1e-9:
            relations.append(0)
        # i right of j: cx_i - s_i/2 >= cx_j + s_j/2
        elif cx_i - s_i/2 >= cx_j + s_j/2 - 1e-9:
            relations.append(1)
        # i below j: cy_i + s_i/2 <= cy_j - s_j/2
        elif cy_i + s_i/2 <= cy_j - s_j/2 + 1e-9:
            relations.append(2)
        # i above j: cy_i - s_i/2 >= cy_j + s_j/2
        elif cy_i - s_i/2 >= cy_j + s_j/2 - 1e-9:
            relations.append(3)
        else:
            # Should not happen if packing is valid.
            # Default to left
            relations.append(0)
            
    return relations

def _solve_milp(n=10, deadline=None):
    """
    Solves the MILP for the packing problem.
    Returns (centers, sides, value) or None if infeasible or timeout.
    """
    # Variables:
    # 0..n-1: cx_i
    # n..2n-1: cy_i
    # 2n..3n-1: s_i
    # 3n..: binary variables for each pair (i,j), 4 per pair
    
    n_pairs = n * (n - 1) // 2
    n_bin = 4 * n_pairs
    n_vars = 3 * n + n_bin
    
    # Objective: maximize sum(s_i) => minimize -sum(s_i)
    c = np.zeros(n_vars)
    for i in range(n):
        c[2*n + i] = -1.0
        
    # Bounds
    # cx, cy in [0, 1]
    # s in [0.01, 1.0]
    # bin in [0, 1]
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    # Set bounds for s
    lb[2*n:3*n] = 0.01
    ub[2*n:3*n] = 1.0
    
    # Integrality
    integrality = np.zeros(n_vars)
    integrality[3*n:] = 1
    
    # Constraints
    # 1. Boundary constraints
    # cx_i - s_i/2 >= 0  => -cx_i + 0.5*s_i <= 0
    # cx_i + s_i/2 <= 1
    # Same for y
    
    A_rows = []
    lb_vals = []
    ub_vals = []
    
    for i in range(n):
        # -cx_i + 0.5*s_i <= 0
        row = np.zeros(n_vars)
        row[i] = -1.0
        row[2*n + i] = 0.5
        A_rows.append(row)
        lb_vals.append(-np.inf)
        ub_vals.append(0.0)
        
        # cx_i + 0.5*s_i <= 1
        row = np.zeros(n_vars)
        row[i] = 1.0
        row[2*n + i] = 0.5
        A_rows.append(row)
        lb_vals.append(-np.inf)
        ub_vals.append(1.0)
        
        # -cy_i + 0.5*s_i <= 0
        row = np.zeros(n_vars)
        row[n + i] = -1.0
        row[2*n + i] = 0.5
        A_rows.append(row)
        lb_vals.append(-np.inf)
        ub_vals.append(0.0)
        
        # cy_i + 0.5*s_i <= 1
        row = np.zeros(n_vars)
        row[n + i] = 1.0
        row[2*n + i] = 0.5
        A_rows.append(row)
        lb_vals.append(-np.inf)
        ub_vals.append(1.0)
        
    # 2. Topology constraints
    # For each pair (i, j), we have 4 binary variables b0, b1, b2, b3
    # b0: i left of j
    # b1: i right of j
    # b2: i below j
    # b3: i above j
    
    # Sum of binaries for each pair must be 1
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    for k, (i, j) in enumerate(pairs):
        base_idx = 3*n + 4*k
        
        # Sum of 4 binaries = 1
        row = np.zeros(n_vars)
        row[base_idx:base_idx+4] = 1.0
        A_rows.append(row)
        lb_vals.append(1.0)
        ub_vals.append(1.0)
        
        # Constraints for each relation
        # b0 * (cx_i + s_i/2 - cx_j + s_j/2) <= 0
        # This is nonlinear. We linearize it using big-M.
        # cx_i - cx_j + 0.5*s_i + 0.5*s_j <= M * (1 - b0)
        # M is an upper bound on the left side.
        # Max cx_i - cx_j is 1. Max 0.5*s_i + 0.5*s_j is 1. So M = 2.
        
        # b0: i left of j => cx_i + s_i/2 <= cx_j - s_j/2 + M*(1-b0)
        # cx_i - cx_j + 0.5*s_i + 0.5*s_j <= M*(1-b0)
        M = 2.0 # Big M
        
        # b0
        row = np.zeros(n_vars)
        row[i] = 1.0
        row[j] = -1.0
        row[2*n + i] = 0.5
        row[2*n + j] = 0.5
        row[base_idx] = -M
        A_rows.append(row)
        lb_vals.append(-np.inf)
        ub_vals.append(0.0)
        
        # b1: i right of j => cx_i - s_i/2 >= cx_j + s_j/2 - M*(1-b1)
        # -cx_i + cx_j - 0.5*s_i - 0.5*s_j <= M*(1-b1)
        row = np.zeros(n_vars)
        row[i] = -1.0
        row[j] = 1.0
        row[2*n + i] = -0.5
        row[2*n + j] = -0.5
        row[base_idx+1] = -M
        A_rows.append(row)
        lb_vals.append(-np.inf)
        ub_vals.append(0.0)
        
        # b2: i below j => cy_i + s_i/2 <= cy_j - s_j/2 + M*(1-b2)
        # cy_i - cy_j + 0.5*s_i + 0.5*s_j <= M*(1-b2)
        row = np.zeros(n_vars)
        row[n + i] = 1.0
        row[n + j] = -1.0
        row[2*n + i] = 0.5
        row[2*n + j] = 0.5
        row[base_idx+2] = -M
        A_rows.append(row)
        lb_vals.append(-np.inf)
        ub_vals.append(0.0)
        
        # b3: i above j => cy_i - s_i/2 >= cy_j + s_j/2 - M*(1-b3)
        # -cy_i + cy_j - 0.5*s_i - 0.5*s_j <= M*(1-b3)
        row = np.zeros(n_vars)
        row[n + i] = -1.0
        row[n + j] = 1.0
        row[2*n + i] = -0.5
        row[2*n + j] = -0.5
        row[base_idx+3] = -M
        A_rows.append(row)
        lb_vals.append(-np.inf)
        ub_vals.append(0.0)
        
    A = np.array(A_rows)
    lb_arr = np.array(lb_vals)
    ub_arr = np.array(ub_vals)
    
    constraints = LinearConstraint(A, lb_arr, ub_arr)
    bounds = Bounds(lb, ub)
    
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, options={"time_limit": 600})
    except Exception:
        return None
        
    if not res.success:
        # Check if it timed out but found a feasible solution
        if res.x is not None and np.all(np.isfinite(res.x)):
             # We might still have a solution if it's feasible
             pass
        else:
            return None
            
    if res.x is None:
        return None
        
    x = res.x
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    for i in range(n):
        centers[i, 0] = x[i]
        centers[i, 1] = x[n + i]
        sides[i] = x[2*n + i]
        
    val = np.sum(sides)
    return centers, sides, val

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 620
    
    # Initial best
    best_centers, best_sides = _initial_packing_grid()
    best_val = np.sum(best_sides)
    
    # Try MILP first
    milp_res = _solve_milp(n, deadline)
    if milp_res is not None:
        c_new, s_new, val_new = milp_res
        if _is_valid_vectorized(c_new, s_new, tol=tol):
            if val_new > best_val:
                best_val = val_new
                best_centers = c_new
                best_sides = s_new
    
    # If we have time left, we could try to improve with SA on topologies
    # But MILP is likely to find the global optimum or very close to it.
    # Let's just return the best found so far.
    # If MILP failed or timed out, we fall back to the initial grid or previous best.
        
    # Final check
    if not _is_valid_vectorized(best_centers, best_sides, tol=tol):
        # Fallback
        best_centers, best_sides = _initial_packing_grid()
        best_val = 3.0
        
    return best_centers, best_sides
