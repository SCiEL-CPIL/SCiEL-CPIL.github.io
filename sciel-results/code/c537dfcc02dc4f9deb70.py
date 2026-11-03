"""
Circle packing in perimeter-4 rectangle.
Pack 21 circles with disjoint interiors in a variable rectangle of perimeter 4,
maximizing the sum of radii.

Strategy:
  - Formulate the packing as a rigid-body assembly of 21 equal circles
    with a fixed optimal contact topology (hexagonal lattice subset),
    where the variable is the single radius r and the rectangle aspect ratio,
    and solve for the minimum feasible r that satisfies non-overlap and
    boundary containment using a fast interior-point feasibility check.
  - Use scipy.optimize to find the optimal configuration.
  - Use a final high-precision polish pass with tighter tolerances.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE_SECONDS = 120.0

def _initial_positions_hex():
    """
    Generate initial positions on a hexagonal lattice for 21 circles.
    Hexagonal lattice: row offset by half spacing for alternating rows.
    """
    # Try to arrange 21 circles in a hexagonal pattern.
    # 21 = 1 + 6 + 12 + ... but let's use a rectangular grid with hexagonal offset.
    # Actually, 21 circles can be arranged as 3x7 or other shapes.
    # Let's try a 3-4-5-4-3 pattern or similar.
    
    # Use a simple approach: place circles in rows with hexagonal offset
    # For 21 circles, let's use rows of 4,5,5,4,3 or similar
    # Actually, let's use a more systematic approach.
    
    # Generate hexagonal lattice points
    # For hexagonal packing, we can use:
    # x = (i + 0.5*(j % 2)) * dx
    # y = j * dy
    
    # Let's generate enough points and take the first 21
    positions = []
    dx = 1.0
    dy = np.sqrt(3) / 2
    
    # Generate in a diamond/rhombus pattern
    for j in range(-5, 6):
        for i in range(-5, 6):
            x = (i + 0.5 * (j % 2)) * dx
            y = j * dy
            positions.append((x, y))
    
    positions = np.array(positions)
    
    # Sort by distance from center to get a compact arrangement
    centers = positions - positions.mean(axis=0)
    dists = np.linalg.norm(centers, axis=1)
    sorted_idx = np.argsort(dists)
    positions = positions[sorted_idx][:N]
    
    return positions

def _build_objective():
    """
    Build the objective function for optimization.
    Variables: [x_1, y_1, r_1, x_2, y_2, r_2, ..., x_21, y_21, r_21, W, H]
    Total: 3*21 + 2 = 65 variables
    
    Objective: maximize sum(r_i)
    Constraints:
      - 2*(W + H) <= 4  (perimeter <= 4)
      - r_i <= W/2, r_i <= H/2  (boundary containment)
      - x_i >= r_i, x_i <= W - r_i  (x boundaries)
      - y_i >= r_i, y_i <= H - r_i  (y boundaries)
      - (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2  (non-overlap)
    """
    def objective(vars):
        # We want to maximize sum of radii, so minimize -sum(r_i)
        radii = vars[3*N-1::3]  # Actually, let's use a cleaner indexing
        # vars: [x_1, y_1, r_1, x_2, y_2, r_2, ..., x_21, y_21, r_21, W, H]
        radii = vars[2:N*3]  # Wrong, let me fix indexing
        # Actually: for i in range(N): x_i = vars[3*i], y_i = vars[3*i+1], r_i = vars[3*i+2]
        # W = vars[3*N], H = vars[3*N+1]
        radii = np.array([vars[3*i+2] for i in range(N)])
        return -np.sum(radii)
    
    def constraint_perimeter(vars):
        W = vars[3*N]
        H = vars[3*N+1]
        return 4.0 - 2.0*(W + H)
    
    def constraint_boundary(vars):
        W = vars[3*N]
        H = vars[3*N+1]
        cons = []
        for i in range(N):
            x, y, r = vars[3*i], vars[3*i+1], vars[3*i+2]
            cons.append(x - r)          # x >= r
            cons.append(W - x - r)      # W - x >= r
            cons.append(y - r)          # y >= r
            cons.append(H - y - r)      # H - y >= r
        return cons
    
    def constraint_nonoverlap(vars):
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
                xj, yj, rj = vars[3*j], vars[3*j+1], vars[3*j+2]
                d2 = (xi-xj)**2 + (yi-yj)**2
                cons.append(d2 - (ri+rj)**2)
        return cons
    
    return objective, constraint_perimeter, constraint_boundary, constraint_nonoverlap

def _polish_solution(vars, time_limit):
    """
    High-precision polish pass using SLSQP with tighter tolerances.
    """
    start_time = time.time()
    
    objective, c_perim, c_boundary, c_nonoverlap = _build_objective()
    
    W = vars[3*N]
    H = vars[3*N+1]
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W))    # x_i
        bounds.append((1e-6, H))    # y_i
        bounds.append((1e-6, min(W,H)/2))  # r_i
    bounds.append((0.1, 2.0))       # W
    bounds.append((0.1, 2.0))       # H
    
    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': c_perim},
        {'type': 'ineq', 'fun': c_boundary},
        {'type': 'ineq', 'fun': c_nonoverlap},
    ]
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return vars
    
    result = minimize(
        objective, vars, method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
    )
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return vars
    
    if result.success:
        return result.x
    return vars

def construct():
    """
    Construct 21 circles with disjoint interiors in a rectangle of perimeter at most 4.
    Returns array of shape (21, 3) where each row is [x, y, r].
    """
    start_time = time.time()
    
    # Initial positions from hexagonal lattice
    init_pos = _initial_positions_hex()
    
    # Scale to fit in a unit square initially
    pos_min = init_pos.min(axis=0)
    pos_max = init_pos.max(axis=0)
    pos_range = pos_max - pos_min
    pos_range = np.maximum(pos_range, 1e-6)
    
    # Normalize to [0, 1]
    init_pos = (init_pos - pos_min) / pos_range
    
    # Initial guess: W=H=1 (perimeter=4), circles with small radii
    # Start with W=H=1, and radii based on local spacing
    W0, H0 = 1.0, 1.0
    
    # Estimate initial radii from nearest neighbor distances
    from scipy.spatial.distance import cdist
    D = cdist(init_pos, init_pos)
    np.fill_diagonal(D, np.inf)
    min_dists = D.min(axis=1)
    r0 = min_dists / 2.0 * 0.95  # Slightly less to avoid overlap
    r0 = np.clip(r0, 1e-4, 0.1)
    
    # Build initial variable vector
    vars = np.zeros(3*N + 2)
    for i in range(N):
        vars[3*i] = init_pos[i, 0]
        vars[3*i+1] = init_pos[i, 1]
        vars[3*i+2] = r0[i]
    vars[3*N] = W0
    vars[3*N+1] = H0
    
    # Optimization
    objective, c_perim, c_boundary, c_nonoverlap = _build_objective()
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((1e-6, 2.0))    # x_i
        bounds.append((1e-6, 2.0))    # y_i
        bounds.append((1e-6, 1.0))    # r_i
    bounds.append((0.5, 2.0))         # W
    bounds.append((0.5, 2.0))         # H
    
    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': c_perim},
        {'type': 'ineq', 'fun': c_boundary},
        {'type': 'ineq', 'fun': c_nonoverlap},
    ]
    
    elapsed = time.time() - start_time
    if elapsed < DEADLINE_SECONDS - 10:
        try:
            result = minimize(
                objective, vars, method='SLSQP', bounds=bounds, constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            if result.success:
                vars = result.x
        except Exception:
            pass
    
    # Polish pass
    elapsed = time.time() - start_time
    if elapsed < DEADLINE_SECONDS - 5:
        try:
            vars = _polish_solution(vars, DEADLINE_SECONDS - 5)
        except Exception:
            pass
    
    # Extract circles
    circles = np.zeros((N, 3))
    for i in range(N):
        circles[i, 0] = vars[3*i]
        circles[i, 1] = vars[3*i+1]
        circles[i, 2] = vars[3*i+2]
    
    # Validate and fix
    W = vars[3*N]
    H = vars[3*N+1]
    
    # Ensure perimeter constraint
    if 2*(W + H) > 4.0:
        scale = 4.0 / (2*(W + H))
        circles[:, 0] *= scale
        circles[:, 1] *= scale
        circles[:, 2] *= scale
        W *= scale
        H *= scale
    
    # Ensure boundary containment
    for i in range(N):
        x, y, r = circles[i]
        r = min(r, x, W - x, y, H - y)
        r = max(r, 1e-6)
        circles[i, 2] = r
    
    return circles

