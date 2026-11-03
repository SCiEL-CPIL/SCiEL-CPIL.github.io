"""Circle packing in unit square: pack 26 disjoint circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
DEADLINE = 140.0  # safety margin before 150s kill

def _feasible_solution():
    """Return a valid initial packing (grid of equal circles)."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-6])
    cells = cells[:N]
    return np.array(cells)

def _objective_and_constraints(x, n):
    """
    x: flattened array of length 3*n: [x0,y0,r0, x1,y1,r1, ...]
    Returns (objective, constraint_dict) for SLSQP.
    Objective: negative sum of radii (we minimize).
    Constraints:
      - Each circle inside [0,1]^2: x - r >= 0, y - r >= 0, x + r <= 1, y + r <= 1
      - Pairwise non-overlap: dist(i,j) >= ri + rj
    """
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    
    obj = -np.sum(rs)
    
    # Wall constraints (inequalities: g >= 0)
    wall_cons = []
    for i in range(n):
        wall_cons.append(xs[i] - rs[i])
        wall_cons.append(ys[i] - rs[i])
        wall_cons.append(1.0 - xs[i] - rs[i])
        wall_cons.append(1.0 - ys[i] - rs[i])
    
    # Pairwise non-overlap
    pair_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            # dist >= ri + rj  =>  dist - ri - rj >= 0
            # To avoid sqrt issues when dist is very small, use squared form? 
            # SLSQP can handle sqrt. But let's use the direct form.
            # Actually, for better conditioning, we can use:
            # (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
            # This is a quadratic constraint. SLSQP handles nonlinear constraints.
            # Let's use the squared distance form to avoid derivative issues at dist=0.
            constraint_val = (dx * dx + dy * dy) - (rs[i] + rs[j]) ** 2
            pair_cons.append(constraint_val)
    
    constraints = list(wall_cons) + list(pair_cons)
    return obj, constraints

def _run_optimization(x0, max_time=130.0):
    """Run SLSQP optimization with time limit."""
    n = len(x0) // 3
    start_time = time.time()
    
    # Bounds: x, y in [0,1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((1e-8, 0.5))  # r
    
    # We need to build the constraint dict for SLSQP
    # SLSQP expects constraints as list of dicts or a single dict with 'fun' returning array
    
    def constraint_fun(x):
        _, cons = _objective_and_constraints(x, n)
        return np.array(cons)
    
    def objective_fun(x):
        obj, _ = _objective_and_constraints(x, n)
        return obj
    
    constraints = [{'type': 'ineq', 'fun': constraint_fun}]
    
    # Run SLSQP
    try:
        result = minimize(
            objective_fun,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
        )
    except Exception:
        pass
    
    return result.x if hasattr(result, 'x') else x0

def construct():
    t_start = time.time()
    
    # Start with a valid grid solution
    x0 = _feasible_solution().flatten()
    best_x = x0.copy()
    best_val = -np.sum(best_x[2::3])  # negative sum of radii
    
    # Try multiple random initializations and local optimization
    rng = np.random.default_rng(42)
    
    n_init = 5  # number of different starting points
    
    for k in range(n_init):
        if time.time() - t_start > DEADLINE:
            break
        
        # Create a perturbed initial solution
        if k == 0:
            x_curr = x0.copy()
        else:
            # Random valid starting point: jitter the grid
            x_curr = x0.copy()
            # Add small random perturbations to positions
            for i in range(N):
                idx_x = 3 * i
                idx_y = 3 * i + 1
                idx_r = 3 * i + 2
                x_curr[idx_x] += rng.uniform(-0.05, 0.05)
                x_curr[idx_y] += rng.uniform(-0.05, 0.05)
                # Keep radius slightly smaller initially to ensure feasibility
                x_curr[idx_r] = x_curr[idx_r] * rng.uniform(0.9, 1.0)
            # Clip to bounds
            for i in range(N):
                idx_x = 3 * i
                idx_y = 3 * i + 1
                idx_r = 3 * i + 2
                x_curr[idx_x] = np.clip(x_curr[idx_x], 0.01, 0.99)
                x_curr[idx_y] = np.clip(x_curr[idx_y], 0.01, 0.99)
                x_curr[idx_r] = np.clip(x_curr[idx_r], 1e-8, 0.49)
        
        # Run optimization
        result_x = _run_optimization(x_curr, max_time=DEADLINE - (time.time() - t_start))
        
        # Check if result is better
        val = -np.sum(result_x[2::3])
        if val < best_val:
            best_val = val
            best_x = result_x.copy()
    
    # Convert back to (x, y, r) array
    result = best_x.reshape(N, 3)
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Sum of radii: {np.sum(res[:, 2]):.6f}")
