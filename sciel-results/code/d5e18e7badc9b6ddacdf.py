"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

def _make_initial_guess(n, seed=42):
    """Create a well-separated initial guess using a jittered grid."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= n:
                break
            x = (i + 0.5) / side + rng.uniform(-0.05, 0.05)
            y = (j + 0.5) / side + rng.uniform(-0.05, 0.05)
            # Clamp to [0,1] with small margin
            x = np.clip(x, 0.02, 0.98)
            y = np.clip(y, 0.02, 0.98)
            pts.append((x, y))
    
    # If we don't have enough, add random points
    while len(pts) < n:
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        pts.append((x, y))
    
    pts = pts[:n]
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    
    # Initial radii: small but non-zero
    radii = np.full(n, 0.05)
    
    return xs, ys, radii


def _optimize_packing(n, seed=42, time_limit=100.0):
    """
    Optimize circle packing using SLSQP with hard constraints.
    Variables: [x_1, ..., x_n, y_1, ..., y_n, r_1, ..., r_n]
    Maximize sum(r_i)
    """
    start_time = time.time()
    
    xs, ys, radii = _make_initial_guess(n, seed)
    
    # Initial variable vector
    x0 = np.concatenate([xs, ys, radii])
    
    def objective(v):
        """Minimize negative sum of radii."""
        if time.time() - start_time > time_limit:
            return 1e10  # Penalize if over time
        r = v[2*n:]
        return -np.sum(r)
    
    def constraints_func(v):
        """Return constraint violations (should be >= 0 for feasibility)."""
        x = v[:n]
        y = v[n:2*n]
        r = v[2*n:]
        
        # 1. Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        #    i.e., x_i - r_i >= 0, 1 - x_i - r_i >= 0, y_i - r_i >= 0, 1 - y_i - r_i >= 0
        c_boundary = np.concatenate([
            x - r,
            1 - x - r,
            y - r,
            1 - y - r
        ])
        
        # 2. Non-overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        #    i.e., (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        c_overlap = []
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                c_overlap.append(dx*dx + dy*dy - (r[i] + r[j])**2)
        c_overlap = np.array(c_overlap)
        
        # 3. Positive radius: r_i >= 0
        c_radius = r
        
        return np.concatenate([c_boundary, c_overlap, c_radius])
    
    # Create constraint dictionary for SLSQP
    # SLSQP expects constraints as: {'type': 'ineq', 'fun': lambda v: ...} where fun(v) >= 0
    # But for efficiency, we use a single vector constraint
    
    # Actually, SLSQP's 'ineq' constraint type expects fun(v) >= 0
    # Let's use a single constraint function that returns the vector
    
    constraints = ({'type': 'ineq', 'fun': constraints_func})
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x_i
    for i in range(n):
        bounds.append((0.0, 1.0))  # y_i
    for i in range(n):
        bounds.append((0.0, 0.5))  # r_i
    
    # Run optimization
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
    )
    
    # Check if we finished in time
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        # Use the last feasible point if available
        if result.success or result.status == 0:
            pass
        # Try to extract best feasible solution from result
        v = result.x
    
    # Ensure feasibility by clipping
    x = np.clip(v[:n], 0.0, 1.0)
    y = np.clip(v[n:2*n], 0.0, 1.0)
    r = np.clip(v[2*n:], 0.0, 0.5)
    
    # Verify and fix any remaining violations
    # Re-check constraints and reduce radii if needed
    for i in range(n):
        # Boundary fix
        r[i] = min(r[i], x[i], 1-x[i], y[i], 1-y[i])
        r[i] = max(r[i], 0.0)
    
    # Check pairwise overlaps and reduce radii
    for it in range(10):
        fixed = False
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j]:
                    # Reduce the larger radius
                    if r[i] > r[j]:
                        r[i] = max(0, (dist - r[j]) * 0.99)
                    else:
                        r[j] = max(0, (dist - r[i]) * 0.99)
                    fixed = True
        if not fixed:
            break
    
    # Final boundary check
    for i in range(n):
        r[i] = min(r[i], x[i], 1-x[i], y[i], 1-y[i])
        r[i] = max(r[i], 0.0)
    
    return x, y, r


def construct():
    """
    Pack 26 circles with disjoint interiors inside [0,1]^2 and maximize the sum of their radii.
    Returns: array of shape (n, 3): rows (x, y, radius) of n disjoint circles inside the unit square
    """
    deadline = time.time() + 140.0  # Safety margin
    
    best_x = None
    best_y = None
    best_r = None
    best_sum = -1e10
    
    # Try multiple seeds
    for seed in range(10):
        if time.time() > deadline:
            break
        
        try:
            x, y, r = _optimize_packing(N, seed=seed, time_limit=12.0)
            s = np.sum(r)
            if s > best_sum:
                best_sum = s
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()
        except Exception:
            continue
    
    # If no optimization succeeded, fall back to grid
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result
    
    result = np.column_stack([best_x, best_y, best_r])
    return result
# EVOLVE-BLOCK-END
