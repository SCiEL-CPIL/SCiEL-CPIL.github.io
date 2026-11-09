"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 50
_DEADLINE = 115.0  # seconds, safety margin before 120s wall-clock kill


def _hex_grid_init(n):
    """Generate a hexagonal lattice initial packing, trimmed to n points."""
    # Try several spacings to get at least n points inside unit disk
    best_pts = None
    for s in np.linspace(0.15, 0.35, 40):
        # Generate hex lattice points
        pts = []
        k = int(1.0 / s) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if x * x + y * y <= (1.0 - s / 2.0) ** 2:
                    pts.append((x, y))
        if len(pts) >= n:
            pts = np.array(pts)
            # Sort by distance from origin, take closest n
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists, kind='stable')[:n]
            if best_pts is None or np.min(dists[idx]) > np.min(np.hypot(best_pts[:, 0], best_pts[:, 1])):
                best_pts = pts[idx]
    if best_pts is not None:
        return best_pts
    # Fallback: random
    return np.random.randn(n, 2) * 0.5


def _score(pts):
    """Compute the packing radius (minimum of min pair dist/2 and min distance to boundary)."""
    n = len(pts)
    # Pairwise distances
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
    min_pair = np.min(d[np.triu_indices(n, k=1)]) if n > 1 else 1.0
    # Distance to boundary
    norms = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - norms)
    return min(min_pair / 2.0, min_boundary)


def _objective_and_constraints(pts, n):
    """
    We maximize r such that:
      |p_i - p_j| >= 2r for all i < j
      |p_i| + r <= 1 for all i
    
    Using L-BFGS-B with variable x = [p_1, ..., p_n, r] (2n+1 vars).
    Objective: minimize -r (i.e., maximize r)
    Constraints (inequality, must be >= 0):
      |p_i - p_j|^2 - (2r)^2 >= 0  -> use soft version or just penalty
      1 - |p_i| - r >= 0
    
    Actually L-BFGS-B handles bounds but not general constraints well.
    Let's use SLSQP which handles inequality constraints.
    """
    pass


def construct():
    t0 = time.time()
    n = N
    
    # Initial hex grid
    pts0 = _hex_grid_init(n)
    
    # We'll use SLSQP to maximize r
    # Variables: x = [x_1, y_1, x_2, y_2, ..., x_n, y_n, r]  (2n + 1 variables)
    
    def unpack(x):
        pts = x[:2*n].reshape(n, 2)
        r = x[2*n]
        return pts, r
    
    def objective(x):
        """Minimize -r (to maximize r)."""
        return -x[2*n]
    
    def constraints(x):
        """All constraints must be >= 0."""
        pts, r = unpack(x)
        cons = []
        # Boundary constraints: 1 - |p_i| - r >= 0
        for i in range(n):
            norm = np.hypot(pts[i, 0], pts[i, 1])
            cons.append(1.0 - norm - r)
        # Pairwise distance constraints: |p_i - p_j|^2 - 4r^2 >= 0
        # Use squared form to avoid sqrt (smoother)
        for i in range(n):
            for j in range(i+1, n):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                cons.append(dx*dx + dy*dy - 4.0*r*r)
        return cons
    
    def jac(x):
        """Jacobian of objective: only r component is -1."""
        g = np.zeros(2*n + 1)
        g[2*n] = -1.0
        return g
    
    def con_jac(x):
        """Jacobian of constraints."""
        pts, r = unpack(x)
        # Number of constraints: n (boundary) + n*(n-1)/2 (pairwise)
        n_boundary = n
        n_pair = n*(n-1)//2
        total_cons = n_boundary + n_pair
        J = np.zeros((total_cons, 2*n + 1))
        
        # Boundary constraints
        for i in range(n):
            xi = pts[i, 0]
            yi = pts[i, 1]
            norm = np.hypot(xi, yi)
            if norm < 1e-12:
                # Gradient wrt xi, yi is 0 (at origin)
                pass
            else:
                J[i, 2*i] = -xi / norm
                J[i, 2*i+1] = -yi / norm
            J[i, 2*n] = -1.0
        
        # Pairwise constraints
        idx = n_boundary
        for i in range(n):
            for j in range(i+1, n):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                # d/dx_i (dx^2 + dy^2 - 4r^2) = 2*dx
                # d/dy_i = 2*dy
                # d/dx_j = -2*dx
                # d/dy_j = -2*dy
                # d/dr = -8*r
                J[idx, 2*i] = 2.0 * dx
                J[idx, 2*i+1] = 2.0 * dy
                J[idx, 2*j] = -2.0 * dx
                J[idx, 2*j+1] = -2.0 * dy
                J[idx, 2*n] = -8.0 * r
                idx += 1
        
        return J
    
    # Initial guess
    r0 = _score(pts0)
    x0 = np.concatenate([pts0.flatten(), [r0]])
    
    # Bounds: each coordinate in [-1, 1], r in [0, 1]
    bounds = [(-1.0, 1.0)] * (2*n) + [(0.0, 1.0)]
    
    best_pts = pts0.copy()
    best_r = r0
    
    # Try SLSQP with multiple initial conditions
    inits = [pts0]
    # Add some perturbed versions
    rng = np.random.RandomState(42)
    for _ in range(5):
        pert = pts0 + rng.randn(n, 2) * 0.02
        inits.append(pert)
    
    for init_pts in inits:
        if time.time() - t0 > _DEADLINE - 5:
            break
        
        r_init = _score(init_pts)
        x_init = np.concatenate([init_pts.flatten(), [r_init]])
        
        try:
            res = minimize(
                objective, x_init, method='SLSQP',
                jac=jac,
                constraints=[{'type': 'ineq', 'fun': lambda x: constraints(x), 'jac': lambda x: con_jac(x)}],
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success or res.fun < -r_init:
                pts_res, r_res = unpack(res.x)
                # Verify feasibility
                actual_r = _score(pts_res)
                if actual_r > best_r - 1e-10:
                    best_pts = pts_res.copy()
                    best_r = actual_r
                    
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() - t0 < _DEADLINE - 10:
        try:
            x_init = np.concatenate([best_pts.flatten(), [best_r]])
            res = minimize(
                objective, x_init, method='SLSQP',
                jac=jac,
                constraints=[{'type': 'ineq', 'fun': lambda x: constraints(x), 'jac': lambda x: con_jac(x)}],
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
            )
            if res.fun < -best_r:
                pts_res, r_res = unpack(res.x)
                actual_r = _score(pts_res)
                if actual_r > best_r:
                    best_pts = pts_res.copy()
                    best_r = actual_r
        except Exception:
            pass
    
    # Verify final result is valid
    final_r = _score(best_pts)
    if final_r <= 0 or not np.all(np.isfinite(best_pts)):
        # Fallback to hex grid
        best_pts = _hex_grid_init(n)
    
    return best_pts
