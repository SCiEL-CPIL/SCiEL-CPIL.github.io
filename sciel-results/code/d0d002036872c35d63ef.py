"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

DEADLINE = 150.0
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _is_feasible(circles, tol=1e-7):
    """Check if circles are disjoint and inside unit square."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True


def _objective(params):
    """Maximize sum of radii = minimize negative sum of radii."""
    # params: [x1, y1, r1, x2, y2, r2, ...]
    radii = params[2::3]
    return -np.sum(radii)


def _boundary_constraints(params):
    """Each circle must be inside [0,1]^2: x >= r, y >= r, x + r <= 1, y + r <= 1."""
    n = len(params) // 3
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    # x - r >= 0, y - r >= 0, 1 - x - r >= 0, 1 - y - r >= 0
    return np.concatenate([xs - rs, ys - rs, 1.0 - xs - rs, 1.0 - ys - rs])


def _nonoverlap_constraints(params):
    """Pairwise non-overlap: dist(i,j) >= r_i + r_j."""
    n = len(params) // 3
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            vals.append(dist - rs[i] - rs[j])
    return np.array(vals)


def _radius_lower(params):
    return params[2::3]


def _radius_upper(params):
    return params[2::3]


def _run_slsqp_from_initial(x0, maxiter=3000):
    """Run SLSQP optimization from a given initial point."""
    n = len(x0) // 3
    
    # Define constraints
    # Boundary: x - r >= 0, y - r >= 0, 1 - x - r >= 0, 1 - y - r >= 0
    # Non-overlap: dist_ij - r_i - r_j >= 0
    
    # Use NonlinearConstraint for boundary
    boundary_lb = np.zeros(4 * n)
    boundary_ub = np.inf * np.ones(4 * n)
    boundary_con = NonlinearConstraint(_boundary_constraints, boundary_lb, boundary_ub, 
                                         jac=lambda p: _boundary_jac(p))
    
    # Non-overlap
    n_pairs = n * (n - 1) // 2
    nonoverlap_lb = np.zeros(n_pairs)
    nonoverlap_ub = np.inf * np.ones(n_pairs)
    nonoverlap_con = NonlinearConstraint(_nonoverlap_constraints, nonoverlap_lb, nonoverlap_ub,
                                          jac=lambda p: _nonoverlap_jac(p))
    
    # Variable bounds: x, y in [0.01, 0.99], r in [0.01, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.01, 0.99))  # x
        bounds.append((0.01, 0.99))  # y
        bounds.append((0.01, 0.5))   # r
    
    res = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                   constraints=[boundary_con, nonoverlap_con],
                   options={'maxiter': maxiter, 'ftol': 1e-12})
    return res


def _boundary_jac(params):
    n = len(params) // 3
    # 4n constraints: [x1-r1, y1-r1, 1-x1-r1, 1-y1-r1, x2-r2, ...]
    jac = np.zeros((4 * n, 3 * n))
    for i in range(n):
        base = 3 * i
        # x_i - r_i
        jac[4*i, base] = 1.0
        jac[4*i, base+2] = -1.0
        # y_i - r_i
        jac[4*i+1, base+1] = 1.0
        jac[4*i+1, base+2] = -1.0
        # 1 - x_i - r_i
        jac[4*i+2, base] = -1.0
        jac[4*i+2, base+2] = -1.0
        # 1 - y_i - r_i
        jac[4*i+3, base+1] = -1.0
        jac[4*i+3, base+2] = -1.0
    return jac


def _nonoverlap_jac(params):
    n = len(params) // 3
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    n_pairs = n * (n - 1) // 2
    jac = np.zeros((n_pairs, 3 * n))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-12:
                dist = 1e-12
                ux, uy = 1.0, 0.0
            else:
                ux, uy = dx/dist, dy/dist
            jac[idx, 3*i] = ux
            jac[idx, 3*i+1] = uy
            jac[idx, 3*i+2] = -1.0
            jac[idx, 3*j] = -ux
            jac[idx, 3*j+1] = -uy
            jac[idx, 3*j+2] = -1.0
            idx += 1
    return jac


def _grid_initial():
    """Create a grid of circles as initial guess."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.005
                cells.append([x, y, r])
    params = np.array([v for cell in cells for v in cell])
    return params


def _random_initial(n_restarts=1):
    """Generate random initial configurations."""
    results = []
    for _ in range(n_restarts):
        params = np.random.rand(N, 3)
        params[:, 0] = 0.1 + 0.8 * params[:, 0]  # x in [0.1, 0.9]
        params[:, 1] = 0.1 + 0.8 * params[:, 1]  # y in [0.1, 0.9]
        params[:, 2] = 0.05 + 0.2 * params[:, 2]  # r in [0.05, 0.25]
        results.append(params.flatten())
    return results


def _polish(circles, maxiter=1000):
    """Final high-precision polish pass."""
    params = np.array([v for c in circles for v in c])
    
    n = len(params) // 3
    boundary_lb = np.zeros(4 * n)
    boundary_ub = np.inf * np.ones(4 * n)
    boundary_con = NonlinearConstraint(_boundary_constraints, boundary_lb, boundary_ub,
                                         jac=lambda p: _boundary_jac(p))
    
    n_pairs = n * (n - 1) // 2
    nonoverlap_lb = np.zeros(n_pairs)
    nonoverlap_ub = np.inf * np.ones(n_pairs)
    nonoverlap_con = NonlinearConstraint(_nonoverlap_constraints, nonoverlap_lb, nonoverlap_ub,
                                          jac=lambda p: _nonoverlap_jac(p))
    
    bounds = []
    for _ in range(n):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.5))
    
    res = minimize(_objective, params, method='SLSQP', bounds=bounds,
                   constraints=[boundary_con, nonoverlap_con],
                   options={'maxiter': maxiter, 'ftol': 1e-14})
    
    if res.success and _is_feasible(res.x.reshape(-1, 3)):
        return res.x.reshape(-1, 3)
    return circles


def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_params = None
    best_score = -np.inf
    
    # Start with grid initial
    grid_params = _grid_initial()
    
    # Try SLSQP from grid
    if _elapsed() < DEADLINE - 30:
        try:
            res = _run_slsqp_from_initial(grid_params, maxiter=2000)
            if res.success and _is_feasible(res.x.reshape(-1, 3)):
                score = np.sum(res.x[2::3])
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Try random restarts
    n_restarts = 8
    for k in range(n_restarts):
        if _elapsed() > DEADLINE - 20:
            break
        try:
            inits = _random_initial(1)
            for init in inits:
                if _elapsed() > DEADLINE - 20:
                    break
                res = _run_slsqp_from_initial(init, maxiter=1500)
                if res.success and _is_feasible(res.x.reshape(-1, 3)):
                    score = np.sum(res.x[2::3])
                    if score > best_score:
                        best_score = score
                        best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish the best result
    if best_params is not None and _elapsed() < DEADLINE - 10:
        try:
            circles = best_params.reshape(-1, 3)
            polished = _polish(circles, maxiter=500)
            if _is_feasible(polished):
                score = np.sum(polished[:, 2])
                if score > best_score:
                    best_score = score
                    best_params = polished.flatten()
        except Exception:
            pass
    
    if best_params is not None:
        result = best_params.reshape(-1, 3)
        # Final feasibility check
        if _is_feasible(result):
            return result
    
    # Fallback: grid
    return _grid_initial().reshape(-1, 3)

# EVOLVE-BLOCK-END
