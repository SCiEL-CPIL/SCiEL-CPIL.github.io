"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.
Strategy: Direct constrained NLP via L-BFGS-B on (positions, radii) with
non-overlap and boundary constraints, seeded from hexagonal and grid lattices.
Polish with SLSQP on the Delaunay contact graph.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-9
DEADLINE = 145.0  # Safety margin for 150s limit


def _valid_check(arr):
    """Check that all circles are inside [0,1]^2 and non-overlapping."""
    pos = arr[:, :2]
    rad = arr[:, 2]
    if np.any(rad < 0) or np.any(pos < rad - 1e-7) or np.any(pos > 1 - rad + 1e-7):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            if d < rad[i] + rad[j] - 1e-7:
                return False
    return True


def _delaunay_edges(pos):
    """Return list of (i, j) edges from Delaunay triangulation."""
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        # Fallback: all pairs
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _constraints_slsqp(pos, rad):
    """Build constraint list for SLSQP: non-overlap for Delaunay edges + boundaries."""
    edges = _delaunay_edges(pos)
    cons = []
    # Non-overlap constraints: dist_ij >= r_i + r_j
    for i, j in edges:
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        d = np.hypot(dx, dy)
        if d < 1e-12:
            d = 1e-12
        cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j:
                     np.hypot(x[2*i] - x[2*j], x[2*i+1] - x[2*j+1]) - x[2*N+i] - x[2*N+j],
                     'jac': lambda x, i=i, j=j: _overlap_jac(x, i, j)})
    # Boundary constraints: x_i >= r_i, 1-x_i >= r_i, y_i >= r_i, 1-y_i >= r_i
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N+i]})
    return cons


def _overlap_jac(x, i, j):
    """Jacobian of constraint dist_ij - r_i - r_j w.r.t. x = [pos..., rad...]."""
    dx = x[2*i] - x[2*j]
    dy = x[2*i+1] - x[2*j+1]
    d = np.hypot(dx, dy)
    if d < 1e-12:
        d = 1e-12
    jac = np.zeros(2 * N + N)
    jac[2*i] = dx / d
    jac[2*i+1] = dy / d
    jac[2*j] = -dx / d
    jac[2*j+1] = -dy / d
    jac[2*N + i] = -1.0
    jac[2*N + j] = -1.0
    return jac


def _slsqp_polish(arr, max_iters=30):
    """Polish with SLSQP on the contact graph."""
    if not _valid_check(arr):
        return arr
    pos = arr[:, :2].copy()
    rad = arr[:, 2].copy()
    x0 = np.concatenate([pos.flatten(), rad])
    
    def objective(x):
        return -x[2*N:].sum()
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        cons = _constraints_slsqp(pos, rad)
        try:
            res = minimize(objective, x0, method='SLSQP',
                           bounds=[(0.01, 0.99)] * 2 * N + [(0.001, 0.5)] * N,
                           constraints=cons,
                           options={'maxiter': 50, 'ftol': 1e-10})
            if res.success and res.fun < objective(x0):
                x0 = res.x
                pos = x0[:2*N].reshape(N, 2)
                rad = x0[2*N:]
            else:
                break
        except Exception:
            break
    return np.column_stack([pos, rad])


def _lbfgsb_optimize(pos0, rad0, max_iters=200):
    """Direct L-BFGS-B optimization of -sum(radii) with penalty for violations."""
    pos = pos0.copy()
    rad = rad0.copy()
    
    def penalty_objective(x):
        """Objective with penalty for constraint violations."""
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        obj = -r.sum()
        # Penalty for boundary violations
        for i in range(N):
            for val in [p[i, 0] - r[i], 1 - p[i, 0] - r[i],
                        p[i, 1] - r[i], 1 - p[i, 1] - r[i]]:
                if val < 0:
                    obj += 1000 * val**2
        # Penalty for overlap violations
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(p[i, 0] - p[j, 0], p[i, 1] - p[j, 1])
                violation = r[i] + r[j] - d
                if violation > 0:
                    obj += 1000 * violation**2
        return obj
    
    x0 = np.concatenate([pos.flatten(), rad])
    bounds = [(0.01, 0.99)] * 2 * N + [(0.001, 0.5)] * N
    
    best_x = x0.copy()
    best_obj = penalty_objective(x0)
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 15:
            break
        try:
            res = minimize(penalty_objective, x0, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 30, 'ftol': 1e-12})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
                x0 = res.x
            else:
                # Try a smaller step
                x0 = best_x + 0.1 * (res.x - best_x)
        except Exception:
            break
    
    pos_final = best_x[:2*N].reshape(N, 2)
    rad_final = best_x[2*N:]
    return pos_final, rad_final


def _hexagonal_init(n_circles):
    """Generate hexagonal lattice initialization."""
    pos_list = []
    # Hexagonal packing with varying row offsets
    row = 0
    spacing = 0.18
    while len(pos_list) < n_circles:
        for col in range(10):
            if len(pos_list) >= n_circles:
                break
            x = 0.1 + col * spacing + (0.09 if row % 2 == 1 else 0)
            y = 0.1 + row * spacing * 0.866
            if x < 0.95 and y < 0.95:
                pos_list.append((x, y))
        row += 1
    pos = np.array(pos_list[:n_circles])
    rad = np.full(n_circles, 0.08)
    # Grow radii to max feasible
    for _ in range(30):
        for i in range(n_circles):
            dmin = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
            for j in range(n_circles):
                if i != j:
                    d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                    dmin = min(dmin, d - rad[j])
            rad[i] = max(0.01, dmin)
    return pos, rad


def _grid_init(side, n_circles):
    """Generate grid initialization."""
    pos_list = []
    for i in range(side):
        for j in range(side):
            if len(pos_list) >= n_circles:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            pos_list.append((x, y))
    pos = np.array(pos_list[:n_circles])
    rad = np.full(n_circles, 0.5 / side - 0.01)
    for _ in range(30):
        for i in range(n_circles):
            dmin = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
            for j in range(n_circles):
                if i != j:
                    d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                    dmin = min(dmin, d - rad[j])
            rad[i] = max(0.01, dmin)
    return pos, rad


def construct():
    t_start = time.time()
    rng = np.random.default_rng(123)
    best = None
    best_score = -1.0

    def consider(arr):
        nonlocal best, best_score
        if arr is None or len(arr) < N:
            return
        arr = np.array(arr[:N])
        if not _valid_check(arr):
            return
        s = float(arr[:, 2].sum())
        if s > best_score:
            best_score = s
            best = arr

    # Generate diverse initializations
    all_starts = []

    # 1. Hexagonal
    pos_h, rad_h = _hexagonal_init(N)
    all_starts.append((pos_h, rad_h))

    # 2. Grid 5x5 (25 circles, add one)
    pos_g, rad_g = _grid_init(5, N)
    all_starts.append((pos_g, rad_g))

    # 3. Grid 6x6 (first 26)
    pos_g6, rad_g6 = _grid_init(6, N)
    all_starts.append((pos_g6, rad_g6))

    # 4. Perturbed hexagonal
    for k in range(3):
        if time.time() > DEADLINE - 30:
            break
        p0 = pos_h + 0.03 * (rng.random((N, 2)) * 2 - 1)
        p0 = np.clip(p0, 0.05, 0.95)
        r0 = rad_h.copy()
        all_starts.append((p0, r0))

    # 5. Perturbed grid
    for k in range(3):
        if time.time() > DEADLINE - 30:
            break
        p0 = pos_g + 0.03 * (rng.random((N, 2)) * 2 - 1)
        p0 = np.clip(p0, 0.05, 0.95)
        r0 = rad_g.copy()
        all_starts.append((p0, r0))

    # Run optimization on each start
    for pos0, rad0 in all_starts:
        if time.time() > DEADLINE - 25:
            break
        try:
            pos_f, rad_f = _lbfgsb_optimize(pos0, rad0, max_iters=100)
            arr = np.column_stack([pos_f, rad_f])
            # Polish with SLSQP
            arr_polished = _slsqp_polish(arr, max_iters=20)
            consider(arr_polished)
        except Exception:
            pass

    # Final polish on best
    if best is not None:
        arr_polished = _slsqp_polish(best, max_iters=30)
        consider(arr_polished)

    if best is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
