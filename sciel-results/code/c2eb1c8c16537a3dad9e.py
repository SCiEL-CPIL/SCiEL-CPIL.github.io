"""Circle packing in perimeter-4 rectangle: soft-min NM + SA exploration + L-BFGS-B polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 21
DEADLINE = time.time() + 138  # safety margin before 150s kill


def _softmin_radii(cx, cy, w, h, k):
    """
    For given centers and box (w,h) with w+h=2:
      r_i = min( min_j dist(i,j)/2 for j!=i, cx_i, cy_i, w-cx_i, h-cy_i )
    Use soft-min approximation for smoothness.
    """
    n = len(cx)
    bx = np.minimum(cx, w - cx)
    by = np.minimum(cy, h - cy)
    bdist = np.minimum(bx, by)

    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    pair_min = np.min(dist, axis=1) / 2.0

    r = -k * np.logaddexp(-bdist / k, -pair_min / k)
    return np.maximum(r, 0.0)


def _exact_radii(cx, cy, w, h):
    """Compute exact hard-min radii."""
    n = len(cx)
    bx = np.minimum(cx, w - cx)
    by = np.minimum(cy, h - cy)
    bdist = np.minimum(bx, by)
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    pair_min = np.min(dist, axis=1) / 2.0
    r = np.minimum(bdist, pair_min)
    return np.maximum(r, 1e-6)


def _objective_soft(params, k):
    """
    params: [cx_0, cy_0, ..., cx_{n-1}, cy_{n-1}, w]
    h = 2 - w
    Returns negative sum of soft-min radii (to minimize).
    """
    n = N
    cx = params[:n]
    cy = params[n:2*n]
    w = params[2*n]
    h = 2.0 - w

    if w <= 0.05 or h <= 0.05:
        return 1e6

    # Penalty for centers outside box
    ox = np.maximum(0.0, -cx, cx - w)
    oy = np.maximum(0.0, -cy, cy - h)
    pen = np.sum(ox**2 + oy**2) * 50.0

    r = _softmin_radii(cx, cy, w, h, k)
    return -(np.sum(r)) + pen


def _objective_exact(params):
    """
    Exact objective: negative sum of hard-min radii + overlap penalty.
    Used for final evaluation.
    """
    n = N
    cx = params[:n]
    cy = params[n:2*n]
    w = params[2*n]
    h = 2.0 - w

    if w <= 0.05 or h <= 0.05:
        return 1e6

    ox = np.maximum(0.0, -cx, cx - w)
    oy = np.maximum(0.0, -cy, cy - h)
    pen = np.sum(ox**2 + oy**2) * 100.0

    r = _exact_radii(cx, cy, w, h)
    return -(np.sum(r)) + pen


def _nelder_mead_optimize(x0, k, max_iter=2000):
    """Run Nelder-Mead with time guard."""
    def _obj(p):
        return _objective_soft(p, k)
    
    try:
        res = minimize(
            _obj, x0,
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True,
            }
        )
        return res.x, res.fun
    except Exception:
        return x0, _obj(x0)


def _lbfgsb_optimize(x0, k, max_iter=500):
    """Run L-BFGS-B with bounds on centers and width."""
    n = N
    w_init = x0[2*n]
    h_init = 2.0 - w_init
    
    # Bounds: centers in [0.001, w-0.001] etc., w in [0.1, 1.9]
    bounds = []
    for i in range(N):
        bounds.append((0.001, 1.999))  # cx_i
    for i in range(N):
        bounds.append((0.001, 1.999))  # cy_i
    bounds.append((0.1, 1.9))  # w
    
    def _obj(p):
        return _objective_soft(p, k)
    
    try:
        res = minimize(
            _obj, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': 1e-10,
                'gtol': 1e-8,
            }
        )
        return res.x, res.fun
    except Exception:
        return x0, _obj(x0)


def _simulated_annealing(x0, k, n_iter=500, T_start=0.05, T_end=0.001, rng=None):
    """
    Simulated annealing to explore the parameter space.
    Returns best solution found.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = N
    current = x0.copy()
    current_val = _objective_soft(current, k)
    best = current.copy()
    best_val = current_val
    
    for it in range(n_iter):
        if time.time() > DEADLINE - 20:
            break
        T = T_start * (T_end / T_start) ** (it / n_iter)
        
        # Propose perturbation
        scale = T * 2.0
        proposal = current.copy()
        proposal[:n] += rng.normal(0, scale, n)
        proposal[n:2*n] += rng.normal(0, scale, n)
        proposal[2*n] += rng.normal(0, scale * 0.5)
        
        # Clamp
        w_p = proposal[2*n]
        h_p = 2.0 - w_p
        if w_p < 0.1 or h_p < 0.1:
            continue
        proposal[:n] = np.clip(proposal[:n], 0.001, w_p - 0.001)
        proposal[n:2*n] = np.clip(proposal[n:2*n], 0.001, h_p - 0.001)
        
        prop_val = _objective_soft(proposal, k)
        delta = prop_val - current_val
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current = proposal
            current_val = prop_val
            if current_val < best_val:
                best_val = current_val
                best = current.copy()
    
    return best, best_val


def _make_grid_init(w, h, rng):
    """Create a grid-like initialization."""
    side = int(np.ceil(np.sqrt(N)))
    params = np.zeros(2 * N + 1)
    params[2*N] = w
    for i in range(N):
        ix = i % side
        iy = i // side
        params[i] = (ix + 0.5) / side * w
        params[N + i] = (iy + 0.5) / side * h
    # Add small random perturbation
    params[:N] += rng.uniform(-0.02, 0.02, N)
    params[N:2*N] += rng.uniform(-0.02, 0.02, N)
    # Clamp
    params[:N] = np.clip(params[:N], 0.01, w - 0.01)
    params[N:2*N] = np.clip(params[N:2*N], 0.01, h - 0.01)
    return params


def _polish_final(cx, cy, w, h, k=0.003, max_iter=300):
    """Final high-precision polish using L-BFGS-B with very small k."""
    x0 = np.concatenate([cx, cy, [w]])
    
    n = N
    bounds = []
    for i in range(N):
        bounds.append((0.0005, w - 0.0005))
    for i in range(N):
        bounds.append((0.0005, h - 0.0005))
    bounds.append((max(0.05, w - 0.3), min(1.95, w + 0.3)))
    
    def _obj(p):
        return _objective_soft(p, k)
    
    try:
        res = minimize(
            _obj, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': 1e-12,
                'gtol': 1e-10,
            }
        )
        cx_l = res.x[:N]
        cy_l = res.x[N:2*N]
        w_l = res.x[2*N]
        h_l = 2.0 - w_l
        return cx_l, cy_l, w_l, h_l
    except Exception:
        return cx, cy, w, h


def construct():
    start = time.time()
    rng = np.random.default_rng(42)
    best_params = None
    best_val = np.inf

    k_initial = 0.02
    k_mid = 0.01
    k_final = 0.003

    # Generate diverse initializations
    inits = []
    
    # Grid in square
    inits.append(_make_grid_init(1.0, 1.0, rng))
    
    # Grid in various aspect ratios
    for w_ratio in [0.5, 0.7, 1.3, 1.5, 2.0]:
        w_try = min(w_ratio, 1.95)
        h_try = 2.0 - w_try
        if h_try > 0.05:
            inits.append(_make_grid_init(w_try, h_try, rng))
    
    # Random inits
    for _ in range(6):
        w_try = rng.uniform(0.4, 1.6)
        h_try = 2.0 - w_try
        params = np.zeros(2 * N + 1)
        params[2*N] = w_try
        params[:N] = rng.uniform(0.02, w_try - 0.02, N)
        params[N:2*N] = rng.uniform(0.02, h_try - 0.02, N)
        inits.append(params)
    
    # Phase 1: Nelder-Mead from each init with moderate k
    for idx, x0 in enumerate(inits):
        if time.time() > DEADLINE - 40:
            break
        x_opt, val = _nelder_mead_optimize(x0, k_initial, max_iter=1500)
        if val < best_val:
            best_val = val
            best_params = x_opt.copy()
    
    if best_params is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    cx = best_params[:N]
    cy = best_params[N:2*N]
    w = best_params[2*N]
    h = 2.0 - w

    # Phase 2: Simulated annealing to escape local minima
    if time.time() < DEADLINE - 30:
        sa_params, sa_val = _simulated_annealing(
            best_params, k_mid, n_iter=400, T_start=0.04, T_end=0.002, rng=rng
        )
        if sa_val < best_val:
            best_val = sa_val
            best_params = sa_params
            cx = best_params[:N]
            cy = best_params[N:2*N]
            w = best_params[2*N]
            h = 2.0 - w

    # Phase 3: L-BFGS-B refinement
    if time.time() < DEADLINE - 20:
        x_opt, val = _lbfgsb_optimize(best_params, k_mid, max_iter=300)
        if val < best_val:
            best_val = val
            best_params = x_opt
            cx = best_params[:N]
            cy = best_params[N:2*N]
            w = best_params[2*N]
            h = 2.0 - w

    # Phase 4: Final high-precision polish
    if time.time() < DEADLINE - 10:
        cx, cy, w, h = _polish_final(cx, cy, w, h, k=k_final, max_iter=200)

    # Compute final radii (hard min, no soft)
    r = _exact_radii(cx, cy, w, h)

    # Slightly shrink radii to ensure disjoint interiors (strict inequality)
    r = r * (1.0 - 1e-6)

    result = np.column_stack([cx, cy, r])
    return result
