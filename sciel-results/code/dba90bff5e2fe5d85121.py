"""Circle packing in a unit square: maximize common radius for 50 equal circles."""
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 50
DEADLINE = 118.0  # safety margin before 120s kill (evaluator kills at 150s)

def _score(pos):
    """Compute the maximum valid radius for given positions."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(pos)):
        return -1.0
    # Wall constraint
    wall = 0.5 - np.abs(pos).max(axis=1)
    r_wall = wall.min()
    if r_wall < 0:
        return -1.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    r_pair = (dist.min() / 2.0)
    return min(r_wall, r_pair)

def _relax(c, R, iters, eta0, eta1, t0=None, deadline=None):
    """
    Relaxation step: push overlapping circles apart and keep them inside the square.
    """
    c = c.copy()
    n = len(c)
    for it in range(iters):
        if t0 is not None and deadline is not None:
            if time.time() - t0 > deadline - 3:
                break
        t = it / max(iters, 1)
        eta = eta0 + (eta1 - eta0) * t
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * R - d, 0.0) / d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, np.newaxis] * diff).sum(1)
        # Walls
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c

def _grow(start, R0, R1, step, iters, eta0, eta1, t0, deadline):
    """
    Grow the radius from R0 to R1 in steps, relaxing at each step.
    """
    c = start.copy()
    best = (c.copy(), _score(c))
    R = R0
    while R <= R1 and (time.time() - t0) < deadline - 5:
        c = _relax(c, R, iters, eta0, eta1, t0, deadline)
        r = _score(c)
        if r > best[1]:
            best = (c.copy(), r)
        R += step
        if (time.time() - t0) > deadline - 5:
            break
    return best

def _init_grid_7x7():
    """7x7 grid with one extra circle at center."""
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    return np.vstack([g, extra])

def _init_jitter8(rng):
    """8x8 grid, pick 50, add jitter."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.03, 0.03, c.shape)
    return c

def _init_rand(rng):
    """Random uniform positions."""
    return rng.uniform(-0.45, 0.45, (N, 2))

def _init_hex():
    """Hexagonal lattice starting position."""
    positions = []
    spacing = 0.14
    row_spacing = spacing * np.sqrt(3) / 2
    for row in range(8):
        for col in range(7):
            x = (col - 3) * spacing + (0.5 * spacing if row % 2 else 0)
            y = (row - 3.5) * row_spacing
            positions.append([x, y])
    positions = np.array(positions[:N])
    max_abs = np.abs(positions).max()
    if max_abs > 0.45:
        positions = positions * (0.45 / max_abs)
    return positions

def _init_grid_8x8():
    """8x8 grid, pick 50."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    return g[:N].copy()

def _get_delaunay_edges(c):
    """Get Delaunay edges as a list of (i, j) pairs."""
    try:
        tri = Delaunay(c)
        edges = set()
        for simplex in tri.simplices:
            edges.add(tuple(sorted((simplex[0], simplex[1]))))
            edges.add(tuple(sorted((simplex[1], simplex[2]))))
            edges.add(tuple(sorted((simplex[0], simplex[2]))))
        return list(edges)
    except Exception:
        # Fallback: all pairs (expensive but safe)
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                edges.append((i, j))
        return edges

def _delaunay_objective(x, edges, wall_margin):
    """
    Objective: minimize negative minimum radius.
    x: flattened (2N,) array of centers
    edges: list of (i, j) Delaunay edges
    wall_margin: margin from walls (0.5)
    """
    c = x.reshape(N, 2)
    
    # Wall constraint: min distance to walls
    r_wall = (0.5 - np.abs(c).max(axis=1)).min()
    
    # Pairwise constraint on Delaunay edges
    r_pair = np.inf
    for i, j in edges:
        d = np.linalg.norm(c[i] - c[j])
        r_pair = min(r_pair, d / 2.0)
    
    r_min = min(r_wall, r_pair)
    return -r_min

def _delaunay_gradient(x, edges, wall_margin):
    """
    Gradient of the objective (negative min radius).
    This is approximate - we only compute gradients for the active constraint.
    """
    c = x.reshape(N, 2)
    grad = np.zeros_like(x)
    
    # Find active constraint
    r_wall_vals = 0.5 - np.abs(c).max(axis=1)
    r_wall = r_wall_vals.min()
    
    r_pair = np.inf
    active_edge = None
    for i, j in edges:
        d = np.linalg.norm(c[i] - c[j])
        half_d = d / 2.0
        if half_d < r_pair:
            r_pair = half_d
            active_edge = (i, j)
    
    r_min = min(r_wall, r_pair)
    
    if r_wall <= r_pair:
        # Wall constraint is active
        # Gradient of (0.5 - |c_i|) w.r.t. c_i
        for i in range(N):
            if r_wall_vals[i] <= r_wall + 1e-10:
                # Which coordinate is limiting?
                ax = c[i, 0]
                ay = c[i, 1]
                if abs(ax) >= abs(ay):
                    if ax > 0:
                        grad[2*i] = -1.0
                    else:
                        grad[2*i] = 1.0
                else:
                    if ay > 0:
                        grad[2*i+1] = -1.0
                    else:
                        grad[2*i+1] = 1.0
    else:
        # Pair constraint is active
        if active_edge is not None:
            i, j = active_edge
            diff = c[i] - c[j]
            d = np.linalg.norm(diff)
            if d > 1e-12:
                # Gradient of (d/2) w.r.t. c_i is diff/d, w.r.t. c_j is -diff/d
                grad[2*i:2*i+2] = -diff / d
                grad[2*j:2*j+2] = diff / d
    
    return grad

def _delaunay_lbfgsb(c_init, t0, deadline, max_iter=200, ftol=1e-10, gtol=1e-10):
    """
    Optimize using L-BFGS-B with Delaunay-sparse objective.
    """
    x0 = c_init.flatten()
    
    def obj(x):
        if time.time() - t0 > deadline - 5:
            return _delaunay_objective(x, _get_delaunay_edges(x.reshape(N,2)), 0.5)
        return _delaunay_objective(x, _get_delaunay_edges(x.reshape(N,2)), 0.5)
    
    def grad(x):
        if time.time() - t0 > deadline - 5:
            return _delaunay_gradient(x, _get_delaunay_edges(x.reshape(N,2)), 0.5)
        return _delaunay_gradient(x, _get_delaunay_edges(x.reshape(N,2)), 0.5)
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        result = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                         bounds=bounds,
                         options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol,
                                  'maxfun': 5000, 'maxls': 50})
        c_opt = result.x.reshape(N, 2)
        c_opt = np.clip(c_opt, -0.5, 0.5)
        return c_opt
    except Exception:
        return c_init

def _soft_min_radius(c, beta=50.0):
    """
    Soft-min of all constraints using log-sum-exp.
    Returns a smooth approximation of the min radius.
    """
    c = np.asarray(c)
    # Wall distances
    r_wall = 0.5 - np.abs(c).max(axis=1)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    r_pair = dist / 2.0
    
    # Soft min of all constraints
    # Use log-sum-exp for stability
    all_r = np.concatenate([r_wall, r_pair[r_pair < np.inf]])
    if len(all_r) == 0:
        return 0.0
    
    # Soft min: -logsumexp(-beta * all_r) / beta
    shifted = -beta * all_r
    shifted -= shifted.max()
    logsumexp = np.log(np.exp(shifted).sum())
    soft_min = -(logsumexp + shifted.max()) / beta
    
    return soft_min

def _soft_min_objective(x, beta=50.0):
    """Objective for L-BFGS-B: minimize negative soft-min radius."""
    c = x.reshape(N, 2)
    c = np.clip(c, -0.5, 0.5)
    return -_soft_min_radius(c, beta)

def _soft_min_gradient(x, beta=50.0):
    """Gradient of soft-min objective."""
    c = x.reshape(N, 2)
    c = np.clip(c, -0.5, 0.5)
    
    # Wall distances
    r_wall = 0.5 - np.abs(c).max(axis=1)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    r_pair = dist / 2.0
    
    # All radii
    all_r = np.concatenate([r_wall, r_pair[r_pair < np.inf]])
    if len(all_r) == 0:
        return np.zeros_like(x)
    
    # Soft min weights
    shifted = -beta * all_r
    shifted -= shifted.max()
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / exp_shifted.sum()
    
    # Gradient
    grad = np.zeros_like(x)
    
    # Wall part
    for i in range(N):
        w = weights[i]
        if w < 1e-15:
            continue
        ax = c[i, 0]
        ay = c[i, 1]
        if abs(ax) >= abs(ay):
            if ax > 0:
                grad[2*i] = -w * beta
            else:
                grad[2*i] = w * beta
        else:
            if ay > 0:
                grad[2*i+1] = -w * beta
            else:
                grad[2*i+1] = w * beta
    
    # Pair part
    offset = N
    for i in range(N):
        for j in range(i+1, N):
            w = weights[offset + (i*(2*N-1) - i*(i+1)//2 + (j-i-1))] if False else 0  # This indexing is complex
            # Simpler: iterate through pairs
            pass
    
    # Simpler approach: compute gradient numerically or use a different method
    # For now, use finite differences
    eps = 1e-7
    grad_fd = np.zeros_like(x)
    for k in range(len(x)):
        x_plus = x.copy()
        x_plus[k] += eps
        grad_fd[k] = (_soft_min_objective(x_plus, beta) - _soft_min_objective(x, beta)) / eps
    
    return grad_fd

def _langevin_refine(c, r, t0, deadline, steps=5000, T0=0.001, T1=0.0001, lr=0.002):
    """
    Langevin dynamics refinement: add stochastic kicks to escape local minima.
    """
    c = c.copy()
    best_c = c.copy()
    best_r = r
    rng = np.random.default_rng(999)
    
    for step in range(steps):
        if time.time() - t0 > deadline - 5:
            break
        t = step / steps
        T = T0 + (T1 - T0) * t
        lr_t = lr * (1.0 - 0.5 * t)
        
        # Compute forces (same as relaxation)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * r - d, 0.0) / d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, np.newaxis] * diff).sum(1)
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - r), 0.0) + np.maximum((-0.5 + r) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - r), 0.0) + np.maximum((-0.5 + r) - c[:, 1], 0.0)
        
        # Langevin update: deterministic + stochastic
        noise = T * rng.standard_normal(c.shape)
        c += lr_t * f + noise
        c = np.clip(c, -0.5, 0.5)
        
        r_new = _score(c)
        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()
    
    return best_c, best_r

def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    
    # Multiple diverse initializations
    inits = [
        _init_grid_7x7(),
        _init_jitter8(rng),
        _init_hex(),
        _init_grid_8x8(),
        _init_rand(np.random.default_rng(42)),
        _init_rand(np.random.default_rng(77)),
    ]
    
    best = None
    best_r = -1.0
    
    # Phase 1: Grow radius from each starting configuration
    n_inits = len(inits)
    time_per_init = (DEADLINE - (time.time() - t0) - 30) / n_inits  # reserve 30s for final refinement
    
    for i, start in enumerate(inits):
        remaining = DEADLINE - (time.time() - t0)
        if remaining < 15:
            break
        
        budget = min(time_per_init, remaining * 0.85 / (n_inits - i))
        budget = max(budget, 3.0)
        
        # Grow from 0.05 to 0.0715
        c, r = _grow(start, 0.05, 0.0715, 0.0004, 1500, 0.02, 0.002, t0, DEADLINE)
        
        if r > best_r:
            best_r = r
            best = c.copy()
        
        # Check time
        if time.time() - t0 > DEADLINE - 35:
            break
    
    if best is None:
        best = _init_grid_7x7()
        best_r = _score(best)
    
    # Phase 2: Delaunay-based L-BFGS-B optimization
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 20:
        c_opt = _delaunay_lbfgsb(best, t0, DEADLINE, max_iter=150, ftol=1e-12, gtol=1e-12)
        r_opt = _score(c_opt)
        if r_opt > best_r:
            best_r = r_opt
            best = c_opt
    
    # Phase 3: Langevin refinement on the best configuration
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 15:
        c, r = _langevin_refine(best, best_r, t0, DEADLINE, steps=6000, T0=0.0012, T1=0.00005, lr=0.0025)
        if r > best_r:
            best_r = r
            best = c
    
    # Phase 4: Another Delaunay L-BFGS-B pass with tighter tolerance
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 12:
        c_opt = _delaunay_lbfgsb(best, t0, DEADLINE, max_iter=200, ftol=1e-14, gtol=1e-14)
        r_opt = _score(c_opt)
        if r_opt > best_r:
            best_r = r_opt
            best = c_opt
    
    # Phase 5: Final fine-tune relaxation at current radius
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 8:
        r_cur = _score(best)
        c = _relax(best, r_cur, 2000, 0.008, 0.0003, t0, DEADLINE)
        r_new = _score(c)
        if r_new >= r_cur:
            best = c
            best_r = r_new
    
    # Phase 6: One more Langevin pass if time allows
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 6:
        c, r = _langevin_refine(best, best_r, t0, DEADLINE, steps=3000, T0=0.0006, T1=0.00002, lr=0.0015)
        if r > best_r:
            best_r = r
            best = c
    
    # Final check
    if not np.all(np.isfinite(best)) or best.shape != (N, 2):
        best = _init_grid_7x7()
    
    return best
