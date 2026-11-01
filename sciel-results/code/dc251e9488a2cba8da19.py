"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull

N = 14
D = 3
DEADLINE = 130.0


def _true_score(pts):
    """Compute (max/min pairwise distance)^2."""
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2


def _opt_obj(x, tau=1.0):
    """
    Smooth surrogate objective: soft log-sum-exp of log(max/min).
    We minimize log(max) - log(min) which is equivalent to minimizing max/min.
    Using soft versions:
      soft_log_max = tau * log-sum-exp(log(d_i))
      soft_log_min = -tau * log-sum-exp(-log(d_i))
    Objective = soft_log_max - soft_log_min + regularization
    """
    pts = x.reshape(N, D)
    # Center
    pts = pts - pts.mean(axis=0)
    
    # Pairwise distances
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    d_off = d[~np.eye(N, dtype=bool)]
    
    if np.any(d_off < 1e-10):
        return 1e10
    
    log_d = np.log(d_off)
    
    # Soft log-max and soft log-min using log-sum-exp
    # log-sum-exp trick for numerical stability
    max_log_d = np.max(log_d)
    soft_log_max = max_log_d + tau * np.log(np.sum(np.exp(log_d - max_log_d)))
    
    min_log_d = np.min(log_d)
    soft_log_min = min_log_d + tau * np.log(np.sum(np.exp(log_d - min_log_d)))
    # soft_log_min = log-sum-exp(log_d) = min_log_d + tau*log(sum(exp(log_d - min_log_d)))
    # But we want -log-sum-exp(-log_d) = -(-min_log_d + tau*log(sum(exp(-log_d + min_log_d))))
    # Let's just use: soft_log_min = -tau * log-sum-exp(-log_d)
    neg_log_d = -log_d
    max_neg = np.max(neg_log_d)
    soft_neg_lse = max_neg + tau * np.log(np.sum(np.exp(neg_log_d - max_neg)))
    soft_log_min = -soft_neg_lse
    
    obj = soft_log_max - soft_log_min
    
    # Scale regularization: encourage sum of squared distances ~ N (unit-ish scale)
    s2 = np.sum(pts ** 2)
    scale_pen = 0.01 * (s2 - N) ** 2
    
    return obj + scale_pen


def _opt_obj_grad(x, tau=1.0):
    """Gradient of the objective (approximate via finite differences if needed, but let's try analytical)."""
    # For simplicity and robustness, use numerical gradient
    eps = 1e-6
    grad = np.zeros_like(x)
    f0 = _opt_obj(x, tau)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        f_plus = _opt_obj(x_plus, tau)
        grad[i] = (f_plus - f0) / eps
    return grad


def _fib_sphere(n, r=1.0):
    """Generate n points on a sphere using Fibonacci spiral."""
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.column_stack([x, y, z])


def _icosa14():
    """
    Construct 14 points: 12 icosahedron vertices + 2 more.
    Icosahedron has 12 vertices. Add 2 points near the center or on faces.
    For a better start, use icosahedron and add 2 points at opposite face centers.
    """
    phi = (1 + 5 ** 0.5) / 2  # golden ratio
    # 12 icosahedron vertices (normalized)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Add 2 points: center of two opposite faces
    # Face centers of icosahedron are at distance ~0.7947 from center (for unit sphere)
    # Let's add points at (0,0,1) and (0,0,-1) scaled appropriately, or just use two more vertices
    # Actually, let's add points that are slightly inside to avoid coinciding
    p1 = np.array([0.0, 0.0, 0.8])
    p2 = np.array([0.0, 0.0, -0.8])
    
    pts = np.vstack([verts, p1, p2])
    return pts


def _random_init():
    """Random initialization on a sphere."""
    pts = np.random.randn(N, D)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts


def _optimize_from_start(x0, tau_schedule, time_budget):
    """
    Optimize from a starting point with a tau annealing schedule.
    Returns the best point found.
    """
    t_start = time.time()
    best_pts = x0.copy()
    best_score = _true_score(best_pts)
    
    # Anneal tau from large to small
    for tau in tau_schedule:
        if time.time() - t_start > time_budget:
            break
        
        # Define objective for this tau
        def obj(x):
            return _opt_obj(x, tau)
        
        # Use L-BFGS-B
        res = minimize(
            obj,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        x_opt = res.x
        pts = x_opt.reshape(N, D)
        pts = pts - pts.mean(axis=0)
        
        score = _true_score(pts)
        if score < best_score:
            best_score = score
            best_pts = pts.copy()
        
        x0 = x_opt.copy()
    
    return best_pts, best_score


def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    t_start = time.time()
    
    # Initial points from different strategies
    inits = []
    
    # 1. Fibonacci sphere
    inits.append(_fib_sphere(N))
    
    # 2. Icosahedron + 2 points
    inits.append(_icosa14())
    
    # 3. Random starts
    np.random.seed(42)
    for _ in range(5):
        inits.append(_random_init())
    
    # 4. Grid-based (from original, but improved)
    side = int(np.ceil(N ** (1 / D)))
    import itertools
    grid = list(itertools.product(range(side), repeat=D))[:N]
    grid_pts = np.array(grid, dtype=float)
    # Center and normalize
    grid_pts = grid_pts - grid_pts.mean(axis=0)
    norm = np.linalg.norm(grid_pts, axis=1, keepdims=True)
    norm[norm < 1e-9] = 1
    grid_pts = grid_pts / norm
    inits.append(grid_pts)
    
    # Tau schedule: from smooth (large tau) to sharp (small tau)
    tau_schedule = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.1]
    
    best_pts = None
    best_score = np.inf
    
    for i, x0 in enumerate(inits):
        remaining = DEADLINE - (time.time() - t_start) - 10.0  # 10s safety margin
        if remaining < 5:
            break
        
        # Allocate time proportional to remaining
        per_init_time = remaining / max(1, len(inits) - i)
        per_init_time = min(per_init_time, 15.0)  # Cap per-init time
        
        pts, score = _optimize_from_start(x0, tau_schedule, per_init_time)
        
        if score < best_score:
            best_score = score
            best_pts = pts
    
    # Final polish: use the best point found and do a final optimization with small tau
    if best_pts is not None and time.time() - t_start < DEADLINE - 15:
        def final_obj(x):
            return _opt_obj(x, tau=0.1)
        
        res = minimize(
            final_obj,
            best_pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        final_pts = res.x.reshape(N, D)
        final_pts = final_pts - final_pts.mean(axis=0)
        final_score = _true_score(final_pts)
        
        if final_score < best_score:
            best_score = final_score
            best_pts = final_pts
    
    if best_pts is None:
        # Fallback
        best_pts = _fib_sphere(N)
    
    return best_pts


# Module-level timer
_t0 = None
