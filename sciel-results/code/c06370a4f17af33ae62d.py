"""Combined packing: hexagonal seed + L-BFGS-B potential optimization + simulated annealing."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 142.0  # safety margin under 150s wall-clock


def _radius(pos):
    """Score of a configuration (double precision)."""
    pos = np.asarray(pos, dtype=np.float64)
    # Wall margins: distance from each centre to nearest wall
    wall_margins = 0.5 - np.abs(pos)  # (N, 2)
    min_wall = wall_margins.min()
    # Pairwise distances
    d = cdist(pos, pos)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    r = min(min_wall, 0.5 * min_pair)
    return float(r)


def _potential(x):
    """Potential to minimize: sum of inverse 4th powers of wall and pair distances."""
    centers = x.reshape(-1, 2)
    
    # Wall distances
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(dx, dy)
    
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, 1e10)
    min_pair_dists = np.min(dists, axis=1)
    
    eps = 1e-6
    wall_safe = np.clip(wall_dists, eps, None)
    pair_safe = np.clip(min_pair_dists, eps, None)
    
    V = np.sum(1.0 / (wall_safe ** 4)) + np.sum(1.0 / (pair_safe ** 4))
    return V


def _hexagonal_seed(rng):
    """Generate a hexagonal lattice seed, clipped to the square."""
    # Estimate spacing for 50 circles in unit square
    # Area per circle ~ 0.02, hex packing density ~ 0.9069
    # r^2 * pi * 0.9069 ~ 0.02 => r ~ 0.084, d = 2r ~ 0.168
    # Let's try d = 0.145
    d = 0.145
    pts = []
    range_x = int(np.ceil(1.0 / d)) + 2
    range_y = int(np.ceil(1.0 / (d * np.sqrt(3) / 2))) + 2
    
    for j in range(-range_y, range_y + 1):
        for i in range(-range_x, range_x + 1):
            x = i * d + (j % 2) * d / 2
            y = j * d * np.sqrt(3) / 2
            # Center the grid
            x -= 0.5
            y -= 0.5
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append((x, y))
    
    pts = np.array(pts, dtype=np.float64)
    
    if len(pts) > N:
        # Select N closest to center for compact packing
        dists_to_center = np.sum(pts ** 2, axis=1)
        idx = np.argsort(dists_to_center)
        pts = pts[idx[:N]]
    elif len(pts) < N:
        # Pad with random points if needed
        extra = rng.uniform(-0.45, 0.45, size=(N - len(pts), 2))
        pts = np.vstack([pts, extra])
    
    # Add small jitter to break symmetry
    pts += rng.normal(0, 0.003, pts.shape)
    pts = np.clip(pts, -0.49, 0.49)
    
    return pts


def _lbfgs_optimize(pos, deadline):
    """Run L-BFGS-B on the potential."""
    x0 = pos.flatten()
    try:
        res = minimize(_potential, x0, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14})
        x_opt = res.x
        centers = x_opt.reshape(-1, 2)
        centers = np.clip(centers, -0.5, 0.5)
        return centers
    except Exception:
        return pos


def _anneal(pos, iters, t0, tmin, deadline):
    """Simulated annealing on the radius score."""
    pos = np.array(pos, dtype=np.float64)
    r = _radius(pos)
    best = (r, pos.copy())
    temp = t0
    rng = np.random.default_rng()
    
    for it in range(iters):
        if time.time() > deadline - 2.0:
            break
        temp = max(tmin, temp * 0.999)
        delta = 0.003 * rng.standard_normal((N, 2))
        cand = pos + delta
        cand = np.clip(cand, -0.5, 0.5)
        rc = _radius(cand)
        if rc >= r or (temp > 0 and rng.random() < np.exp((rc - r) / max(temp, 1e-9))):
            pos, r = cand, rc
            if r > best[0]:
                best = (r, pos.copy())
    return best


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best = (0.0, None)
    rng = np.random.default_rng(42)
    
    # First: hexagonal seed + L-BFGS
    pos = _hexagonal_seed(rng)
    r = _radius(pos)
    best = (r, pos.copy())
    
    # L-BFGS optimization
    pos_opt = _lbfgs_optimize(pos, deadline)
    r_opt = _radius(pos_opt)
    if r_opt > best[0]:
        best = (r_opt, pos_opt.copy())
    
    # Annealing from the L-BFGS result
    pos_a = _anneal(pos_opt, iters=500, t0=0.008, tmin=0.0003, deadline=deadline)
    if pos_a[0] > best[0]:
        best = pos_a
    
    # More restarts with different seeds and perturbations
    for attempt in range(30):
        if time.time() > deadline - 5.0:
            break
        
        # Mix of fresh hex seeds and perturbed best
        if attempt % 5 == 0:
            pos = _hexagonal_seed(rng)
        else:
            pos = best[1] + 0.008 * rng.standard_normal((N, 2))
            pos = np.clip(pos, -0.5, 0.5)
        
        # L-BFGS
        pos_opt = _lbfgs_optimize(pos, deadline)
        r_opt = _radius(pos_opt)
        
        # Anneal
        pos_a = _anneal(pos_opt, iters=300, t0=0.006, tmin=0.0002, deadline=deadline)
        
        cand = max((r_opt, pos_opt), pos_a, key=lambda t: t[0])
        if cand[0] > best[0]:
            best = cand
        
        # Early stop if near best known
        if best[0] >= 0.07137:
            break
    
    return np.asarray(best[1], dtype=np.float64)
