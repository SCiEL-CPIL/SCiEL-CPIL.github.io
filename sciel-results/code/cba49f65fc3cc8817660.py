"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

def _true_score(pts):
    """Compute the maximum valid radius for given centers."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Distance to walls
    wall_dists = np.minimum(
        np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]),
        np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
    )
    min_wall = np.min(wall_dists)
    return min(min_pair, 2.0 * min_wall) / 2.0

def _soft_min(vals, alpha=50.0):
    """Soft minimum approximation."""
    vals = np.asarray(vals, dtype=np.float64)
    return -np.log(np.sum(np.exp(-alpha * vals))) / alpha

def _soft_radius(pts, alpha=50.0):
    """Compute a smooth surrogate for the minimum radius."""
    pts = np.asarray(pts, dtype=np.float64)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1e10)
    # Wall distances
    wall_dists = np.minimum(
        np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]),
        np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
    )
    # All constraints
    all_constraints = np.concatenate([dists.ravel(), 2.0 * wall_dists])
    return _soft_min(all_constraints, alpha) / 2.0

def _negative_soft_radius(pts, alpha=50.0):
    """Negative soft radius for minimization."""
    return -_soft_radius(pts, alpha)

def _init_grid():
    """Create a square grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _init_hexagonal():
    """Create a hexagonal lattice initialization."""
    pts = []
    m = 8
    for row in range(m):
        for col in range(m):
            x = (col + 0.5) / m - 0.5
            y = (row + 0.5) / m - 0.5
            if row % 2 == 1:
                x += 0.5 / m
            pts.append([x, y])
    pts = np.array(pts[:N])
    max_coord = np.max(np.abs(pts))
    if max_coord > 0.45:
        pts = pts * (0.45 / max_coord)
    return pts

def _init_delaunay_based():
    """Create initialization based on Delaunay triangulation of random points."""
    np.random.seed(42)
    pts = np.random.uniform(-0.45, 0.45, (N, 2))
    return pts

def _init_random(seed=0):
    """Create a random initialization."""
    rng = np.random.RandomState(seed)
    pts = rng.uniform(-0.45, 0.45, (N, 2))
    return pts

def _force_relax(pts, n_iter=200, dt=0.002):
    """Force-directed relaxation to spread points apart."""
    pts = pts.copy()
    for _ in range(n_iter):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, 1.0)
        
        force = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i != j:
                    d = dists[i, j]
                    if d < 0.15:
                        f = 0.005 / (d * d)
                        force[i] += f * diff[i, j] / d
        
        # Wall repulsion
        for i in range(N):
            for axis in range(2):
                if pts[i, axis] > 0.4:
                    force[i, axis] -= 0.05 * (pts[i, axis] - 0.4)
                elif pts[i, axis] < -0.4:
                    force[i, axis] -= 0.05 * (pts[i, axis] + 0.4)
        
        pts += dt * force
        pts = np.clip(pts, -0.5, 0.5)
    
    return pts

def _optimize_lbfgsb(pts, alpha=50.0, deadline=None, maxiter=100):
    """Optimize using L-BFGS-B on the soft surrogate."""
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def objective(x):
        return _negative_soft_radius(x.reshape(N, 2), alpha)
    
    result = minimize(objective, pts.ravel(), method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if deadline is not None and time.time() > deadline:
        return pts, _true_score(pts)
    
    if result.success or result.fun < _negative_soft_radius(pts, alpha):
        new_pts = result.x.reshape(N, 2)
        new_pts = np.clip(new_pts, -0.5, 0.5)
        new_score = _true_score(new_pts)
        old_score = _true_score(pts)
        if new_score > old_score:
            return new_pts, new_score
    
    return pts, _true_score(pts)

def _optimize_sa(pts, t_start=0.03, t_end=0.0005, n_iter=15000, deadline=None):
    """Simulated annealing optimization."""
    best_pts = pts.copy()
    best_score = _true_score(pts)
    current_pts = pts.copy()
    current_score = best_score
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        new_pts = current_pts.copy()
        idx = np.random.randint(N)
        step = t_start * (t_end / t_start) ** (i / n_iter)
        new_pts[idx] += np.random.normal(0, step, 2)
        new_pts[idx] = np.clip(new_pts[idx], -0.5, 0.5)
        
        new_score = _true_score(new_pts)
        
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_pts = new_pts.copy()
                best_score = new_score
        else:
            temp = t_start * (t_end / t_start) ** (i / n_iter)
            if temp > 1e-10:
                prob = np.exp((new_score - current_score) / temp)
                if np.random.random() < prob:
                    current_pts = new_pts
                    current_score = new_score
    
    return best_pts, best_score

def _local_refine(pts, n_iter=200, deadline=None):
    """Local refinement using coordinate descent."""
    best_pts = pts.copy()
    best_score = _true_score(pts)
    
    for _ in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            for axis in range(2):
                for delta in [0.001, -0.001, 0.0005, -0.0005, 0.0002, -0.0002]:
                    new_pts = best_pts.copy()
                    new_pts[idx, axis] += delta
                    if -0.5 <= new_pts[idx, axis] <= 0.5:
                        new_score = _true_score(new_pts)
                        if new_score > best_score + 1e-14:
                            best_pts = new_pts
                            best_score = new_score
                            improved = True
        
        if not improved:
            break
    
    return best_pts, best_score

def _radius_expansion(pts, max_iter=1000, deadline=None):
    """Iteratively expand the radius by pushing points apart."""
    pts = pts.copy()
    current_r = _true_score(pts)
    
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Find the most constrained pair or wall
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        wall_dists = np.minimum(
            np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]),
            np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
        )
        
        # Find minimum constraint
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_wall_idx = np.argmin(wall_dists)
        
        min_pair_dist = dists[min_pair_idx]
        min_wall_dist = 2.0 * wall_dists[min_wall_idx]
        
        if min_pair_dist < min_wall_dist:
            # Push apart the closest pair
            i, j = min_pair_idx
            direction = (pts[i] - pts[j]) / min_pair_dist
            shift = 0.001
            pts[i] += direction * shift
            pts[j] -= direction * shift
        else:
            # Push away from wall
            i = min_wall_idx
            for axis in range(2):
                if pts[i, axis] > 0:
                    pts[i, axis] -= 0.001
                else:
                    pts[i, axis] += 0.001
        
        pts = np.clip(pts, -0.5, 0.5)
        
        new_r = _true_score(pts)
        if new_r > current_r:
            current_r = new_r
        else:
            break
    
    return pts, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Multiple diverse initializations
    inits = [
        _init_grid(),
        _init_hexagonal(),
        _init_delaunay_based(),
        _init_random(0),
        _init_random(1),
        _init_random(2),
    ]
    
    best_pts = None
    best_score = 0.0
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Force relaxation
        relaxed = _force_relax(init, n_iter=100)
        
        # L-BFGS-B optimization on soft surrogate
        lbfgsb_pts, lbfgsb_score = _optimize_lbfgsb(relaxed, alpha=50.0, deadline=deadline)
        if lbfgsb_score > best_score:
            best_pts = lbfgsb_pts.copy()
            best_score = lbfgsb_score
        
        # Simulated annealing
        if time.time() < deadline - 20.0:
            sa_pts, sa_score = _optimize_sa(lbfgsb_pts, t_start=0.02, t_end=0.0003, 
                                           n_iter=10000, deadline=deadline)
            if sa_score > best_score:
                best_pts = sa_pts.copy()
                best_score = sa_score
        
        # Local refinement
        if time.time() < deadline - 10.0:
            ref_pts, ref_score = _local_refine(best_pts, n_iter=100, deadline=deadline)
            if ref_score > best_score:
                best_pts = ref_pts.copy()
                best_score = ref_score
        
        # Radius expansion
        if time.time() < deadline - 5.0:
            exp_pts, exp_score = _radius_expansion(best_pts, max_iter=500, deadline=deadline)
            if exp_score > best_score:
                best_pts = exp_pts.copy()
                best_score = exp_score
    
    # Final refinement pass
    if best_pts is not None and time.time() < deadline - 2.0:
        final_pts, final_score = _local_refine(best_pts, n_iter=200, deadline=deadline)
        if final_score > best_score:
            best_pts = final_pts
            best_score = final_score
    
    if best_pts is None:
        return _init_grid()
    
    return best_pts
# EVOLVE-BLOCK-END
