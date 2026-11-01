"""Pack 200 equal circles in a unit square using force-directed + L-BFGS-B + active-set polish."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import time

N = 200

def _compute_radius(pts):
    """Compute the largest common radius for given centers."""
    d_bound = np.min(np.stack([
        0.5 - np.abs(pts[:, 0]),
        0.5 - np.abs(pts[:, 1])
    ]), axis=1)
    min_bound = np.min(d_bound)
    tree = cKDTree(pts)
    pairs, dists = tree.query_pairs(2.0, output_type='ndarray')
    if len(pairs) == 0:
        min_pair = np.inf
    else:
        min_pair = np.min(dists) / 2.0
    return min(min_bound, min_pair)

def _soft_min_neg(vals, k=100.0):
    """Smooth approximation of -min(vals) using log-sum-exp."""
    if len(vals) == 0:
        return 0.0
    v = np.asarray(vals)
    v_min = np.min(v)
    if v_min >= 0:
        return 0.0
    # -softmin = logsumexp(-k*v) / k - v_min approximation
    # We want to minimize the minimum distance, so we maximize softmin
    # softmin(x) ≈ -(1/k)*log(sum(exp(-k*x_i)))
    # For numerical stability:
    shifted = -k * (v - v_min)
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    softmin_val = v_min - log_sum / k
    return -softmin_val  # return negative of softmin, so minimizing this maximizes softmin

def _objective(x, pts):
    """Objective to minimize: negative of the radius (we want to maximize radius)."""
    pts2 = x.reshape(N, 2)
    # Boundary distances
    d_bound = np.stack([
        0.5 - np.abs(pts2[:, 0]),
        0.5 - np.abs(pts2[:, 1])
    ], axis=1)
    min_bound = np.min(d_bound)
    
    # Pairwise distances using KDTree
    tree = cKDTree(pts2)
    pairs, dists = tree.query_pairs(2.0, output_type='ndarray')
    if len(pairs) == 0:
        min_pair = np.inf
    else:
        min_pair = np.min(dists) / 2.0
    
    r = min(min_bound, min_pair)
    return -r

def _hex_init(aspect_ratio=1.0, rng=None):
    """Generate hexagonal lattice initialization."""
    if rng is None:
        rng = np.random.RandomState(0)
    
    # Estimate spacing
    s = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    # Generate hexagonal lattice
    max_i = int(np.ceil(1.0 / s)) + 2
    max_j = int(np.ceil(1.0 / (s * np.sqrt(3) / 2))) + 2
    
    candidates = []
    for j in range(-max_j, max_j + 1):
        for i in range(-max_i, max_i + 1):
            x = (i * s + j * s * 0.5) * aspect_ratio
            y = j * s * np.sqrt(3) / 2
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) < N:
        # Fallback: grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        pts = grid[:N].copy()
    else:
        # Select N points, biased toward center
        dists_from_center = np.sum(candidates**2, axis=1)
        weights = 1.0 / (1.0 + dists_from_center * 10)
        weights /= weights.sum()
        idx = rng.choice(len(candidates), size=N, replace=False, p=weights)
        pts = candidates[idx].copy()
    
    # Ensure strictly inside
    margin = 0.005
    pts[:, 0] = np.clip(pts[:, 0], -0.5 + margin, 0.5 - margin)
    pts[:, 1] = np.clip(pts[:, 1], -0.5 + margin, 0.5 - margin)
    
    return pts

def _random_init(rng=None):
    """Generate random initialization."""
    if rng is None:
        rng = np.random.RandomState(0)
    pts = rng.uniform(-0.45, 0.45, size=(N, 2))
    return pts

def _force_directed(pts, max_iters=2000, deadline=None):
    """Force-directed relaxation using repulsive forces."""
    pts = pts.copy()
    margin = 0.001
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Repulsive forces between pairs
        tree = cKDTree(pts)
        pairs, dists = tree.query_pairs(0.2, output_type='ndarray')
        
        if len(pairs) > 0:
            for p in pairs:
                i, j = p
                di = pts[i] - pts[j]
                d = np.linalg.norm(di)
                if d > 1e-10:
                    # Repulsive force ~ 1/d^2
                    f = di / (d * d * d) * 0.01
                    forces[i] += f
                    forces[j] -= f
        
        # Attractive force toward center (weak)
        forces -= pts * 0.001
        
        # Boundary repulsion
        for axis in range(2):
            for side in [-1, 1]:
                # If point is close to boundary, push it inward
                if side == -1:
                    close = pts[:, axis] < -0.5 + 0.1
                    if np.any(close):
                        forces[close, axis] += (0.5 - 0.1 - pts[close, axis]) * 0.5
                else:
                    close = pts[:, axis] > 0.5 - 0.1
                    if np.any(close):
                        forces[close, axis] -= (pts[close, axis] - (0.5 - 0.1)) * 0.5
        
        # Update positions
        step = 0.01 / (1.0 + it * 0.01)
        pts += forces * step
        
        # Clamp to square
        pts[:, 0] = np.clip(pts[:, 0], -0.5 + margin, 0.5 - margin)
        pts[:, 1] = np.clip(pts[:, 1], -0.5 + margin, 0.5 - margin)
    
    return pts

def _get_active_pairs(pts, threshold=1.05):
    """Get pairs of points that are close together (active constraints)."""
    tree = cKDTree(pts)
    pairs = tree.query_pairs(threshold, output_type='ndarray')
    return pairs

def _active_set_polish(pts, max_iters=500, deadline=None):
    """Polish the solution by focusing on active (close) pairs."""
    pts = pts.copy()
    margin = 0.0001
    current_r = _compute_radius(pts)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Get active pairs
        pairs = _get_active_pairs(pts, threshold=current_r * 1.1)
        
        if len(pairs) == 0:
            # No active pairs, try boundary
            break
        
        # Compute forces from active pairs
        forces = np.zeros_like(pts)
        for p in pairs:
            i, j = p
            di = pts[i] - pts[j]
            d = np.linalg.norm(di)
            if d > 1e-10:
                # Push apart
                f = di / d * 0.001
                forces[i] += f
                forces[j] -= f
        
        # Boundary forces
        for axis in range(2):
            for side in [-1, 1]:
                if side == -1:
                    close = pts[:, axis] < -0.5 + current_r
                    if np.any(close):
                        forces[close, axis] += (0.5 - current_r - pts[close, axis]) * 0.1
                else:
                    close = pts[:, axis] > 0.5 - current_r
                    if np.any(close):
                        forces[close, axis] -= (pts[close, axis] - (0.5 - current_r)) * 0.1
        
        # Small step
        step = 0.005 / (1.0 + it * 0.05)
        pts += forces * step
        
        # Clamp
        pts[:, 0] = np.clip(pts[:, 0], -0.5 + margin, 0.5 - margin)
        pts[:, 1] = np.clip(pts[:, 1], -0.5 + margin, 0.5 - margin)
        
        # Update radius
        new_r = _compute_radius(pts)
        if new_r < current_r - 1e-6:
            # Revert if got worse
            pts -= forces * step
            pts[:, 0] = np.clip(pts[:, 0], -0.5 + margin, 0.5 - margin)
            pts[:, 1] = np.clip(pts[:, 1], -0.5 + margin, 0.5 - margin)
        else:
            current_r = new_r
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    rng = np.random.RandomState(42)
    best_pts = None
    best_score = 0.0
    
    # Try multiple initializations
    init_configs = [
        ('hex_ar1', lambda: _hex_init(aspect_ratio=1.0, rng=rng)),
        ('hex_ar12', lambda: _hex_init(aspect_ratio=1.2, rng=rng)),
        ('hex_ar08', lambda: _hex_init(aspect_ratio=0.8, rng=rng)),
        ('random', lambda: _random_init(rng=rng)),
    ]
    
    for name, init_fn in init_configs:
        if time.time() > deadline - 30:
            break
        
        try:
            pts = init_fn()
        except Exception:
            continue
        
        # Force-directed relaxation
        pts = _force_directed(pts, max_iters=500, deadline=deadline - 20)
        
        # L-BFGS-B optimization
        try:
            x0 = pts.flatten()
            bounds = [(-0.5 + 0.0001, 0.5 - 0.0001)] * (2 * N)
            
            def obj(x):
                return _objective(x, pts)
            
            result = minimize(
                obj, x0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < obj(x0):
                pts = result.x.reshape(N, 2)
                pts[:, 0] = np.clip(pts[:, 0], -0.5 + 0.0001, 0.5 - 0.0001)
                pts[:, 1] = np.clip(pts[:, 1], -0.5 + 0.0001, 0.5 - 0.0001)
        except Exception:
            pass
        
        # Active set polish
        pts = _active_set_polish(pts, max_iters=200, deadline=deadline - 10)
        
        score = _compute_radius(pts)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
        
        if time.time() > deadline - 15:
            break
    
    if best_pts is None:
        # Fallback
        best_pts = _hex_init(aspect_ratio=1.0, rng=rng)
    
    return best_pts
