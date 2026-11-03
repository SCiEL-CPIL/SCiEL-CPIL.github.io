"""Packing 50 equal circles in unit disk - Enhanced multi-stage optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50


def _pairwise_distances(points):
    """Compute all pairwise distances between points."""
    diff = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    return dists


def _score(points):
    """Compute the packing radius for a set of points."""
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    d = np.linalg.norm(points, axis=1)
    margin = 1.0 - d
    if np.min(margin) < -1e-12:
        return 0.0
    dists = _pairwise_distances(points)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    r = min(min_pair, np.min(margin))
    return max(r, 0.0)


def _softmin_logsumexp(x, beta):
    """Numerically stable soft-min using log-sum-exp."""
    # soft-min(x) = -log(sum(exp(-beta*x))) / beta
    # = -log(sum(exp(-beta*(x - min_x)))) / beta - min_x
    x_min = np.min(x)
    exp_term = np.exp(-beta * (x - x_min))
    log_sum = np.log(np.sum(exp_term))
    return -log_sum / beta - x_min


def _smooth_objective(points, beta):
    """Smooth objective: negative soft-min of constraints."""
    d = np.linalg.norm(points, axis=1)
    margin = 1.0 - d
    dists = _pairwise_distances(points)
    np.fill_diagonal(dists, np.inf)
    pair_dist = dists / 2.0
    
    # Collect all constraint values (we want to maximize the minimum)
    # margin: how far each point is from boundary
    # pair_dist: half the distance between each pair
    all_constraints = np.concatenate([margin, pair_dist.ravel()])
    
    # Penalize infeasibility
    if np.min(margin) < -1e-9:
        return 1e6 - _score(points)
    
    soft_min = _softmin_logsumexp(all_constraints, beta)
    return -soft_min


def _ring_to_points(ring_params, ring_counts):
    """Convert ring parameters to point positions.
    
    ring_params: list of (radius, [angle_offsets]) for each ring
    ring_counts: list of counts per ring
    """
    pts = []
    for i, (count, (r, offsets)) in enumerate(zip(ring_counts, ring_params)):
        if count == 0:
            continue
        if count == 1:
            pts.append(np.array([[r, 0.0]]))
        else:
            base = np.linspace(0, 2 * np.pi, count, endpoint=False)
            angles = base + offsets
            xs = r * np.cos(angles)
            ys = r * np.sin(angles)
            pts.append(np.column_stack([xs, ys]))
    return np.vstack(pts)


def _hexagonal_init(n_points, radius_max=1.0):
    """Generate initial hexagonal packing."""
    # Hexagonal lattice spacing
    spacing = 2.0 / (np.sqrt(n_points) + 1.0)
    points = []
    for row in range(-20, 21):
        for col in range(-20, 21):
            x = col * spacing + (row % 2) * spacing * 0.5
            y = row * spacing * np.sqrt(3) / 2
            if np.hypot(x, y) <= radius_max - spacing / 2:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n_points:
        # Fallback to random
        rng = np.random.RandomState(42)
        points = rng.uniform(-1, 1, size=(n_points, 2))
        # Ensure inside unit circle
        d = np.linalg.norm(points, axis=1)
        mask = d <= 1.0
        while np.sum(mask) < n_points:
            extra = rng.uniform(-1, 1, size=(n_points - np.sum(mask), 2))
            points = np.vstack([points[mask], extra])
            d = np.linalg.norm(points, axis=1)
            mask = d <= 1.0
        points = points[:n_points]
    else:
        # Sort by distance from center and take closest n_points
        d = np.linalg.norm(points, axis=1)
        points = points[np.argsort(d)][:n_points]
    return points


def _anneal_points(points, t_start=1.0, t_end=0.01, n_steps=5000, beta_schedule=None):
    """Simulated annealing to improve packing."""
    pts = points.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for step in range(n_steps):
        t = t_start * (t_end / t_start) ** (step / n_steps)
        
        # Pick a random point to move
        i = np.random.randint(N)
        old_pos = pts[i].copy()
        
        # Random perturbation
        delta = np.random.normal(0, t * 0.1, size=2)
        new_pos = old_pos + delta
        
        # Check if still inside unit circle
        if np.linalg.norm(new_pos) > 1.0:
            continue
        
        pts[i] = new_pos
        new_score = _score(pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = pts.copy()
        elif np.random.random() < np.exp((new_score - _score(pts)) / max(t, 1e-10)):
            pts[i] = new_pos
        else:
            pts[i] = old_pos
    
    return best_pts


def _constraint_push(points, max_iters=1000, tol=1e-12):
    """Iteratively push apart violating constraints."""
    pts = points.copy()
    for _ in range(max_iters):
        d = np.linalg.norm(pts, axis=1)
        margin = 1.0 - d
        dists = _pairwise_distances(pts)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        min_margin = np.min(margin)
        r = min(min_pair / 2.0, min_margin)
        
        if min_pair / 2.0 < min_margin - 1e-12:
            i, j = np.unravel_index(np.argmin(dists), dists.shape)
            if i == j:
                continue
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-14:
                continue
            di /= dist
            push = (2 * r - dist) / 2.0
            if push > 1e-14:
                pts[i] += push * di
                pts[j] -= push * di
        else:
            i = int(np.argmin(margin))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-14:
                continue
            di = pi / dist
            push = (1.0 - r - dist)
            if push > 1e-14:
                pts[i] += push * di
        
        if abs(r - _score(pts)) < tol:
            break
    return pts


def _binary_search_polish(points, r_low=0.0, r_high=1.0, n_iters=30):
    """Binary search on radius with constraint satisfaction."""
    best_pts = points.copy()
    best_score = _score(points)
    
    for _ in range(n_iters):
        r_mid = (r_low + r_high) / 2
        # Try to find a valid packing with radius r_mid
        # Use constraint push to try to achieve this radius
        pts = _constraint_push(points, max_iters=200)
        score = _score(pts)
        
        if score >= r_mid - 1e-10:
            r_low = r_mid
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        else:
            r_high = r_mid
    
    return best_pts


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # 5 second safety margin
    
    # Start with hexagonal initialization
    points = _hexagonal_init(N)
    
    # Stage 1: Annealing to find a good topology
    if time.time() < deadline - 30:
        points = _anneal_points(points, t_start=0.5, t_end=0.01, n_steps=3000)
    
    # Stage 2: Constraint push for local improvement
    if time.time() < deadline - 15:
        points = _constraint_push(points, max_iters=500)
    
    # Stage 3: L-BFGS polish on smooth objective
    if time.time() < deadline - 10:
        def smooth_obj_flat(x):
            pts = x.reshape(-1, 2)
            return _smooth_objective(pts, beta=50.0)
        
        x0 = points.ravel()
        try:
            result = minimize(
                smooth_obj_flat,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.fun < smooth_obj_flat(x0):
                pts = result.x.reshape(-1, 2)
                if np.all(np.isfinite(pts)) and _score(pts) > _score(points):
                    points = pts
        except Exception:
            pass
    
    # Stage 4: Final constraint push
    if time.time() < deadline - 5:
        points = _constraint_push(points, max_iters=300)
    
    # Ensure validity
    if not np.all(np.isfinite(points)) or len(points) != N:
        # Fallback
        points = _hexagonal_init(N)
    
    return points
