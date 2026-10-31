import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist

N = 16
D = 2

def _pairwise_distances(pts):
    """Compute all pairwise Euclidean distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))

def _score(pts):
    """Compute (max/min)^2 for pairwise distances."""
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    dmax = d.max()
    if dmin <= 1e-12:
        return np.inf
    return (dmax / dmin) ** 2

def _smooth_objective(pts, beta):
    """
    Smooth approximation of the score.
    Uses log-sum-exp for soft min and soft max of squared distances.
    Returns a value close to (max_d / min_d)^2 for large beta.
    """
    # Use squared distances for efficiency and stability
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d_sq = np.sum(diff ** 2, axis=2)
    
    # Mask out diagonal (self-distances)
    iu = np.triu_indices(N, 1)
    d_sq_valid = d_sq[iu]
    
    # Ensure positive values for log
    d_sq_valid = np.maximum(d_sq_valid, 1e-12)
    
    # Soft Max: max(d_sq) approx
    max_d_sq = np.max(d_sq_valid)
    # To avoid overflow, shift
    exp_terms_max = np.exp(beta * (d_sq_valid - max_d_sq))
    soft_max_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(exp_terms_max))
    
    # Soft Min: min(d_sq) approx
    min_d_sq = np.min(d_sq_valid)
    exp_terms_min = np.exp(-beta * (d_sq_valid - min_d_sq))
    soft_min_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(exp_terms_min))
    
    if soft_min_sq < 1e-8:
        return 1e10
        
    # The score is (max/min)^2 = max_sq / min_sq
    return soft_max_sq / soft_min_sq

def _make_hexagonal_cluster(n, spread=6):
    """Generate a hexagonal lattice cluster of n points centered at origin."""
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])

    pts = []
    for i in range(-spread, spread + 1):
        for j in range(-spread, spread + 1):
            p = i * v1 + j * v2
            pts.append(p)
    pts = np.array(pts)

    # Center the cluster
    pts -= pts.mean(axis=0)

    # Sort by distance to center and pick the closest n
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    return pts[order[:n]]

def _make_circular_config(n):
    """1 on center + 7 on inner ring + 8 on outer ring, scaled to fit."""
    pts = [np.array([0.0, 0.0])]
    r1 = 0.5
    r2 = 1.0
    for k in range(7):
        theta = 2 * np.pi * k / 7
        pts.append(r1 * np.array([np.cos(theta), np.sin(theta)]))
    for k in range(8):
        theta = 2 * np.pi * k / 8 + np.pi / 8
        pts.append(r2 * np.array([np.cos(theta), np.sin(theta)]))
    return np.array(pts)

def _make_grid_config(n):
    """4x4 grid, centered and scaled to unit diameter."""
    x, y = np.meshgrid(np.arange(4), np.arange(4))
    pts = np.column_stack([x.ravel(), y.ravel()]).astype(float)
    pts -= pts.mean(axis=0)
    # Scale to diameter 1
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, 0)
    diam = np.max(d)
    if diam > 0:
        pts /= diam
    return pts

def _make_radial_grid_config(n):
    """
    Square-root radial distribution based on a 4x4 grid.
    Points are placed at grid positions but radially scaled by sqrt(index) factor 
    to mitigate corner disparity.
    """
    # Generate 4x4 grid indices
    indices = np.arange(n).reshape(4, 4)
    # Center the grid indices
    grid_x, grid_y = np.meshgrid(np.arange(4), np.arange(4))
    grid_x -= 1.5
    grid_y -= 1.5
    
    # Radial distance from center for each grid point
    r_grid = np.sqrt(grid_x**2 + grid_y**2)
    
    # Scale factor: proportional to sqrt of normalized radial distance? 
    # Or simply use the grid position but adjust radius.
    # Let's try: r_final = base_r * (r_grid / max_r_grid)^0.5 * max_r_grid?
    # Actually, let's just take the grid points and apply a radial distortion.
    # Or simpler: Use the grid, but pull corners in.
    
    # Let's stick to a simpler radial seed: 4 rings of 4 points?
    # No, let's use the inspiration's "square-root radial" idea.
    # Create points on a 4x4 grid, then project them radially to a circle with 
    # radius proportional to sqrt(distance from center)?
    
    # Let's try a specific configuration:
    # Center point? No, 16 points.
    # Let's use 4 concentric squares?
    # Or just perturb the grid radially.
    
    # For now, let's use a standard grid but with radial compression.
    pts = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(float)
    
    # Calculate distance from center
    d_center = np.linalg.norm(pts, axis=1)
    max_d = d_center.max()
    
    # Apply radial scaling: new_r = max_d * sqrt(d_center / max_d)
    # This pulls corners in more than edges?
    # If d_center is large (corner), sqrt(d_center) is smaller than d_center linearly?
    # Wait, if we want to reduce max distance, we should compress the outer points.
    # Let's try: new_r = max_d * (d_center / max_d)^0.8
    scale_factor = (d_center / max_d) ** 0.8
    new_r = max_d * scale_factor
    
    # Keep angle, change radius
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    # Handle center (0,0) if it exists (it doesn't in 4x4 centered at 1.5)
    # 4x4 centered at 1.5 has no point at (0,0). Closest are (0.5, 0.5) etc.
    
    pts_new = np.column_stack([new_r * np.cos(angles), new_r * np.sin(angles)])
    
    # Normalize to unit diameter
    d = _pairwise_distances(pts_new)
    np.fill_diagonal(d, 0)
    diam = np.max(d)
    if diam > 0:
        pts_new /= diam
        
    return pts_new

def _make_random_config(n, seed=0):
    """Random points in unit square, centered and scaled."""
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    pts -= pts.mean(axis=0)
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, 0)
    diam = np.max(d)
    if diam > 0:
        pts /= diam
    return pts

def _make_ring_config(n):
    """Two rings: 8 inner, 8 outer."""
    r_in = 0.6
    r_out = 1.0
    pts_in = []
    pts_out = []
    for k in range(8):
        theta_in = 2 * np.pi * k / 8
        pts_in.append(r_in * np.array([np.cos(theta_in), np.sin(theta_in)]))
        theta_out = 2 * np.pi * k / 8 + np.pi / 8
        pts_out.append(r_out * np.array([np.cos(theta_out), np.sin(theta_out)]))
    
    pts = np.vstack([pts_in, pts_out])
    # Normalize
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, 0)
    diam = np.max(d)
    if diam > 0:
        pts /= diam
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    best_pts = None
    best_score = np.inf

    def update_best(pts, score):
        nonlocal best_pts, best_score
        if score < best_score:
            best_score = score
            best_pts = pts.copy()

    # 1. Generate diverse initial candidates
    candidates = [
        _make_hexagonal_cluster(N),
        _make_circular_config(N),
        _make_grid_config(N),
        _make_radial_grid_config(N),
        _make_ring_config(N),
        _make_random_config(N, seed=1),
        _make_random_config(N, seed=2),
        _make_random_config(N, seed=3),
        _make_random_config(N, seed=4),
        _make_random_config(N, seed=5),
    ]

    # Evaluate all candidates
    for c in candidates:
        s = _score(c)
        update_best(c, s)

    # 2. Global Search using Differential Evolution
    # Use the smooth objective for DE to handle non-differentiability.
    
    if time.time() < deadline - 60:
        try:
            beta_de = 50.0
            
            def de_objective(pts_flat):
                return _smooth_objective(pts_flat.reshape(N, 2), beta_de)

            bounds = [(-1, 1)] * (N * 2)
            result = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=150,
                popsize=25,
                tol=1e-5,
                polish=False,
                init='sobol',
            )
            if result.fun < np.inf:
                new_pts = result.x.reshape(N, 2)
                new_score = _score(new_pts)
                update_best(new_pts, new_score)
        except Exception:
            pass

    # 3. Local Refinement
    # Strategy:
    # a) Powell on exact objective (good for non-smooth)
    # b) L-BFGS-B on smooth objective (good for gradient-based convergence)
    # c) Powell again on exact objective
    
    def local_refine(pts_init, time_limit):
        current_pts = pts_init.copy()
        current_score = _score(current_pts)
        
        # Stage 1: Powell on exact
        if time.time() < time_limit - 10:
            try:
                res1 = minimize(
                    lambda x: _score(x.reshape(N, 2)),
                    current_pts.ravel(),
                    method='Powell',
                    options={'maxiter': 100, 'maxfev': 1000}
                )
                new_pts = res1.x.reshape(N, 2)
                new_score = _score(new_pts)
                if new_score < current_score:
                    current_pts = new_pts
                    current_score = new_score
            except Exception:
                pass

        # Stage 2: L-BFGS-B on smooth (beta=50)
        if time.time() < time_limit - 10:
            try:
                res2 = minimize(
                    lambda x: _smooth_objective(x.reshape(N, 2), 50.0),
                    current_pts.ravel(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                new_pts = res2.x.reshape(N, 2)
                new_score = _score(new_pts)
                if new_score < current_score:
                    current_pts = new_pts
                    current_score = new_score
            except Exception:
                pass

        # Stage 3: Powell on exact again
        if time.time() < time_limit - 5:
            try:
                res3 = minimize(
                    lambda x: _score(x.reshape(N, 2)),
                    current_pts.ravel(),
                    method='Powell',
                    options={'maxiter': 200, 'maxfev': 2000}
                )
                new_pts = res3.x.reshape(N, 2)
                new_score = _score(new_pts)
                if new_score < current_score:
                    current_pts = new_pts
                    current_score = new_score
            except Exception:
                pass

        # Stage 4: L-BFGS-B on sharper smooth (beta=200)
        if time.time() < time_limit - 2:
            try:
                res4 = minimize(
                    lambda x: _smooth_objective(x.reshape(N, 2), 200.0),
                    current_pts.ravel(),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-14}
                )
                new_pts = res4.x.reshape(N, 2)
                new_score = _score(new_pts)
                if new_score < current_score:
                    current_pts = new_pts
                    current_score = new_score
            except Exception:
                pass

        return current_pts, current_score

    # Refine top candidates
    candidates_to_refine = list(candidates)
    if best_pts is not None:
        candidates_to_refine.append(best_pts)
    
    # Sort by raw score to prioritize best ones
    candidates_to_refine.sort(key=lambda x: _score(x))
    
    # Refine the top 6 distinct candidates
    refined_count = 0
    for c in candidates_to_refine:
        if time.time() > deadline - 30:
            break
        if refined_count >= 6:
            break
            
        # Avoid refining exact duplicates
        is_dup = False
        for r in candidates_to_refine[:refined_count]:
            if np.allclose(c, r, atol=1e-5):
                is_dup = True
                break
        if is_dup:
            continue
            
        try:
            refined_pts, refined_score = local_refine(c, deadline - 25)
            update_best(refined_pts, refined_score)
            refined_count += 1
        except Exception:
            pass

    # 4. Final Polish with Nelder-Mead on the raw objective
    if best_pts is not None and time.time() < deadline - 5:
        try:
            def raw_obj(x):
                pts = x.reshape(N, 2)
                return _score(pts)
            
            res = minimize(
                raw_obj,
                best_pts.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_score:
                new_pts = res.x.reshape(N, 2)
                new_score = _score(new_pts)
                update_best(new_pts, new_score)
        except Exception:
            pass

    if best_pts is None:
        return _make_hexagonal_cluster(N)

    return best_pts
