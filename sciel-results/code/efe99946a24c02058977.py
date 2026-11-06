"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

_START_TIME = time.time()
DEADLINE = _START_TIME + 145.0  # 5s safety margin before 150s kill

def compute_radius(c):
    """Compute the radius for a set of centres c of shape (N, 2)."""
    if c is None:
        return 0.0
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Boundary constraint: distance from centre to nearest edge
    edge_dists = np.minimum(
        np.minimum(0.5 - c[:, 0], c[:, 0] + 0.5),
        np.minimum(0.5 - c[:, 1], c[:, 1] + 0.5)
    )
    r_edge = float(np.min(edge_dists))
    if r_edge <= 0:
        return 0.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = float(np.min(dists) / 2.0)
    return min(r_edge, r_pair)

def _grid_initial():
    """Generate a square grid initial layout."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _hex_lattice_params_to_centers(params):
    """Convert 6 hex lattice parameters to N centres.
    
    params = [a, b, theta, x0, y0, n_rows]
    where a, b are lattice vector magnitudes, theta is angle between them,
    x0, y0 is the offset, n_rows is number of rows (we'll figure out arrangement)
    
    Actually, let's use a simpler parameterization:
    We'll use a parallelogram lattice defined by two vectors v1, v2 and generate
    points in a rectangular region, then take the best N.
    
    params = [a, b, theta, x0, y0]
    v1 = (a, 0)
    v2 = (b*cos(theta), b*sin(theta))
    Generate all lattice points within [-0.5, 0.5]^2 + offset, sort by distance to
    origin or just take all, then pick the N closest to center or all if <= N.
    """
    a, b, theta, x0, y0 = params
    v1 = np.array([a, 0.0])
    v2 = np.array([b * np.cos(theta), b * np.sin(theta)])
    
    # Generate lattice points in a range that covers the square
    # The square is [-0.5, 0.5]^2
    # We need to find all (i, j) such that i*v1 + j*v2 + (x0, y0) is in [-0.5, 0.5]^2
    # Let's estimate the range of i and j
    max_range = 20  # Should be more than enough for 50 circles
    
    points = []
    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            p = i * v1 + j * v2 + np.array([x0, y0])
            if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5:
                points.append(p)
    
    if len(points) == 0:
        return None
    
    points = np.array(points)
    
    # If we have more than N points, keep the N closest to the center (0,0)
    if len(points) > N:
        dists_to_center = np.sum(points ** 2, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        points = points[idx]
    elif len(points) < N:
        # Not enough points - this is a penalty case
        return None
    
    return points

def _objective(params):
    """Objective function for differential evolution.
    We want to maximize the radius, so we minimize negative radius.
    """
    if time.time() > DEADLINE:
        return 0.0  # Return 0 meaning worst case
    
    centers = _hex_lattice_params_to_centers(params)
    if centers is None or len(centers) < N:
        return 10.0  # Large penalty
    
    r = compute_radius(centers)
    return -r

def construct():
    """Construct 50 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    start = time.time()
    
    # Start with grid as baseline
    best_centers = _grid_initial()
    best_radius = compute_radius(best_centers)
    
    # Try differential evolution on hex lattice parameters
    # params = [a, b, theta, x0, y0]
    # a, b: lattice vector magnitudes, roughly in [0.05, 0.5]
    # theta: angle in [0, pi]
    # x0, y0: offset in [-0.5, 0.5]
    
    bounds = [
        (0.05, 0.5),   # a
        (0.05, 0.5),   # b
        (0.0, np.pi),  # theta
        (-0.5, 0.5),   # x0
        (-0.5, 0.5),   # y0
    ]
    
    try:
        # Run differential evolution
        result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=20,
            tol=1e-6,
            polish=False,
            updating='deferred',
            init='sobol',
            workers=1,
        )
        
        if result.success or result.fun < 0:  # If we found something valid
            best_params = result.x
            centers = _hex_lattice_params_to_centers(best_params)
            if centers is not None and len(centers) == N:
                r = compute_radius(centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = centers
    except Exception:
        pass
    
    # Check if we have time for a second round with different seed
    if time.time() < DEADLINE - 20:
        try:
            result2 = differential_evolution(
                _objective,
                bounds,
                seed=123,
                maxiter=150,
                popsize=20,
                tol=1e-6,
                polish=False,
                updating='deferred',
                init='latinhypercube',
                workers=1,
            )
            
            if result2.success or result2.fun < 0:
                best_params2 = result2.x
                centers2 = _hex_lattice_params_to_centers(best_params2)
                if centers2 is not None and len(centers2) == N:
                    r2 = compute_radius(centers2)
                    if r2 > best_radius:
                        best_radius = r2
                        best_centers = centers2
        except Exception:
            pass
    
    # Final validity check
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _grid_initial()
    
    # Ensure all centres are strictly inside or on the boundary
    # (compute_radius already checks this, but let's be safe)
    if np.any(best_centers < -0.5) or np.any(best_centers > 0.5):
        # Clip to boundary (though this shouldn't happen with valid parameters)
        best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

# EVOLVE-BLOCK-END
