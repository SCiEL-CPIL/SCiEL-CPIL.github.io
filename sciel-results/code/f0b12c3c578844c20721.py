# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance
import scipy.optimize


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 570  # 30s safety margin before 600s kill

    def compute_score(points):
        """Compute R = (d_min/d_max)^2"""
        dists = scipy.spatial.distance.pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-15:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x):
        """Minimize negative log of (d_min/d_max) to maximize the ratio"""
        points = x.reshape(n, 2)
        dists = scipy.spatial.distance.pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-15 or d_min < 1e-15:
            return 1e10  # large penalty
        return -(d_min / d_max)

    def normalize(points):
        """Normalize points so that d_max = 1 and centroid at origin"""
        # Center
        points = points - np.mean(points, axis=0)
        dists = scipy.spatial.distance.pdist(points)
        d_max = np.max(dists)
        if d_max < 1e-15:
            return points
        points = points / d_max
        return points

    # Generate several initial configurations
    def init_grid():
        """4x4 grid"""
        xs = np.array([0, 1, 2, 3], dtype=float)
        grid = np.array(np.meshgrid(xs, xs)).T.reshape(-1, 2)
        return grid - 1.5

    def init_concentric_squares():
        """Two concentric squares of 4 points each, rotated 45 deg"""
        r1 = 1.0
        r2 = 0.5
        angles1 = np.array([0, 90, 180, 270]) * np.pi / 180
        angles2 = (np.array([0, 90, 180, 270]) + 45) * np.pi / 180
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        return np.vstack([pts1, pts2])

    def init_hexagonal():
        """Hexagonal-like arrangement for 16 points"""
        # Try a pattern: center + rings
        # 1 center, 6 in first ring, 9 in second ring? That's 16
        pts = [np.array([0.0, 0.0])]
        r1 = 1.0
        for k in range(6):
            angle = k * np.pi / 3
            pts.append(np.array([r1 * np.cos(angle), r1 * np.sin(angle)]))
        r2 = 2.0
        # 9 points in second ring - use angles that are evenly spaced
        for k in range(9):
            angle = k * 2 * np.pi / 9
            pts.append(np.array([r2 * np.cos(angle), r2 * np.sin(angle)]))
        return np.array(pts)

    def init_circles():
        """Points on two concentric circles"""
        r1 = 1.0
        r2 = 0.5
        angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        return np.vstack([pts1, pts2])

    def init_random(seed):
        """Random initialization"""
        np.random.seed(seed)
        return np.random.randn(n, 2) * 1.0

    # List of initial configurations
    inits = [
        init_grid(),
        init_concentric_squares(),
        init_hexagonal(),
        init_circles(),
        init_random(0),
        init_random(1),
        init_random(2),
        init_random(3),
        init_random(4),
        init_random(5),
    ]

    best_points = None
    best_score = -np.inf

    # Evaluate initial configurations
    for init_pts in inits:
        if time.time() > deadline:
            break
        pts = normalize(init_pts.copy())
        score = compute_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Optimize from each initial configuration
    for i, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
        x0 = normalize(init_pts.copy()).flatten()
        
        # Try L-BFGS-B
        try:
            result = scipy.optimize.minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < 0:
                pts = normalize(result.x.reshape(n, 2))
                score = compute_score(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
        except Exception:
            pass

        # Try SLSQP with fewer iterations
        if time.time() > deadline:
            break
        try:
            result = scipy.optimize.minimize(
                objective,
                x0,
                method='SLSQP',
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            if result.success or result.fun < 0:
                pts = normalize(result.x.reshape(n, 2))
                score = compute_score(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
        except Exception:
            pass

    # If we still have time, do more random restarts
    if best_points is None:
        # Fallback
        best_points = normalize(inits[0].copy())
        best_score = compute_score(best_points)

    # Additional random restarts if time permits
    rng = np.random.RandomState(42)
    for restart in range(20):
        if time.time() > deadline:
            break
        x0 = (rng.randn(n, 2) * 1.0).flatten()
        x0 = normalize(x0.reshape(n, 2)).flatten()
        
        try:
            result = scipy.optimize.minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < 0:
                pts = normalize(result.x.reshape(n, 2))
                score = compute_score(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
        except Exception:
            pass

    # Final check
    if best_points is None:
        best_points = np.random.randn(n, 2)

    # Ensure shape is correct
    best_points = np.array(best_points, dtype=float).reshape(16, 2)
    
    return best_points


# EVOLVE-BLOCK-END
