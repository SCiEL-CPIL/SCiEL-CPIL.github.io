# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 570  # seconds, with safety margin

    def score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def neg_score(points):
        return -score(points)

    # Generate multiple structurally informed initializations
    def hexagonal_init():
        """Hexagonal lattice arrangement for 16 points"""
        pts = []
        # Center point
        pts.append([0.0, 0.0])
        # Ring 1: 6 points
        r1 = 1.0
        for i in range(6):
            angle = i * np.pi / 3
            pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
        # Ring 2: 9 points (3 on each side, staggered)
        r2 = 2.0
        for i in range(3):
            angle = i * 2 * np.pi / 3
            pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
            # Add two more points between ring 1 and ring 2
            angle_mid1 = angle + np.pi / 3
            angle_mid2 = angle - np.pi / 3
            r_mid = 1.5
            pts.append([r_mid * np.cos(angle_mid1), r_mid * np.sin(angle_mid1)])
            pts.append([r_mid * np.cos(angle_mid2), r_mid * np.sin(angle_mid2)])
        return np.array(pts[:16])

    def circular_init():
        """Points on concentric circles"""
        pts = []
        # Center
        pts.append([0.0, 0.0])
        # Inner ring: 5 points
        r1 = 1.0
        for i in range(5):
            angle = i * 2 * np.pi / 5
            pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
        # Outer ring: 10 points
        r2 = 2.0
        for i in range(10):
            angle = i * 2 * np.pi / 10 + np.pi / 10  # offset by half step
            pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
        return np.array(pts)

    def ring_based_init():
        """Multi-ring with optimal spacing"""
        pts = []
        # Center
        pts.append([0.0, 0.0])
        # Ring 1: 6 points at radius 1
        r1 = 1.0
        for i in range(6):
            angle = i * 2 * np.pi / 6
            pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
        # Ring 2: 9 points at radius ~1.732 (sqrt(3))
        r2 = np.sqrt(3)
        for i in range(9):
            angle = i * 2 * np.pi / 9 + np.pi / 9
            pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
        return np.array(pts)

    def random_init(seed):
        """Random initialization"""
        rng = np.random.RandomState(seed)
        return rng.randn(16, 2) * 2

    def grid_init():
        """Grid-based initialization"""
        # 4x4 grid with some perturbation
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        X, Y = np.meshgrid(x, y)
        pts = np.column_stack([X.flatten(), Y.flatten()])
        return pts

    # Generate candidate initializations
    candidates = []
    candidates.append(hexagonal_init())
    candidates.append(circular_init())
    candidates.append(ring_based_init())
    candidates.append(grid_init())
    for seed in [42, 123, 456, 789]:
        candidates.append(random_init(seed))

    # Optimize each candidate using L-BFGS-B
    best_points = None
    best_score = -np.inf

    for init_pts in candidates:
        if time.time() - start_time > deadline:
            break

        # Normalize to have similar scale
        init_pts = init_pts - np.mean(init_pts, axis=0)
        init_scale = np.max(pdist(init_pts)) if len(pdist(init_pts)) > 0 else 1.0
        if init_scale > 0:
            init_pts = init_pts / init_scale * 2  # Scale to have max distance ~4

        try:
            from scipy.optimize import minimize

            result = minimize(
                neg_score,
                init_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )

            opt_pts = result.x.reshape(16, 2)
            s = score(opt_pts)

            if s > best_score:
                best_score = s
                best_points = opt_pts

        except Exception:
            continue

    # If no optimization succeeded, fall back to best initial
    if best_points is None:
        for init_pts in candidates:
            s = score(init_pts)
            if s > best_score:
                best_score = s
                best_points = init_pts.copy()

    # Final refinement: try a few more random perturbations from best
    if time.time() - start_time < deadline - 30:
        from scipy.optimize import minimize

        rng = np.random.RandomState(999)
        for _ in range(5):
            if time.time() - start_time > deadline - 10:
                break

            perturbed = best_points + rng.randn(16, 2) * 0.1
            try:
                result = minimize(
                    neg_score,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                opt_pts = result.x.reshape(16, 2)
                s = score(opt_pts)
                if s > best_score:
                    best_score = s
                    best_points = opt_pts
            except Exception:
                continue

    return best_points


# EVOLVE-BLOCK-END
