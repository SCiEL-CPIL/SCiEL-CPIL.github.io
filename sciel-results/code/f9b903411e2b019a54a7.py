# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize (d_min/d_max)^2.
    
    Strategy:
    1. Reframe the objective to maximize the soft-min of pairwise distances.
       This is a smooth surrogate for the hard min, enabling gradient-based optimization.
    2. Use L-BFGS-B for fast convergence on this smooth landscape.
    3. Initialize with diverse geometric structures (hexagonal, circular, random).
    4. Perform a final polish on the raw objective to ensure the hard min is maximized.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 585  # Safety margin for 600s limit

    def time_exceeded():
        return time.time() > deadline

    def evaluate(points):
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    best_points = None
    best_score = -1.0

    def update_best(points):
        nonlocal best_score, best_points
        score = evaluate(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # --- Objective Functions ---

    def soft_min_objective(params):
        """
        Minimize -soft_min(distances).
        soft_min(d) = -1/beta * log(sum(exp(-beta * d_i)))
        As beta -> infinity, soft_min -> min.
        We use a large beta to approximate the min closely while keeping it smooth.
        """
        points = params.reshape((n, 2))
        dists = pdist(points, 'euclidean')
        
        # Scale distances to avoid overflow in exp
        # If distances are very small, soft_min is very negative.
        # We want to maximize the minimum distance.
        
        beta = 50.0  # Sharpness of the soft-min
        
        # To ensure numerical stability, subtract the max (or min) before exp
        # Here we compute: -1/beta * log(sum(exp(-beta * d_i)))
        # Let m = min(d_i). exp(-beta * d_i) = exp(-beta * (d_i - m)) * exp(-beta * m)
        # log(sum(exp(-beta * d_i))) = log(exp(-beta * m) * sum(exp(-beta * (d_i - m))))
        #                             = -beta * m + log(sum(exp(-beta * (d_i - m))))
        # soft_min = -1/beta * (-beta * m + log(...)) = m - 1/beta * log(...)
        
        m = np.min(dists)
        # If m is 0, points are coincident.
        if m < 1e-12:
            # Penalize coincident points heavily
            return 1e6
        
        shifted = dists - m
        # exp(-beta * shifted) is <= 1, so no overflow risk. Underflow is fine (becomes 0).
        exp_terms = np.exp(-beta * shifted)
        log_sum = np.log(np.sum(exp_terms))
        
        soft_min_val = m - (1.0 / beta) * log_sum
        return -soft_min_val

    def raw_objective(params):
        """
        Minimize -R where R = (d_min/d_max)^2
        Used for final polish.
        """
        points = params.reshape((n, 2))
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        ratio = d_min / d_max
        return -(ratio ** 2)

    # --- Initialization ---

    def generate_hex_grid(radius, spacing=1.0):
        points = []
        for q in range(-radius, radius + 1):
            r_min = max(-radius, -q - radius)
            r_max = min(radius, -q + radius)
            for r in range(r_min, r_max + 1):
                x = spacing * (q + 0.5 * r)
                y = spacing * (np.sqrt(3) / 2 * r)
                points.append([x, y])
        return np.array(points)

    starts = []

    # 1. Hexagonal lattice
    for radius in [3, 4]:
        hex_pts = generate_hex_grid(radius)
        hex_pts -= np.mean(hex_pts, axis=0)
        norm = np.max(np.linalg.norm(hex_pts, axis=1))
        if norm > 1e-10:
            hex_pts = hex_pts / norm
        dists_center = np.linalg.norm(hex_pts, axis=1)
        sorted_idx = np.argsort(dists_center)
        starts.append(hex_pts[sorted_idx[:16]])

    # 2. Circular arrangement
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    starts.append(circle)

    # 3. Random starts
    for seed in [1, 7, 13, 42, 99, 123, 456]:
        np.random.seed(seed)
        pts = np.random.uniform(-1, 1, (n, 2))
        norm = np.max(np.linalg.norm(pts, axis=1))
        if norm > 1e-10:
            pts = pts / norm
        starts.append(pts)

    # 4. Square grid
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    starts.append(grid)

    # 5. Two concentric circles
    angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    r1, r2 = 0.5, 1.0
    pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
    pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
    starts.append(np.vstack([pts1, pts2]))

    # --- Optimization Phase 1: L-BFGS-B on Soft-Min ---

    for i, start in enumerate(starts):
        if time_exceeded():
            break

        def callback_lbfgs(xk):
            return time_exceeded()

        try:
            res = minimize(
                soft_min_objective,
                start.ravel(),
                method='L-BFGS-B',
                options={
                    'maxiter': 1000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                },
                callback=callback_lbfgs
            )
            if not time_exceeded():
                update_best(res.x.reshape((n, 2)))
        except Exception:
            pass

    # --- Optimization Phase 2: Nelder-Mead on Raw Objective (Polish) ---
    # L-BFGS-B gets us close to the optimum of the smooth surrogate.
    # Now we use a derivative-free method on the actual non-smooth objective
    # to handle the "kinks" where the min/max switch.

    if not time_exceeded() and best_points is not None:
        # Refine the best point found so far
        def callback_nm(xk):
            return time_exceeded()

        try:
            res = minimize(
                raw_objective,
                best_points.ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-14,
                    'fatol': 1e-16,
                    'adaptive': True
                },
                callback=callback_nm
            )
            if not time_exceeded():
                update_best(res.x.reshape((n, 2)))
        except Exception:
            pass

        # Try a few perturbations of the best point and polish them
        for trial in range(10):
            if time_exceeded():
                break
            np.random.seed(2000 + trial)
            noise_scale = 0.02 / (trial + 1)
            noise = np.random.randn(n, 2) * noise_scale
            perturbed = best_points + noise
            
            # Normalize to keep scale reasonable
            norm = np.max(np.linalg.norm(perturbed, axis=1))
            if norm > 1e-10:
                perturbed = perturbed / norm

            try:
                res = minimize(
                    raw_objective,
                    perturbed.ravel(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 3000,
                        'xatol': 1e-14,
                        'fatol': 1e-16,
                        'adaptive': True
                    },
                    callback=callback_nm
                )
                if not time_exceeded():
                    update_best(res.x.reshape((n, 2)))
            except Exception:
                pass

    # Fallback
    if best_points is None:
        np.random.seed(0)
        best_points = np.random.randn(n, 2)

    return best_points

# EVOLVE-BLOCK-END
