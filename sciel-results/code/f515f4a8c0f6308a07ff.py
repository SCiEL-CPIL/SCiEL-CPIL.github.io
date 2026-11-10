# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates.

    """
    start_time = time.time()
    deadline = 600  # seconds
    safety_margin = 30  # seconds

    n = 16
    d = 2

    def compute_R(points):
        """Compute the R metric directly."""
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(params):
        """Objective to minimize: -R (negate because we minimize)."""
        if time.time() > start_time + deadline - safety_margin:
            return 1e10
        points = params.reshape(n, d)
        R = compute_R(points)
        return -R

    def smooth_objective(params, beta=30.0):
        """
        Smooth surrogate using log-sum-exp for min/max.
        Minimizes log(d_max_smooth) - log(d_min_smooth).
        """
        if time.time() > start_time + deadline - safety_margin:
            return 1e10
        points = params.reshape(n, d)
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 1e10
        dists = np.maximum(dists, 1e-10)

        max_d = np.max(dists)
        min_d = np.min(dists)

        exp_pos = np.exp(beta * (dists - max_d))
        log_max_smooth = max_d + (1.0 / beta) * np.log(np.sum(exp_pos))

        exp_neg = np.exp(-beta * (dists - min_d))
        log_min_smooth = min_d - (1.0 / beta) * np.log(np.sum(exp_neg))

        return log_max_smooth - log_min_smooth

    best_params = None
    best_score = -1e10

    # Phase 1: Differential Evolution global search
    bounds = [(-3, 3)] * (n * d)
    
    try:
        de_result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=80,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.8,
            seed=42,
            polish=False,
            init='latinhypercube',
            update_interval=10,
            callback=lambda x, convergence: None
        )
        
        if time.time() > start_time + deadline - safety_margin:
            pass
        else:
            candidate_points = de_result.x.reshape(n, d)
            score = compute_R(candidate_points)
            if score > best_score:
                best_score = score
                best_params = de_result.x.copy()
    except Exception:
        pass

    # Phase 2: Multiple local refinements from diverse starting points
    rng = np.random.RandomState(123)
    
    # Starting points: DE result + structured patterns + random
    start_points_list = []
    
    if best_params is not None:
        start_points_list.append(best_params.copy())
    
    # 4x4 grid
    x = np.linspace(-2, 2, 4)
    y = np.linspace(-2, 2, 4)
    X, Y = np.meshgrid(x, y)
    start_points_list.append(np.column_stack((X.ravel(), Y.ravel())).flatten())
    
    # Circular arrangement
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    start_points_list.append(np.column_stack((1.5 * np.cos(angles), 1.5 * np.sin(angles))).flatten())
    
    # Two concentric circles
    angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    r1, r2 = 0.9, 2.2
    start_points_list.append(np.column_stack([
        np.concatenate([r1 * np.cos(angles1), r2 * np.cos(angles2)]),
        np.concatenate([r1 * np.sin(angles1), r2 * np.sin(angles2)])
    ]).flatten())
    
    # Hexagonal lattice
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    r_inner = 0.6
    r_outer = 2.0
    inner = np.column_stack([r_inner * np.cos(angles), r_inner * np.sin(angles)])
    outer = np.column_stack([r_outer * np.cos(angles + np.pi/6), r_outer * np.sin(angles + np.pi/6)])
    extra_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
    r_mid = 1.3
    mid = np.column_stack([r_mid * np.cos(extra_angles), r_mid * np.sin(extra_angles)])
    start_points_list.append(np.vstack([inner, outer, mid]).flatten())
    
    # Random starts
    for _ in range(3):
        start_points_list.append(rng.uniform(-3, 3, size=(n, d)).flatten())

    for init_params in start_points_list:
        if time.time() > start_time + deadline - safety_margin:
            break
        
        # L-BFGS-B with smooth objective
        try:
            result = minimize(
                smooth_objective,
                init_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 80}
            )
            if result.fun < 1e10:
                candidate_points = result.x.reshape(n, d)
                score = compute_R(candidate_points)
                if score > best_score:
                    best_score = score
                    best_params = result.x.copy()
        except Exception:
            pass
        
        # Powell refinement on the best from this start
        if best_params is not None:
            try:
                result_ref = minimize(
                    objective,
                    best_params,
                    method='Powell',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                candidate_points = result_ref.x.reshape(n, d)
                score = compute_R(candidate_points)
                if score > best_score:
                    best_score = score
                    best_params = result_ref.x.copy()
            except Exception:
                pass

    # Phase 3: Final high-precision polish
    if best_params is not None:
        try:
            result_final = minimize(
                objective,
                best_params,
                method='Powell',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            candidate_points = result_final.x.reshape(n, d)
            score = compute_R(candidate_points)
            if score > best_score:
                best_score = score
                best_params = result_final.x.copy()
        except Exception:
            pass

    # Convert back to points
    if best_params is not None:
        points = best_params.reshape(n, d)
    else:
        # Fallback: 4x4 grid
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack((X.ravel(), Y.ravel()))

    if points.shape != (n, d):
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack((X.ravel(), Y.ravel()))

    return points

# EVOLVE-BLOCK-END
