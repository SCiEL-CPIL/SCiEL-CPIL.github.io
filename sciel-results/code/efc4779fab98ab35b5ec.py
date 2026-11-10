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
    deadline = start_time + 580  # 50s safety margin

    def get_distances(points):
        return scipy.spatial.distance.pdist(points)

    def compute_score(points):
        dists = get_distances(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective_neg(points_flat):
        points = points_flat.reshape(n, d)
        return -compute_score(points)

    def soft_min_max_objective(points_flat, tau=0.05):
        """
        Smooth surrogate: minimize -(log(d_min) - log(d_max))
        using soft-min and soft-max for differentiability.
        """
        points = points_flat.reshape(n, d)
        dists = get_distances(points)
        if np.any(dists <= 1e-12):
            return 1e6
        
        # Soft min: -tau * log(sum(exp(-dists/tau)))
        # Soft max:  tau * log(sum(exp(dists/tau)))
        exp_neg = np.exp(-dists / tau)
        exp_pos = np.exp(dists / tau)
        
        # Clip to avoid overflow
        max_exp_neg = np.max(exp_neg)
        max_exp_pos = np.max(exp_pos)
        
        soft_min = -tau * (np.log(np.sum(exp_neg / max_exp_neg)) + np.log(max_exp_neg))
        soft_max = tau * (np.log(np.sum(exp_pos / max_exp_pos)) + np.log(max_exp_pos))
        
        return -(soft_min - soft_max)

    def normalize_points(points):
        """Center and normalize points so max distance = 1."""
        points = points - np.mean(points, axis=0)
        dists = scipy.spatial.distance.pdist(points)
        d_max = np.max(dists)
        if d_max > 0:
            points = points / d_max
        return points

    # Generate diverse initializations
    def hexagonal_initial(n_points):
        points = np.zeros((n_points, 2))
        spacing = 1.0
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                x = j * spacing + (i % 2) * spacing * 0.5
                y = i * spacing * np.sqrt(3) / 2
                points[idx] = [x, y]
        return normalize_points(points)

    def circular_initial(n_points):
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        return points

    def grid_initial(n_points):
        points = np.zeros((n_points, 2))
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                points[idx] = [j, i]
        return normalize_points(points)

    def fibonacci_spiral_initial(n_points):
        points = np.zeros((n_points, 2))
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_points):
            r = np.sqrt(i + 1) / np.sqrt(n_points)
            theta = i * golden_angle
            points[i] = [r * np.cos(theta), r * np.sin(theta)]
        return normalize_points(points)

    def random_initial(n_points, seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState(int(time.time() * 1000) % 2**31)
        points = rng.randn(n_points, d)
        return normalize_points(points)

    def two_ring_initial(n_points):
        """8 points on inner ring, 8 on outer ring"""
        points = np.zeros((n_points, 2))
        # Inner ring: 8 points at radius 0.5
        for i in range(8):
            angle = i * 2 * np.pi / 8
            points[i] = [0.5 * np.cos(angle), 0.5 * np.sin(angle)]
        # Outer ring: 8 points at radius 1.0
        for i in range(8):
            angle = (i + 0.5) * 2 * np.pi / 8
            points[8 + i] = [np.cos(angle), np.sin(angle)]
        return normalize_points(points)

    initializations = [
        hexagonal_initial(n),
        circular_initial(n),
        grid_initial(n),
        fibonacci_spiral_initial(n),
        two_ring_initial(n),
    ]
    
    # Add several random initializations
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        initializations.append(random_initial(n, seed=seed))

    best_points = None
    best_score = -1.0

    # Phase 1: For each initialization, run L-BFGS-B on the soft surrogate
    for init_idx, init_points in enumerate(initializations):
        if time.time() > deadline - 60:
            break

        current_points = init_points.copy()
        current_score = compute_score(current_points)

        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()

        # L-BFGS-B on soft surrogate
        if time.time() < deadline - 30:
            try:
                result = scipy.optimize.minimize(
                    soft_min_max_objective,
                    current_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                refined_points = result.x.reshape(n, d)
                refined_points = normalize_points(refined_points)
                refined_score = compute_score(refined_points)

                if refined_score > best_score:
                    best_score = refined_score
                    best_points = refined_points.copy()
            except Exception:
                pass

    # Phase 2: For the best point, do a fine-tune with smaller tau
    if time.time() < deadline - 30 and best_points is not None:
        try:
            result = scipy.optimize.minimize(
                lambda x: soft_min_max_objective(x, tau=0.02),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            refined_points = result.x.reshape(n, d)
            refined_points = normalize_points(refined_points)
            refined_score = compute_score(refined_points)

            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_points.copy()
        except Exception:
            pass

    # Phase 3: Final Nelder-Mead polish on the exact objective
    if time.time() < deadline - 15 and best_points is not None:
        try:
            result = scipy.optimize.minimize(
                objective_neg,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            final_points = result.x.reshape(n, d)
            final_points = normalize_points(final_points)
            final_score = compute_score(final_points)

            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
        except Exception:
            pass

    # Phase 4: One more L-BFGS-B with very small tau if time permits
    if time.time() < deadline - 5 and best_points is not None:
        try:
            result = scipy.optimize.minimize(
                lambda x: soft_min_max_objective(x, tau=0.01),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-11}
            )
            final_points = result.x.reshape(n, d)
            final_points = normalize_points(final_points)
            final_score = compute_score(final_points)

            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
        except Exception:
            pass

    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)

    return best_points


# EVOLVE-BLOCK-END
