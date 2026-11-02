import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety

    # Precompute indices for all triples
    triples = list(combinations(range(N), 3))
    n_triples = len(triples)
    i_idx = np.array([t[0] for t in triples])
    j_idx = np.array([t[1] for t in triples])
    k_idx = np.array([t[2] for t in triples])

    def compute_hard_score(points):
        """Compute exact min triangle area / convex hull area."""
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0

        if area_hull < 1e-9:
            return 0.0

        x = points[:, 0]
        y = points[:, 1]
        v1x = x[j_idx] - x[i_idx]
        v1y = y[j_idx] - y[i_idx]
        v2x = x[k_idx] - x[i_idx]
        v2y = y[k_idx] - y[i_idx]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        min_area = np.min(areas)
        score = min_area / area_hull
        return score

    def soft_objective(points_flat, beta=50.0):
        """
        Negative soft-min score for minimization.
        points_flat: 1D array of length 2N
        """
        points = points_flat.reshape(-1, 2)
        
        # Try to compute hull area
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 1e6

        if area_hull < 1e-9:
            return 1e6

        x = points[:, 0]
        y = points[:, 1]
        v1x = x[j_idx] - x[i_idx]
        v1y = y[j_idx] - y[i_idx]
        v2x = x[k_idx] - x[i_idx]
        v2y = y[k_idx] - y[i_idx]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        
        # Log-sum-exp for soft min
        # We want to maximize min(areas) / area_hull
        # So we minimize - (soft_min(areas) / area_hull)
        # soft_min(areas) = -1/beta * log(sum(exp(-beta * areas)))
        
        # Numerical stability
        min_area_val = np.min(areas)
        # If all areas are very large, exp(-beta * areas) might underflow to 0
        # Shift by min area to keep values in a reasonable range
        shifted = areas - min_area_val
        # exp(-beta * (areas - min_area)) = exp(-beta * areas) * exp(beta * min_area)
        # sum(exp(-beta * areas)) = exp(-beta * min_area) * sum(exp(-beta * (areas - min_area)))
        # log(sum(exp(-beta * areas))) = -beta * min_area + log(sum(exp(-beta * (areas - min_area))))
        
        exp_terms = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_terms)
        
        if sum_exp == 0:
            # All areas are effectively infinite? Unlikely unless beta is huge and areas are huge.
            # Fallback to hard min
            soft_min_area = min_area_val
        else:
            log_sum = np.log(sum_exp)
            # soft_min = -1/beta * ( -beta * min_area + log_sum_exp )
            # soft_min = min_area - (1/beta) * log_sum_exp
            soft_min_area = min_area_val - (1.0 / beta) * log_sum

        score = soft_min_area / area_hull
        return -score

    def make_symmetric_points(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner):
        """Generate points with two concentric regular polygons."""
        if n_inner == 0:
            angles = np.linspace(rot_hull, rot_hull + 2*np.pi, n_hull, endpoint=False)
            xs = r_hull * np.cos(angles)
            ys = r_hull * np.sin(angles)
            return np.column_stack([xs, ys])
        
        angles_hull = np.linspace(rot_hull, rot_hull + 2*np.pi, n_hull, endpoint=False)
        angles_inner = np.linspace(rot_inner, rot_inner + 2*np.pi, n_inner, endpoint=False)
        
        xs = np.concatenate([r_hull * np.cos(angles_hull), r_inner * np.cos(angles_inner)])
        ys = np.concatenate([r_hull * np.sin(angles_hull), r_inner * np.sin(angles_inner)])
        return np.column_stack([xs, ys])

    def symmetric_objective(params, n_hull, n_inner, beta=50.0):
        """Objective for symmetric parameterization."""
        r_hull, r_inner, rot_hull, rot_inner = params
        # Constraints to keep geometry valid
        if r_hull < 0.5 or r_inner < 0.0 or r_inner > r_hull:
            return 1e6
        points = make_symmetric_points(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner)
        return soft_objective(points.flatten(), beta)

    best_points = None
    best_score = -float('inf')

    # Strategy 1: Symmetric Optimization
    # Try different splits of points between outer and inner rings
    splits = [(13, 0), (12, 1), (11, 2), (10, 3), (9, 4), (8, 5), (7, 6)]

    for n_hull, n_inner in splits:
        if time.time() > start_time + time_limit - 20:
            break

        # Initial guess
        r_hull_init = 1.0
        r_inner_init = 0.5 if n_inner > 0 else 0.0
        rot_hull_init = 0.0
        rot_inner_init = np.pi / n_inner if n_inner > 1 else 0.0

        x0 = np.array([r_hull_init, r_inner_init, rot_hull_init, rot_inner_init])
        
        # Annealing beta
        for beta in [20.0, 50.0, 100.0]:
            if time.time() > start_time + time_limit - 20:
                break
            try:
                result = opt.minimize(
                    lambda x, b=beta, nh=n_hull, ni=n_inner: symmetric_objective(x, nh, ni, b),
                    x0,
                    method='L-BFGS-B',
                    bounds=[(0.5, 2.0), (0.0, 1.5), (-np.pi, np.pi), (-np.pi, np.pi)],
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                x0 = result.x
                points = make_symmetric_points(n_hull, n_inner, *x0)
                s = compute_hard_score(points)
                if s > best_score:
                    best_score = s
                    best_points = points.copy()
            except Exception:
                pass

        # Quick Nelder-Mead polish on symmetric params
        if time.time() > start_time + time_limit - 20:
            break
        try:
            res = opt.minimize(
                lambda x, nh=n_hull, ni=n_inner: symmetric_objective(x, nh, ni, 200.0),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 50, 'adaptive': True}
            )
            x0 = res.x
            points = make_symmetric_points(n_hull, n_inner, *x0)
            s = compute_hard_score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
        except Exception:
            pass

    # Strategy 2: Free-space optimization from best symmetric result
    if best_points is not None and time.time() < start_time + time_limit - 15:
        current = best_points.copy()
        
        # Multi-beta annealing in free space
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > start_time + time_limit - 15:
                break
            try:
                result = opt.minimize(
                    lambda x, b=beta: soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                optimized = result.x.reshape(-1, 2)
                opt_score = compute_hard_score(optimized)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = optimized.copy()
                current = optimized
            except Exception:
                pass

    # Strategy 3: Random restarts in free space
    if time.time() < start_time + time_limit - 15:
        np.random.seed(42)
        # Limit restarts to fit in time
        max_restarts = 5
        for _ in range(max_restarts):
            if time.time() > start_time + time_limit - 15:
                break
            
            # Generate random configuration
            # Mix of hull and interior
            n_hull = np.random.randint(9, 13)
            n_inner = N - n_hull
            r_hull = np.random.uniform(0.9, 1.1)
            r_inner = np.random.uniform(0.3, 0.6) if n_inner > 0 else 0.0
            rot_hull = np.random.uniform(0, 2*np.pi)
            rot_inner = np.random.uniform(0, 2*np.pi) if n_inner > 1 else 0.0
            
            config = make_symmetric_points(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner)
            # Add noise
            config = config + np.random.normal(0, 0.02, config.shape)

            s = compute_hard_score(config)
            if s > best_score:
                best_score = s
                best_points = config.copy()

            # Quick optimization
            if time.time() > start_time + time_limit - 15:
                break

            current = config.copy()
            for beta in [100.0]:
                if time.time() > start_time + time_limit - 15:
                    break
                try:
                    result = opt.minimize(
                        lambda x, b=beta: soft_objective(x, b),
                        current.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    optimized = result.x.reshape(-1, 2)
                    opt_score = compute_hard_score(optimized)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = optimized.copy()
                    current = optimized
                except Exception:
                    pass

    # Final high-precision polish using Nelder-Mead on hard score
    if best_points is not None and time.time() < start_time + time_limit - 5:
        try:
            # Use a high beta soft objective to guide NM, or just hard?
            # Hard is better for final squeeze, but NM on hard can be slow.
            # Let's try NM on soft with high beta first, then hard if time permits.
            
            # Actually, just use hard score for the final polish as it's the true objective.
            # NM is derivative-free and handles non-smoothness better than L-BFGS.
            
            def hard_neg_score(points_flat):
                points = points_flat.reshape(-1, 2)
                return -compute_hard_score(points)

            res = opt.minimize(
                hard_neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 100, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            final_points = res.x.reshape(-1, 2)
            final_score = compute_hard_score(final_points)
            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback: regular 13-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points

# EVOLVE-BLOCK-END
