# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    def exact_score(pts):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape != (n, d):
            pts = pts.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        d_min = np.min(dists)
        return (d_min / d_max) ** 2

    def exact_score_flat(x):
        return exact_score(x.reshape(n, d))

    def exact_objective(x):
        return -exact_score_flat(x)

    def soft_min(dists, t=0.05):
        """Log-sum-exp approximation to the minimum"""
        return -np.log(np.sum(np.exp(-dists / t)) / len(dists)) * t

    def soft_max(dists, t=0.05):
        """Log-sum-exp approximation to the maximum"""
        return np.log(np.sum(np.exp(dists / t)) / len(dists)) * t

    def smooth_objective(x, t=0.05):
        """Smooth approximation: minimize soft_max - soft_min of distances"""
        pts = x.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        dists_norm = dists / d_max
        return -soft_min(dists_norm, t)

    def smooth_objective_v2(x, t=0.05):
        """Alternative: maximize soft_min of normalized distances"""
        pts = x.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        dists_norm = dists / d_max
        return -soft_min(dists_norm, t)

    best_x = None
    best_score = 0.0

    def update_best(x, score):
        nonlocal best_x, best_score
        if score > best_score:
            best_score = score
            best_x = x.copy()

    # Phase 0: Try known good configurations as starting points
    # 4x4 grid
    grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    grid -= grid.mean(axis=0)
    dmax = np.max(pdist(grid))
    if dmax > 0:
        grid /= dmax
    update_best(grid.ravel(), exact_score(grid))

    # Hexagonal lattice: 4 rows, 4 cols with offset
    hex_pts = []
    for i in range(4):
        for j in range(4):
            x = j + 0.5 * (i % 2)
            y = i * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts -= hex_pts.mean(axis=0)
    dmax = np.max(pdist(hex_pts))
    if dmax > 0:
        hex_pts /= dmax
    update_best(hex_pts.ravel(), exact_score(hex_pts))

    # Ring configuration: 1 center + 15 on circle
    ring = np.zeros((16, 2))
    angles = np.linspace(0, 2 * np.pi, 15, endpoint=False)
    ring[1:, 0] = np.cos(angles)
    ring[1:, 1] = np.sin(angles)
    update_best(ring.ravel(), exact_score(ring))

    # Two rings: 8 inner + 8 outer
    two_ring = np.zeros((16, 2))
    angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles2 = np.linspace(2 * np.pi / 8, 2 * np.pi, 8, endpoint=False)
    two_ring[:8, 0] = 0.5 * np.cos(angles1)
    two_ring[:8, 1] = 0.5 * np.sin(angles1)
    two_ring[8:, 0] = np.cos(angles2)
    two_ring[8:, 1] = np.sin(angles2)
    update_best(two_ring.ravel(), exact_score(two_ring))

    # Three rings: 1 center + 5 inner + 10 outer
    three_ring = np.zeros((16, 2))
    angles_inner = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    angles_outer = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    three_ring[1:6, 0] = 0.4 * np.cos(angles_inner)
    three_ring[1:6, 1] = 0.4 * np.sin(angles_inner)
    three_ring[6:, 0] = np.cos(angles_outer)
    three_ring[6:, 1] = np.sin(angles_outer)
    update_best(three_ring.ravel(), exact_score(three_ring))

    # Phase 1: Differential Evolution (global search) with wider bounds
    if time.time() < deadline - 150:
        try:
            res_de = differential_evolution(
                exact_objective,
                [(-2, 2)] * (n * d),
                maxiter=200,
                popsize=30,
                seed=42,
                tol=1e-5,
                init='latinhypercube',
                polish=False,
                mutation=(0.5, 1.0),
                recombination=0.8
            )
            if time.time() < deadline - 20:
                s = -res_de.fun
                if s > best_score:
                    update_best(res_de.x, s)
        except Exception:
            pass

    # Phase 2: L-BFGS-B on smooth surrogate for gradient-based convergence
    if best_x is not None and time.time() < deadline - 100:
        for t in [0.1, 0.05, 0.02]:
            if time.time() > deadline - 20:
                break
            try:
                res_lbfgs = minimize(
                    lambda x: smooth_objective(x, t=t),
                    best_x,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if time.time() > deadline - 15:
                    break
                s = exact_score_flat(res_lbfgs.x)
                if s > best_score:
                    update_best(res_lbfgs.x, s)
            except Exception:
                continue

    # Phase 3: Nelder-Mead local refinement on exact objective
    if best_x is not None and time.time() < deadline - 60:
        try:
            res_nm = minimize(
                exact_objective,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 8000,
                    'xatol': 1e-9,
                    'fatol': 1e-11,
                    'adaptive': True
                }
            )
            if time.time() < deadline - 10:
                s = -res_nm.fun
                if s > best_score:
                    update_best(res_nm.x, s)
        except Exception:
            pass

    # Phase 4: Multi-start with diverse initializations
    if time.time() < deadline - 50:
        # Generate diverse starting points
        starts = []

        # Random starts
        np.random.seed(123)
        for _ in range(8):
            x = np.random.uniform(-1.5, 1.5, (n * d))
            starts.append(x)

        # Circle with noise
        for noise_scale in [0.05, 0.1, 0.2]:
            x = np.zeros(n * d)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            x[0::2] = np.cos(angles) + np.random.normal(0, noise_scale, n)
            x[1::2] = np.sin(angles) + np.random.normal(0, noise_scale, n)
            starts.append(x)

        # Grid with noise
        for noise_scale in [0.05, 0.1, 0.15]:
            x = np.zeros(n * d)
            for i in range(4):
                for j in range(4):
                    idx = i * 4 + j
                    x[idx * 2] = i + np.random.normal(0, noise_scale)
                    x[idx * 2 + 1] = j + np.random.normal(0, noise_scale)
            starts.append(x)

        # Hex lattice with noise
        for noise_scale in [0.05, 0.1, 0.15]:
            x = np.zeros(n * d)
            for i in range(4):
                for j in range(4):
                    idx = i * 4 + j
                    px = j + 0.5 * (i % 2)
                    py = i * np.sqrt(3) / 2
                    x[idx * 2] = px + np.random.normal(0, noise_scale)
                    x[idx * 2 + 1] = py + np.random.normal(0, noise_scale)
            starts.append(x)

        # Perturbed best
        if best_x is not None:
            for scale in [0.05, 0.1, 0.2]:
                x = best_x + np.random.normal(0, scale, (n * d))
                starts.append(x)

        for x_start in starts:
            if time.time() > deadline - 15:
                break
            try:
                # First try L-BFGS-B on smooth objective
                res_smooth = minimize(
                    lambda x: smooth_objective(x, t=0.05),
                    x_start,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12}
                )
                if time.time() > deadline - 12:
                    break
                s = exact_score_flat(res_smooth.x)
                if s > best_score:
                    update_best(res_smooth.x, s)

                # Then Nelder-Mead on exact
                res_nm = minimize(
                    exact_objective,
                    res_smooth.x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 3000,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if time.time() > deadline - 10:
                    break
                s = -res_nm.fun
                if s > best_score:
                    update_best(res_nm.x, s)
            except Exception:
                continue

    # Phase 5: Final polish with L-BFGS-B using tighter soft-min
    if best_x is not None and time.time() < deadline - 15:
        for t in [0.02, 0.01, 0.005]:
            if time.time() > deadline - 10:
                break
            try:
                res_lbfgs = minimize(
                    lambda x: smooth_objective(x, t=t),
                    best_x,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14}
                )
                if time.time() > deadline - 8:
                    break
                s = exact_score_flat(res_lbfgs.x)
                if s > best_score:
                    update_best(res_lbfgs.x, s)
            except Exception:
                continue

    # Final Nelder-Mead polish
    if best_x is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                exact_objective,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            s = -res_final.fun
            if s > best_score:
                update_best(res_final.x, s)
        except Exception:
            pass

    if best_x is None:
        # Fallback: 4x4 grid
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        grid -= grid.mean(axis=0)
        dmax = np.max(pdist(grid))
        if dmax > 0:
            grid /= dmax
        best_x = grid.ravel()

    return best_x.reshape(n, d)

# EVOLVE-BLOCK-END
