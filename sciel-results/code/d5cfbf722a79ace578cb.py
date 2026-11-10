import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    start_time = time.time()
    deadline = 590.0  # safety margin from 630s limit

    # Precompute pair indices for vectorized distance computation
    pair_i = np.array([i for i in range(n) for j in range(i + 1, n)])
    pair_j = np.array([j for i in range(n) for j in range(i + 1, n)])
    npairs = len(pair_i)

    def compute_score(pts):
        if pts.shape != (16, 2):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def normalize_pts(pts):
        pts = pts - np.mean(pts, axis=0)
        dm = np.max(pdist(pts))
        if dm > 1e-15:
            pts = pts / dm
        return pts

    def distances_from_x(x):
        pts = x.reshape(-1, 2)
        di = pts[pair_i]
        dj = pts[pair_j]
        return np.linalg.norm(di - dj, axis=1)

    def smooth_max_log(dists, scale):
        scaled = scale * dists
        max_s = np.max(scaled)
        return max_s / scale + np.log(np.sum(np.exp(scaled - max_s))) / scale

    def smooth_min_log(dists, scale):
        scaled = -scale * dists
        max_s = np.max(scaled)
        return -(max_s / scale + np.log(np.sum(np.exp(scaled - max_s))) / scale)

    def objective(x, scale=20.0):
        dists = distances_from_x(x)
        if np.min(dists) < 1e-12:
            return 1e6
        return -smooth_min_log(dists, scale) + smooth_max_log(dists, scale)

    def gradient(x, scale=20.0):
        pts = x.reshape(-1, 2)
        di = pts[pair_i]
        dj = pts[pair_j]
        vecs = di - dj
        dists = np.linalg.norm(vecs, axis=1)
        safe_dists = np.maximum(dists, 1e-12)

        scaled_max = scale * dists
        max_sm = np.max(scaled_max)
        exp_max = np.exp(scaled_max - max_sm)
        w_max = exp_max / np.sum(exp_max)

        scaled_min = -scale * dists
        max_sm2 = np.max(scaled_min)
        exp_min = np.exp(scaled_min - max_sm2)
        v_min = exp_min / np.sum(exp_min)

        coeffs = (w_max - v_min) / safe_dists
        grad = np.zeros_like(x)
        for k in range(npairs):
            c = coeffs[k]
            if abs(c) > 1e-15:
                grad[2 * pair_i[k]:2 * pair_i[k] + 2] += c * vecs[k]
                grad[2 * pair_j[k]:2 * pair_j[k] + 2] -= c * vecs[k]
        return grad

    def local_optimize(x0, scale=20.0, max_iter=200):
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           jac=lambda x: gradient(x, scale),
                           args=(scale,),
                           options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            return res.x
        except Exception:
            return x0

    def simulated_annealing(pts, max_iters=2000, T_start=0.05, T_end=0.0005):
        """SA with fast vectorized scoring."""
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        cur_score = best_score

        for it in range(max_iters):
            if time.time() - start_time > deadline - 20:
                break

            T = T_start * (T_end / T_start) ** (it / max(1, max_iters - 1))
            delta = np.random.randn(n, 2) * T
            new_pts = pts + delta
            new_score = compute_score(new_pts)

            if new_score >= cur_score:
                pts = new_pts
                cur_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
            else:
                accept_prob = np.exp((new_score - cur_score) / max(T, 1e-10))
                if np.random.random() < accept_prob:
                    pts = new_pts
                    cur_score = new_score

        return best_pts, best_score

    def improve_from_init(init_pts):
        """SA → multi-scale gradient → SA → gradient pipeline."""
        if time.time() - start_time > deadline - 30:
            return init_pts, compute_score(init_pts)

        pts = normalize_pts(init_pts)
        score = compute_score(pts)

        # Phase 1: SA for global exploration
        sa_pts, sa_score = simulated_annealing(pts, max_iters=2500, T_start=0.06, T_end=0.0003)
        if sa_score > score:
            pts = sa_pts
            score = sa_score

        # Phase 2: Multi-scale smooth gradient refinement
        if time.time() - start_time > deadline - 25:
            return pts, score
        x = pts.flatten()
        for scale in [15.0, 30.0, 60.0, 120.0]:
            if time.time() - start_time > deadline - 20:
                break
            x = local_optimize(x, scale=scale, max_iter=150)
            pts = normalize_pts(x.reshape(-1, 2))
            new_score = compute_score(pts)
            if new_score > score:
                score = new_score

        # Phase 3: Second SA round with smaller perturbations
        if time.time() - start_time > deadline - 15:
            return pts, score
        sa2_pts, sa2_score = simulated_annealing(pts, max_iters=1500, T_start=0.02, T_end=0.0001)
        if sa2_score > score:
            pts = sa2_pts
            score = sa2_score

        # Phase 4: Final gradient polish
        if time.time() - start_time > deadline - 10:
            return pts, score
        x = pts.flatten()
        for scale in [80.0, 150.0, 300.0]:
            if time.time() - start_time > deadline - 8:
                break
            x = local_optimize(x, scale=scale, max_iter=100)
            pts = normalize_pts(x.reshape(-1, 2))
            new_score = compute_score(pts)
            if new_score > score:
                score = new_score

        return pts, score

    # Generate diverse initializations
    np.random.seed(42)
    initializations = []

    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    initializations.append(np.column_stack([np.cos(angles), np.sin(angles)]))

    # 2. Two-ring (inner 6, outer 10)
    inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    inner_pts = np.column_stack([0.45 * np.cos(inner_angles), 0.45 * np.sin(inner_angles)])
    outer_angles = np.linspace(0, 2 * np.pi, 10, endpoint=False) + np.pi / 10
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    initializations.append(np.vstack([inner_pts, outer_pts]))

    # 3. Three-ring
    r1 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    r2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    r3 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    initializations.append(np.vstack([
        np.column_stack([0.3 * np.cos(r1), 0.3 * np.sin(r1)]),
        np.column_stack([0.65 * np.cos(r2), 0.65 * np.sin(r2)]),
        np.column_stack([np.cos(r3), np.sin(r3)])
    ]))

    # 4. Hexagonal lattice subset
    hex_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            hex_pts.append([i + 0.5 * j, j * np.sqrt(3) / 2])
    hex_pts = np.array(hex_pts)
    dc = np.linalg.norm(hex_pts, axis=1)
    idx = np.argsort(dc)[:n]
    initializations.append(hex_pts[idx])

    # 5. Grid 4x4
    gx, gy = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    initializations.append(np.column_stack([gx.ravel(), gy.ravel()]))

    # 6. Perturbed circles
    for _ in range(4):
        angles_p = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.random.randn(n) * 0.2
        radii = 1 + 0.2 * np.random.randn(n)
        initializations.append(np.column_stack([np.cos(angles_p) * radii, np.sin(angles_p) * radii]))

    # 7. Random
    for _ in range(6):
        initializations.append(np.random.randn(n, 2))

    # 8. Two-ring variations
    for r_inner in [0.3, 0.5, 0.6]:
        ia = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        oa = np.linspace(0, 2 * np.pi, 10, endpoint=False) + np.pi / 10
        initializations.append(np.vstack([
            np.column_stack([r_inner * np.cos(ia), r_inner * np.sin(ia)]),
            np.column_stack([np.cos(oa), np.sin(oa)])
        ]))

    # 9. More structured: 4 rings of 4
    for r in [0.25, 0.5, 0.75, 1.0]:
        a = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi * (r - 0.25)
        initializations.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    # Combine all 4 rings into one init
    all_rings = []
    for r in [0.25, 0.5, 0.75, 1.0]:
        a = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi * (r - 0.25)
        all_rings.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    initializations.append(np.vstack(all_rings))

    # Process initializations
    best_score = 0.0
    best_points = None

    for idx_init, init in enumerate(initializations):
        if time.time() - start_time > deadline - 35:
            break

        pts, score = improve_from_init(init)

        if score > best_score:
            best_score = score
            best_points = pts.copy()

        # If we have time, do one more SA+gradient cycle from the best
        if idx_init % 3 == 0 and time.time() - start_time < deadline - 40:
            pts2, score2 = improve_from_init(pts)
            if score2 > best_score:
                best_score = score2
                best_points = pts2.copy()

    # Final polish from best point
    if best_points is not None and time.time() - start_time < deadline - 15:
        x = best_points.flatten()
        for scale in [100.0, 200.0, 400.0]:
            if time.time() - start_time > deadline - 10:
                break
            x = local_optimize(x, scale=scale, max_iter=100)
            pts = normalize_pts(x.reshape(-1, 2))
            score = compute_score(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()

        # One final tiny SA
        if time.time() - start_time < deadline - 8:
            sa_pts, sa_score = simulated_annealing(best_points, max_iters=500, T_start=0.01, T_end=0.00005)
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_pts.copy()

    if best_points is None:
        best_points = normalize_pts(np.random.randn(n, 2))

    # Ensure valid output
    assert best_points.shape == (16, 2)
    assert np.all(np.isfinite(best_points))

    return best_points
