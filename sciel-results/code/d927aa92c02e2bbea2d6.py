import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3

    start_time = time.time()
    deadline = start_time + 575  # Safety margin for 600s limit

    def score(pts):
        """Compute the metric (d_min/d_max)^2"""
        try:
            if not np.all(np.isfinite(pts)):
                return 0.0
            dists = pdist(pts)
            if len(dists) == 0:
                return 0.0
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max <= 1e-12 or not np.isfinite(d_min) or not np.isfinite(d_max):
                return 0.0
            r = d_min / d_max
            return float(r * r)
        except Exception:
            return 0.0

    def neg_score(pts_flat):
        """Negative score for minimization"""
        pts = pts_flat.reshape((n, d))
        return -score(pts)

    def soft_max(y, beta):
        """Stable soft-max via log-sum-exp"""
        m = np.max(y)
        if m > 100:
            return m + np.log(np.sum(np.exp(beta * (y - m)))) / beta
        return (1.0 / beta) * np.log(np.sum(np.exp(beta * (y - m))))

    def soft_min(y, beta):
        """Stable soft-min via log-sum-exp"""
        m = np.min(y)
        if m < -100:
            return m - np.log(np.sum(np.exp(-beta * (y - m)))) / beta
        return -(1.0 / beta) * np.log(np.sum(np.exp(-beta * y)))

    def smooth_objective(x, beta):
        """Smooth surrogate: log(d_max) - log(d_min) using soft min/max on log-distances"""
        if time.time() > deadline:
            return np.inf

        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)

        try:
            dists = pdist(pts)
        except Exception:
            return np.inf

        if np.any(~np.isfinite(dists)) or np.any(dists <= 1e-9):
            return np.inf

        log_dists = np.log(dists)
        val = soft_max(log_dists, beta) - soft_min(log_dists, beta)
        return val

    def icosahedron():
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        norms = np.linalg.norm(verts, axis=1)
        return verts / norms[:, np.newaxis]

    def fibonacci_sphere(num_points):
        """Generate points on a sphere using the Fibonacci lattice."""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(num_points):
            z = 1 - (i / (num_points - 1)) * 2
            radius = np.sqrt(1 - z * z)
            theta = np.pi * i / golden_ratio
            x = np.cos(theta) * radius
            y = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    def rotate_to_z(v, target_z):
        """Rotate vector v to align with target_z direction."""
        v = v / np.linalg.norm(v)
        target = target_z / np.linalg.norm(target_z)
        axis = np.cross(v, target)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-10:
            if np.dot(v, target) < 0:
                # Opposite direction, pick any perpendicular axis
                if abs(v[0]) < 0.9:
                    perp = np.array([1.0, 0, 0])
                else:
                    perp = np.array([0, 1.0, 0])
                axis = np.cross(v, perp)
                axis = axis / np.linalg.norm(axis)
                angle = np.pi
            else:
                return np.eye(3)
        else:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(v, target), -1.0, 1.0))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R

    def generate_seeds():
        seeds = []
        ico = icosahedron()
        z_axis = np.array([0, 0, 1.0])

        # Symmetry-based seeds: icosahedron vertices + 2 axial points
        # 5-fold axis (align ico[0] to z)
        R5 = rotate_to_z(ico[0], z_axis)
        ico_5 = ico @ R5.T
        for r in [0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0]:
            p1 = np.array([0, 0, r])
            p2 = np.array([0, 0, -r])
            seeds.append(np.vstack([ico_5, p1, p2]))

        # 3-fold axis (align mean of ico[0], ico[1], ico[10] to z)
        c3 = (ico[0] + ico[1] + ico[10]) / 3.0
        R3 = rotate_to_z(c3, z_axis)
        ico_3 = ico @ R3.T
        for r in [0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0]:
            p1 = np.array([0, 0, r])
            p2 = np.array([0, 0, -r])
            seeds.append(np.vstack([ico_3, p1, p2]))

        # 2-fold axis (align ico[4] to z)
        R2 = rotate_to_z(ico[4], z_axis)
        ico_2 = ico @ R2.T
        for r in [0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0]:
            p1 = np.array([0, 0, r])
            p2 = np.array([0, 0, -r])
            seeds.append(np.vstack([ico_2, p1, p2]))

        # Fibonacci sphere
        seeds.append(fibonacci_sphere(14))

        # Random sphere points with various noise levels
        np.random.seed(42)
        for seed_idx in range(8):
            np.random.seed(42 + seed_idx)
            pts = np.random.randn(n, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            noise = np.random.randn(n, 3) * 0.15
            pts = pts + noise
            seeds.append(pts)

        # Points on two parallel rings (torus-like)
        for ring_r in [0.8, 1.0, 1.2]:
            for z_sep in [0.3, 0.5, 0.8]:
                pts = []
                for i in range(7):
                    theta = 2 * np.pi * i / 7
                    pts.append([ring_r * np.cos(theta), ring_r * np.sin(theta), z_sep])
                for i in range(7):
                    theta = 2 * np.pi * i / 7 + np.pi / 7
                    pts.append([ring_r * np.cos(theta), ring_r * np.sin(theta), -z_sep])
                seeds.append(np.array(pts))

        return seeds

    seeds = generate_seeds()

    best_points = None
    best_score_val = 0.0

    for s in seeds:
        sc = score(s)
        if sc > best_score_val:
            best_score_val = sc
            best_points = s.copy()

    # Step 1: Differential Evolution for global search
    bounds = [(-2.0, 2.0)] * (n * d)

    if time.time() < deadline - 150:
        try:
            np.random.seed(42)
            de_res = differential_evolution(
                neg_score,
                bounds,
                seed=42,
                maxiter=300,
                tol=1e-6,
                polish=False,
                init='sobol',
                mutation=(0.5, 1.5),
                recombination=0.7,
                popsize=20
            )

            de_pts = de_res.x.reshape((n, d))
            de_score = score(de_pts)

            if de_score > best_score_val:
                best_score_val = de_score
                best_points = de_pts.copy()
        except Exception:
            pass

    # Step 2: Local Refinement with multi-beta annealing via L-BFGS-B
    candidates = []
    if best_points is not None:
        candidates.append(best_points)

    # Add top seeds
    seed_scores = [(score(s), s) for s in seeds]
    seed_scores.sort(key=lambda x: x[0], reverse=True)
    for sc, s in seed_scores[:8]:
        if not any(np.allclose(s, c, atol=1e-6) for c in candidates):
            candidates.append(s)

    betas = [8, 20, 50, 100, 250, 500, 1000]

    for cand in candidates:
        if time.time() > deadline - 30:
            break

        current_x = cand.flatten()
        current_smooth_val = np.inf

        try:
            current_smooth_val = smooth_objective(current_x, 8)
        except Exception:
            pass

        for beta in betas:
            if time.time() > deadline - 15:
                break

            try:
                res = minimize(
                    smooth_objective,
                    current_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if np.isfinite(res.fun) and res.fun < current_smooth_val:
                    current_smooth_val = res.fun
                    current_x = res.x.copy()
            except Exception:
                pass

        final_pts = current_x.reshape((n, d))
        final_score = score(final_pts)
        if final_score > best_score_val:
            best_score_val = final_score
            best_points = final_pts.copy()

    # Step 3: Nelder-Mead polish with adaptive step
    if best_points is not None and time.time() < deadline - 20:
        try:
            nm_res = minimize(
                neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                    'initial_simplex': best_points.flatten() + 0.01 * np.random.randn(n * d)
                }
            )
            nm_pts = nm_res.x.reshape((n, d))
            nm_score = score(nm_pts)
            if nm_score > best_score_val:
                best_score_val = nm_score
                best_points = nm_pts.copy()
        except Exception:
            pass

    # Step 4: High-precision final polish using COBYLA with tight tolerances
    if best_points is not None and time.time() < deadline - 10:
        try:
            res = minimize(
                neg_score,
                best_points.flatten(),
                method='COBYLA',
                options={
                    'maxiter': 5000,
                    'tol': 1e-12
                }
            )
            c_pts = res.x.reshape((n, d))
            c_score = score(c_pts)
            if c_score > best_score_val:
                best_score_val = c_score
                best_points = c_pts.copy()
        except Exception:
            pass

    # Fallback
    if best_points is None:
        best_points = seeds[0]

    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = seeds[0]

    # Center the points
    best_points = best_points - np.mean(best_points, axis=0)

    return best_points
