import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def compute_hull_data(p):
        """Compute hull facets and their circumcenters/radii (vectorized)."""
        try:
            hull = ConvexHull(p)
            facets = hull.simplices
        except Exception:
            return None, None, None

        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None

        # Vectorized circumcenter computation
        a = p[facets[:, 0]]  # (F, 3)
        b = p[facets[:, 1]]
        c = p[facets[:, 2]]

        v1 = a - b  # (F, 3)
        v2 = b - c
        n = np.cross(v1, v2)  # (F, 3)
        norm_n = np.linalg.norm(n, axis=1)  # (F,)

        # Handle degenerate facets
        valid = norm_n > 1e-12
        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)

        if np.any(valid):
            cc = n[valid] / norm_n[valid, np.newaxis]
            # Ensure circumcenter faces outward (toward 'a')
            dots = np.sum(cc * a[valid], axis=1)
            cc[dots < 0] *= -1
            circumcenters[valid] = cc

            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)

        return facets, circumcenters, radii

    def get_max_radius(pts):
        facets, circumcenters, radii = compute_hull_data(pts)
        if radii is None:
            return float('inf')
        return np.max(radii)

    def soft_max_objective(params, beta=200.0):
        """
        Soft-max surrogate: approximates max(radii) via log-sum-exp.
        As beta -> inf, this approaches the true max.
        """
        X = normalize(params.reshape(N, 3))

        facets, circumcenters, radii = compute_hull_data(X)
        if facets is None:
            return 1e10

        # Soft-max: (1/beta) * log(sum(exp(beta * radii)))
        r_max = np.max(radii)
        if r_max < 1e-15:
            return r_max
        shifted = beta * (radii - r_max)
        log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta + r_max
        return log_sum_exp

    def repulsion_relax(pts, n_iters=600, alpha_start=0.08):
        """
        Repulsion relaxation with adaptive step size (backtracking line search).
        Minimize sum 1/||p_i - p_j||^2 via gradient descent on sphere.
        """
        pts = pts.copy()
        alpha = alpha_start

        def repulsion_energy(p):
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            mask = ~np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
            return 0.5 * np.sum(inv_dist_sq)

        current_energy = repulsion_energy(pts)

        for it in range(n_iters):
            # Calculate pairwise differences
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)

            # Avoid division by zero
            mask = np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)

            # Force F_i = sum_j (p_i - p_j) / ||p_i - p_j||^2
            forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)

            # Project onto tangent plane: F_tan = F - (F . p) p
            dot_fp = np.sum(forces * pts, axis=1)
            forces_tan = forces - dot_fp[:, np.newaxis] * pts

            # Adaptive step size with backtracking
            step = alpha
            improved = False
            for _ in range(10):
                new_pts = normalize(pts + step * forces_tan)
                new_energy = repulsion_energy(new_pts)
                if new_energy < current_energy:
                    pts = new_pts
                    current_energy = new_energy
                    improved = True
                    break
                step *= 0.5

            if not improved:
                alpha *= 0.8
            else:
                alpha *= 1.01

            alpha = min(alpha, 0.1)
            alpha = max(alpha, 1e-6)

            if time.time() > deadline - 5.0:
                break

        return pts

    def lloyd_iteration(pts, n_iters=60):
        """
        Improved Lloyd's algorithm: move each point toward the circumcenter
        of its worst facet, with adaptive weighting.
        """
        pts = pts.copy()

        for it in range(n_iters):
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break

            max_rad = np.max(radii)
            if max_rad < 1e-10:
                break

            # For each point, accumulate weighted pull toward facet circumcenters
            updates = np.zeros_like(pts)
            counts = np.zeros(N)

            for f in range(len(facets)):
                facet_pts = facets[f]
                cc = circumcenters[f]
                rad = radii[f]
                if rad < 1e-12:
                    continue
                # Weight by how much this facet contributes to the max
                weight = (rad / max_rad) ** 2
                for idx in facet_pts:
                    updates[idx] += weight * (cc - pts[idx])
                    counts[idx] += weight

            # Apply updates where counts > 0
            mask = counts > 1e-10
            if np.any(mask):
                step_size = 0.05 * (max_rad / (np.mean(radii) + 1e-15))
                step_size = min(step_size, 0.1)  # Cap step size
                pts[mask] += step_size * updates[mask] / counts[mask, np.newaxis]

            # Normalize
            pts = normalize(pts)

            if time.time() > deadline - 5.0:
                break

        return pts

    def optimize_from_start(pts_init, beta, maxiter=80):
        """Run L-BFGS-B optimization on the soft-max surrogate."""
        best_pts_local = pts_init.copy()
        best_radius_local = get_max_radius(pts_init)

        pts_init = normalize(pts_init.copy())

        try:
            res = minimize(lambda p: soft_max_objective(p, beta), pts_init.flatten(),
                           method='L-BFGS-B',
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 200})

            if res.fun < soft_max_objective(pts_init.flatten(), beta):
                new_pts = normalize(res.x.reshape(N, 3))
                current_radius = get_max_radius(new_pts)
                if current_radius < best_radius_local:
                    best_pts_local = new_pts
                    best_radius_local = current_radius
        except Exception:
            pass

        return best_pts_local, best_radius_local

    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best_pts = pts.copy()
    best_radius = get_max_radius(pts)

    # Phase 0: initial repulsion relaxation to spread points
    if time.time() < deadline - 20.0:
        relaxed = repulsion_relax(pts, n_iters=300)
        rad = get_max_radius(relaxed)
        if rad < best_radius:
            best_radius = rad
            best_pts = relaxed.copy()
            pts = relaxed.copy()

    # Phase 1: Beta-annealing with Lloyd + L-BFGS polish
    # Anneal beta from low (smooth) to high (sharp) to escape local minima
    beta_schedule = [30.0, 60.0, 120.0, 200.0, 400.0, 800.0]

    for beta in beta_schedule:
        if time.time() > deadline - 15.0:
            break

        # Lloyd iteration to equalize radii
        lloyded = lloyd_iteration(pts, n_iters=40)
        rad = get_max_radius(lloyded)
        if rad < best_radius:
            best_radius = rad
            best_pts = lloyded.copy()
            pts = lloyded.copy()

        # L-BFGS polish on soft-max surrogate
        polished, pol_rad = optimize_from_start(pts, beta, maxiter=80)
        if pol_rad < best_radius:
            best_radius = pol_rad
            best_pts = polished.copy()
            pts = polished.copy()

    # Phase 2: Fine repulsion relaxation + Lloyd to squeeze out improvement
    if time.time() < deadline - 10.0:
        relaxed = repulsion_relax(pts, n_iters=200, alpha_start=0.03)
        rad = get_max_radius(relaxed)
        if rad < best_radius:
            best_radius = rad
            best_pts = relaxed.copy()
            pts = relaxed.copy()

        lloyded = lloyd_iteration(pts, n_iters=50)
        rad = get_max_radius(lloyded)
        if rad < best_radius:
            best_radius = rad
            best_pts = lloyded.copy()
            pts = lloyded.copy()

    # Phase 3: Final high-precision L-BFGS polish on the sharpest surrogate
    if time.time() < deadline - 3.0:
        polished, pol_rad = optimize_from_start(pts, 1500.0, maxiter=100)
        if pol_rad < best_radius:
            best_radius = pol_rad
            best_pts = polished.copy()

    # Final validation: ensure points are valid
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    return best_pts
