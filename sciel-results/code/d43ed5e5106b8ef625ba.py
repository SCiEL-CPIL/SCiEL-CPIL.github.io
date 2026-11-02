import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def fibonacci_initial(phase=0.0):
        i = np.arange(N) + 0.5 + phase
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

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

        a = p[facets[:, 0]]
        b = p[facets[:, 1]]
        c = p[facets[:, 2]]

        v1 = a - b
        v2 = b - c
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1)

        valid = norm_n > 1e-12
        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)

        if np.any(valid):
            cc = n[valid] / norm_n[valid, np.newaxis]
            dots = np.sum(cc * a[valid], axis=1)
            cc[dots < 0] *= -1
            circumcenters[valid] = cc
            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)

        return facets, circumcenters, radii

    def compute_covering_radius(p):
        facets, circumcenters, radii = compute_hull_data(p)
        if facets is None:
            return float('inf')
        return np.max(radii)

    def compute_forces(p, beta=50.0, repulsion_strength=0.03):
        """Compute smooth forces from soft-max energy + repulsion (vectorized)."""
        facets, circumcenters, radii = compute_hull_data(p)
        if facets is None:
            return None, None, None, None

        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None, None

        r2 = radii * radii
        max_r2 = np.max(r2)
        exp_terms = np.exp(beta * (r2 - max_r2))
        sum_exp = np.sum(exp_terms)
        weights = (2.0 / beta) * radii * exp_terms / sum_exp

        # Vectorized force accumulation from facets
        forces = np.zeros_like(p)
        for k in range(n_facets):
            w = weights[k]
            if w < 1e-15:
                continue
            cc = circumcenters[k]
            for idx in facets[k]:
                forces[idx] += w * (cc - p[idx])

        # Repulsion force (vectorized)
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diffs * diffs, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        forces_rep = np.sum(diffs * inv_dist_sq[:, :, np.newaxis], axis=1)
        forces += repulsion_strength * forces_rep

        return forces, facets, circumcenters, radii

    def lloyd_iteration(pts, n_iters=30):
        """Lloyd's relaxation: move each point toward circumcenters of its adjacent facets."""
        pts = pts.copy()
        for _ in range(n_iters):
            try:
                facets, circumcenters, radii = compute_hull_data(pts)
                if facets is None:
                    break

                # For each point, find all facets it belongs to
                new_pts = np.zeros_like(pts)
                for i in range(N):
                    # Find facets containing point i
                    mask = np.any(facets == i, axis=1)
                    if np.sum(mask) > 0:
                        # Weight by facet radii (larger facets get more weight)
                        wts = radii[mask]
                        wts = wts / (np.sum(wts) + 1e-12)
                        new_pts[i] = np.sum(circumcenters[mask] * wts[:, np.newaxis], axis=0)
                    else:
                        new_pts[i] = pts[i]

                # Blend with original to avoid overshooting
                alpha = 0.5
                pts = normalize((1 - alpha) * pts + alpha * new_pts)
            except Exception:
                break

        return pts

    def repulsion_relax(pts, n_iters=800, alpha_start=0.08):
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
                new_pts = pts + step * forces_tan
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                new_pts = new_pts / norms
                new_energy = repulsion_energy(new_pts)
                if new_energy < current_energy:
                    pts = new_pts
                    current_energy = new_energy
                    improved = True
                    break
                step *= 0.5

            if not improved:
                # If no improvement, try smaller base alpha
                alpha *= 0.8
            else:
                alpha *= 1.01  # Slightly increase if successful

            alpha = min(alpha, 0.1)
            alpha = max(alpha, 1e-6)

            # Check time
            if time.time() > deadline - 5.0:
                break

        return pts

    def facet_targeted_refine(pts, n_iters=200):
        """
        Directly target the worst facet: move its vertices toward its circumcenter
        while keeping other points fixed, with small tangential adjustments.
        """
        pts = pts.copy()
        for it in range(n_iters):
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break

            max_rad = np.max(radii)
            if max_rad < 1e-10:
                break

            # Find the worst facet
            worst_idx = np.argmax(radii)
            worst_facet = facets[worst_idx]
            worst_cc = circumcenters[worst_idx]

            # Move vertices of worst facet toward its circumcenter
            for idx in worst_facet:
                diff = worst_cc - pts[idx]
                radial = np.dot(diff, pts[idx])
                tangential = diff - radial * pts[idx]
                pts[idx] += 0.1 * tangential

            # Also apply small repulsion to avoid clustering
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diffs * diffs, axis=2)
            np.fill_diagonal(dist_sq, 1.0)
            inv_dist_sq = 1.0 / dist_sq
            np.fill_diagonal(inv_dist_sq, 0.0)
            forces_rep = np.sum(diffs * inv_dist_sq[:, :, np.newaxis], axis=1)
            pts += 0.005 * forces_rep

            pts = normalize(pts)

            if time.time() > deadline - 5.0:
                break

        return pts

    def optimize_from_initial(pts_init, time_budget):
        """Run multi-phase optimization from a given initial configuration."""
        best_pts = pts_init.copy()
        best_radius = compute_covering_radius(best_pts)
        end_time = start_time + time_budget

        # Phase 0: Repulsion relaxation for better initial spread
        if time.time() < end_time - 5.0:
            pts_rep = repulsion_relax(best_pts, n_iters=300)
            rad_rep = compute_covering_radius(pts_rep)
            if rad_rep < best_radius:
                best_radius = rad_rep
                best_pts = pts_rep.copy()

        # Phase 1: Lloyd's relaxation (fast, effective for covering radius)
        if time.time() < end_time - 5.0:
            pts_lloyd = lloyd_iteration(best_pts, n_iters=20)
            rad_lloyd = compute_covering_radius(pts_lloyd)
            if rad_lloyd < best_radius:
                best_radius = rad_lloyd
                best_pts = pts_lloyd.copy()

        # Phase 2: Facet-targeted refinement
        if time.time() < end_time - 5.0:
            pts_ft = facet_targeted_refine(best_pts, n_iters=100)
            rad_ft = compute_covering_radius(pts_ft)
            if rad_ft < best_radius:
                best_radius = rad_ft
                best_pts = pts_ft.copy()

        # Phase 3: Coarse soft-max gradient descent
        pts = best_pts.copy()
        eta = 0.02
        beta = 25.0
        n_steps = 800

        for step in range(n_steps):
            if time.time() > end_time - 20.0:
                break

            forces, facets, circumcenters, radii = compute_forces(pts, beta)
            if forces is None:
                break

            pts += eta * forces
            pts = normalize(pts)

            if radii is not None:
                current_radius = np.max(radii)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()

            eta *= 0.997
            beta += 0.3

        # Phase 4: L-BFGS-B on soft-max surrogate for covering radius
        if time.time() < end_time - 15.0:
            pts = best_pts.copy()

            def objective_softmax(params, beta=50.0):
                X = params.reshape(N, 3)
                X = normalize(X)
                facets, circumcenters, radii = compute_hull_data(X)
                if facets is None:
                    return 1e10
                r2 = radii * radii
                max_r2 = np.max(r2)
                exp_terms = np.exp(beta * (r2 - max_r2))
                soft_max = (1.0 / beta) * np.log(np.sum(exp_terms)) + max_r2
                return soft_max

            def objective_sum_sq(params):
                X = params.reshape(N, 3)
                X = normalize(X)
                facets, circumcenters, radii = compute_hull_data(X)
                if facets is None:
                    return 1e10
                return np.sum(radii**2) + 0.005 * np.var(radii)

            for obj_func in [objective_softmax, objective_sum_sq]:
                if time.time() > end_time - 10.0:
                    break
                try:
                    res = minimize(obj_func, pts.flatten(), method='L-BFGS-B',
                                   options={'maxiter': 80, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100})
                    new_pts = normalize(res.x.reshape(N, 3))
                    current_radius = compute_covering_radius(new_pts)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
                        pts = new_pts.copy()
                except Exception:
                    pass

        # Phase 5: Direct minimax projected gradient with adaptive threshold
        if time.time() < end_time - 10.0:
            pts = best_pts.copy()
            eta = 0.003
            threshold_frac = 0.90
            n_steps_p3 = 1000

            for step in range(n_steps_p3):
                if time.time() > end_time - 7.0:
                    break

                facets, circumcenters, radii = compute_hull_data(pts)
                if facets is None:
                    break

                current_radius = np.max(radii)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()

                threshold = threshold_frac * current_radius
                active_facets = np.where(radii >= threshold)[0]

                if len(active_facets) == 0:
                    break

                forces = np.zeros_like(pts)

                # Use smooth quadratic weights
                excess = radii[active_facets] - threshold
                max_excess = np.max(excess) if len(excess) > 0 else 1e-12
                if max_excess < 1e-12:
                    max_excess = 1e-12
                facet_weights = (excess / max_excess) ** 2
                facet_weights = facet_weights / (np.sum(facet_weights) + 1e-12)

                for j, k in enumerate(active_facets):
                    w = facet_weights[j]
                    cc = circumcenters[k]
                    for idx in facets[k]:
                        diff = cc - pts[idx]
                        radial = np.dot(diff, pts[idx])
                        tangential = diff - radial * pts[idx]
                        forces[idx] += w * tangential

                # Add small repulsion to prevent clustering
                diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dist_sq = np.sum(diffs * diffs, axis=2)
                np.fill_diagonal(dist_sq, 1.0)
                inv_dist_sq = 1.0 / dist_sq
                np.fill_diagonal(inv_dist_sq, 0.0)
                forces_rep = np.sum(diffs * inv_dist_sq[:, :, np.newaxis], axis=1)
                forces += 0.01 * forces_rep

                for i in range(N):
                    f_norm = np.linalg.norm(forces[i])
                    if f_norm > 1e-12:
                        forces[i] /= f_norm

                pts += eta * forces
                pts = normalize(pts)

                if step % 200 == 0:
                    threshold_frac = min(0.97, threshold_frac + 0.005)
                eta *= 0.998

        # Phase 6: Fine soft-max polish with higher beta
        if time.time() < end_time - 5.0:
            pts = best_pts.copy()
            eta = 0.001
            beta = 100.0
            n_steps_p4 = 400

            for step in range(n_steps_p4):
                if time.time() > end_time - 3.0:
                    break

                forces, facets, circumcenters, radii = compute_forces(pts, beta)
                if forces is None:
                    break

                pts += eta * forces
                pts = normalize(pts)

                if radii is not None:
                    current_radius = np.max(radii)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = pts.copy()

                eta *= 0.997
                beta += 0.8

        # Phase 7: Final L-BFGS with multiple beta values
        if time.time() < end_time - 1.0:
            pts = best_pts.copy()
            for beta_val in [200.0, 500.0, 1000.0]:
                if time.time() > end_time - 0.5:
                    break
                try:
                    def obj_lbfgs(params, b=beta_val):
                        X = params.reshape(N, 3)
                        X = normalize(X)
                        facets, circumcenters, radii = compute_hull_data(X)
                        if facets is None:
                            return 1e10
                        r2 = radii * radii
                        max_r2 = np.max(r2)
                        exp_terms = np.exp(b * (r2 - max_r2))
                        soft_max = (1.0 / b) * np.log(np.sum(exp_terms)) + max_r2
                        return soft_max

                    res = minimize(obj_lbfgs, pts.flatten(), method='L-BFGS-B',
                                   options={'maxiter': 50, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100})
                    new_pts = normalize(res.x.reshape(N, 3))
                    current_radius = compute_covering_radius(new_pts)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
                        pts = new_pts.copy()
                except Exception:
                    pass

        return best_pts, best_radius

    # Multi-restart strategy
    # Restart 1: Fibonacci
    fib_pts = fibonacci_initial(0.0)
    best_pts, best_radius = optimize_from_initial(fib_pts, 85.0)

    # Restart 2: Fibonacci with phase shift
    if time.time() < deadline - 40.0:
        fib2 = fibonacci_initial(0.5)
        pts2, rad2 = optimize_from_initial(fib2, 25.0)
        if rad2 < best_radius:
            best_radius = rad2
            best_pts = pts2

    # Restart 3: Random perturbation of best
    if time.time() < deadline - 25.0:
        rng = np.random.default_rng(42)
        perturbed = best_pts + rng.normal(0, 0.03, (N, 3))
        perturbed = normalize(perturbed)
        pts3, rad3 = optimize_from_initial(perturbed, 18.0)
        if rad3 < best_radius:
            best_radius = rad3
            best_pts = pts3

    # Restart 4: Different random initial
    if time.time() < deadline - 15.0:
        rng = np.random.default_rng(123)
        random_pts = rng.normal(0, 1, (N, 3))
        random_pts = normalize(random_pts)
        pts4, rad4 = optimize_from_initial(random_pts, 10.0)
        if rad4 < best_radius:
            best_radius = rad4
            best_pts = pts4

    # Restart 5: Another perturbation with different seed
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(777)
        perturbed2 = best_pts + rng.normal(0, 0.015, (N, 3))
        perturbed2 = normalize(perturbed2)
        pts5, rad5 = optimize_from_initial(perturbed2, 7.0)
        if rad5 < best_radius:
            best_radius = rad5
            best_pts = pts5

    # Restart 6: Very small perturbation for fine-tuning
    if time.time() < deadline - 5.0:
        rng = np.random.default_rng(999)
        perturbed3 = best_pts + rng.normal(0, 0.005, (N, 3))
        perturbed3 = normalize(perturbed3)
        pts6, rad6 = optimize_from_initial(perturbed3, 4.0)
        if rad6 < best_radius:
            best_radius = rad6
            best_pts = pts6

    # Final high-precision L-BFGS polish with multiple beta values
    if time.time() < deadline - 1.0:
        for beta_val in [500.0, 1000.0, 2000.0]:
            if time.time() > deadline - 0.5:
                break
            try:
                def final_objective(params, b=beta_val):
                    X = params.reshape(N, 3)
                    X = normalize(X)
                    facets, circumcenters, radii = compute_hull_data(X)
                    if facets is None:
                        return 1e10
                    r2 = radii * radii
                    max_r2 = np.max(r2)
                    exp_terms = np.exp(b * (r2 - max_r2))
                    soft_max = (1.0 / b) * np.log(np.sum(exp_terms)) + max_r2
                    return soft_max

                res = minimize(final_objective, best_pts.flatten(), method='L-BFGS-B',
                               options={'maxiter': 60, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 150})
                new_pts = normalize(res.x.reshape(N, 3))
                current_radius = compute_covering_radius(new_pts)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = new_pts.copy()
            except Exception:
                pass

    # Final Lloyd polish
    if time.time() < deadline - 0.5:
        pts_lloyd_final = lloyd_iteration(best_pts, n_iters=10)
        rad_lloyd_final = compute_covering_radius(pts_lloyd_final)
        if rad_lloyd_final < best_radius:
            best_radius = rad_lloyd_final
            best_pts = pts_lloyd_final.copy()

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = fibonacci_initial(0.0)
        best_radius = compute_covering_radius(best_pts)

    best_pts = normalize(best_pts)
    return best_pts
