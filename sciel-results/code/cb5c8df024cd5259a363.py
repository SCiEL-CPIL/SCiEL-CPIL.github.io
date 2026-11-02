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

        # Use soft-max on radii directly (not r2) for better conditioning
        max_r = np.max(radii)
        if max_r < 1e-12:
            return np.zeros_like(p), facets, circumcenters, radii
        exp_terms = np.exp(beta * (radii - max_r))
        sum_exp = np.sum(exp_terms)
        weights = (1.0 / beta) * radii * exp_terms / sum_exp

        # Vectorized force accumulation from facets
        forces = np.zeros_like(p)
        vertex_indices = facets.ravel()
        facet_indices = np.repeat(np.arange(n_facets), 3)
        w_expanded = weights[facet_indices]
        cc_expanded = circumcenters[facet_indices]
        p_expanded = p[vertex_indices]
        
        contributions = w_expanded[:, np.newaxis] * (cc_expanded - p_expanded)
        np.add.at(forces, vertex_indices, contributions)

        if repulsion_strength > 0:
            diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist_sq = np.sum(diffs * diffs, axis=2)
            np.fill_diagonal(dist_sq, 1.0)
            inv_dist_sq = 1.0 / dist_sq
            np.fill_diagonal(inv_dist_sq, 0.0)
            forces_rep = np.sum(diffs * inv_dist_sq[:, :, np.newaxis], axis=1)
            forces += repulsion_strength * forces_rep

        return forces, facets, circumcenters, radii

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
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            
            mask = np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
            
            forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
            
            dot_fp = np.sum(forces * pts, axis=1)
            forces_tan = forces - dot_fp[:, np.newaxis] * pts
            
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
                alpha *= 0.8
            else:
                alpha *= 1.01
            
            alpha = min(alpha, 0.1)
            alpha = max(alpha, 1e-6)
            
            if time.time() > deadline - 5.0:
                break
        
        return pts

    def lloyd_iteration(pts, n_iters=50):
        """
        Improved Lloyd's algorithm: move each point toward the circumcenter
        of its worst facet, with adaptive step size.
        """
        pts = pts.copy()
        
        for it in range(n_iters):
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break
            
            max_rad = np.max(radii)
            if max_rad < 1e-10:
                break
            
            updates = np.zeros_like(pts)
            counts = np.zeros(N)
            
            for f in range(len(facets)):
                facet_pts = facets[f]
                cc = circumcenters[f]
                rad = radii[f]
                if rad < 1e-12:
                    continue
                weight = (rad / max_rad) ** 2
                for idx in facet_pts:
                    updates[idx] += weight * (cc - pts[idx])
                    counts[idx] += weight
            
            mask = counts > 1e-10
            if np.any(mask):
                step_size = 0.05 * (max_rad / (np.mean(radii) + 1e-15))
                step_size = min(step_size, 0.1)
                pts[mask] += step_size * updates[mask] / counts[mask, np.newaxis]
            
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            pts = pts / norms
            
            if time.time() > deadline - 5.0:
                break
        
        return pts

    def facet_targeted_local_search(pts, n_iters=200, time_budget=10.0):
        """
        Direct facet-based local search: for the worst facets, move their
        vertices toward the circumcenter with backtracking line search.
        This directly reduces the nonsmooth covering radius.
        """
        pts = pts.copy()
        end_time = time.time() + time_budget
        best_radius = compute_covering_radius(pts)
        best_pts = pts.copy()
        
        for it in range(n_iters):
            if time.time() > end_time - 0.5:
                break
            
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break
            
            current_radius = np.max(radii)
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
            
            # Find the worst facet(s)
            worst_idx = np.argmax(radii)
            worst_rad = radii[worst_idx]
            
            if worst_rad < 1e-10:
                break
            
            # Try moving vertices of the worst facet toward its circumcenter
            facet = facets[worst_idx]
            cc = circumcenters[worst_idx]
            
            # Also consider near-worst facets
            threshold = 0.95 * worst_rad
            active = np.where(radii >= threshold)[0]
            
            improved = False
            for f_idx in active:
                if time.time() > end_time - 0.5:
                    break
                f = facets[f_idx]
                cc_f = circumcenters[f_idx]
                rad_f = radii[f_idx]
                
                # Try moving each vertex toward circumcenter
                for v_idx in f:
                    p_v = pts[v_idx]
                    # Direction: from vertex to circumcenter, projected tangent
                    diff = cc_f - p_v
                    radial = np.dot(diff, p_v)
                    tangent = diff - radial * p_v
                    t_norm = np.linalg.norm(tangent)
                    if t_norm < 1e-12:
                        continue
                    tangent /= t_norm
                    
                    # Backtracking line search
                    step = 0.1 * (rad_f / worst_rad)
                    for _ in range(15):
                        new_pts = pts.copy()
                        new_pts[v_idx] = p_v + step * tangent
                        new_pts[v_idx] /= np.linalg.norm(new_pts[v_idx])
                        
                        new_radius = compute_covering_radius(new_pts)
                        if new_radius < current_radius - 1e-12:
                            pts = new_pts
                            current_radius = new_radius
                            improved = True
                            break
                        step *= 0.5
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                # Try small random perturbations of worst facet vertices
                rng = np.random.default_rng(hash((it, worst_idx)) % (2**32))
                for f_idx in active[:3]:
                    if time.time() > end_time - 0.5:
                        break
                    f = facets[f_idx]
                    for v_idx in f:
                        p_v = pts[v_idx]
                        # Random tangent direction
                        rand_dir = rng.normal(0, 1, 3)
                        radial = np.dot(rand_dir, p_v)
                        tangent = rand_dir - radial * p_v
                        t_norm = np.linalg.norm(tangent)
                        if t_norm < 1e-12:
                            continue
                        tangent /= t_norm
                        
                        step = 0.01
                        for _ in range(10):
                            new_pts = pts.copy()
                            new_pts[v_idx] = p_v + step * tangent
                            new_pts[v_idx] /= np.linalg.norm(new_pts[v_idx])
                            
                            new_radius = compute_covering_radius(new_pts)
                            if new_radius < current_radius - 1e-12:
                                pts = new_pts
                                current_radius = new_radius
                                improved = True
                                break
                            step *= 0.5
                        
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved and it > 20:
                break
        
        return best_pts, best_radius

    def optimize_from_initial(pts_init, time_budget):
        """Run multi-phase optimization from a given initial configuration."""
        best_pts = pts_init.copy()
        best_radius = compute_covering_radius(best_pts)
        end_time = start_time + time_budget

        # Phase 1: Coarse soft-max gradient descent
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

        # Phase 2: L-BFGS-B on soft-max surrogate for covering radius
        if time.time() < end_time - 15.0:
            pts = best_pts.copy()

            def objective_softmax(params, beta=50.0):
                X = params.reshape(N, 3)
                X = normalize(X)
                facets, circumcenters, radii = compute_hull_data(X)
                if facets is None:
                    return 1e10
                max_r = np.max(radii)
                if max_r < 1e-12:
                    return 0.0
                exp_terms = np.exp(beta * (radii - max_r))
                soft_max = (1.0 / beta) * np.log(np.sum(exp_terms)) + max_r
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
                                   options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 150})
                    new_pts = normalize(res.x.reshape(N, 3))
                    current_radius = compute_covering_radius(new_pts)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = new_pts.copy()
                        pts = new_pts.copy()
                except Exception:
                    pass

        # Phase 3: Direct minimax projected gradient with adaptive threshold (vectorized)
        if time.time() < end_time - 10.0:
            pts = best_pts.copy()
            eta = 0.003
            threshold_frac = 0.90
            n_steps_p3 = 1500

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

                excess = radii[active_facets] - threshold
                max_excess = np.max(excess) if len(excess) > 0 else 1e-12
                if max_excess < 1e-12:
                    max_excess = 1e-12
                facet_weights = (excess / max_excess) ** 2
                facet_weights = facet_weights / (np.sum(facet_weights) + 1e-12)

                af = active_facets
                w_exp = facet_weights
                cc_exp = circumcenters[af]
                verts = facets[af]
                p_verts = pts[verts]
                
                diff = cc_exp[:, np.newaxis, :] - p_verts
                radial = np.sum(diff * p_verts, axis=2)
                tangential = diff - radial[:, :, np.newaxis] * p_verts
                contrib = w_exp[:, np.newaxis, np.newaxis] * tangential
                
                np.add.at(forces, verts.ravel(), contrib.reshape(-1, 3))

                if step % 5 == 0:
                    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                    dist_sq = np.sum(diffs * diffs, axis=2)
                    np.fill_diagonal(dist_sq, 1.0)
                    inv_dist_sq = 1.0 / dist_sq
                    np.fill_diagonal(inv_dist_sq, 0.0)
                    forces_rep = np.sum(diffs * inv_dist_sq[:, :, np.newaxis], axis=1)
                    forces += 0.01 * forces_rep

                f_norms = np.linalg.norm(forces, axis=1, keepdims=True)
                f_norms[f_norms < 1e-12] = 1.0
                forces /= f_norms

                pts += eta * forces
                pts = normalize(pts)

                if step % 150 == 0:
                    threshold_frac = min(0.97, threshold_frac + 0.005)
                eta *= 0.998

        # Phase 4: Fine soft-max polish with higher beta (vectorized forces)
        if time.time() < end_time - 5.0:
            pts = best_pts.copy()
            eta = 0.001
            beta = 100.0
            n_steps_p4 = 600

            for step in range(n_steps_p4):
                if time.time() > end_time - 3.0:
                    break

                forces, facets, circumcenters, radii = compute_forces(pts, beta, repulsion_strength=0.01 if step % 5 == 0 else 0.0)
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

        # Phase 5: Facet-targeted local search (direct minimax)
        if time.time() < end_time - 5.0:
            pts = best_pts.copy()
            try:
                facet_pts, facet_rad = facet_targeted_local_search(pts, n_iters=150, time_budget=5.0)
                if facet_rad < best_radius:
                    best_radius = facet_rad
                    best_pts = facet_pts.copy()
                    pts = facet_pts.copy()
            except Exception:
                pass

        # Phase 6: Final L-BFGS with multiple beta values
        if time.time() < end_time - 1.0:
            pts = best_pts.copy()
            for beta_val in [300.0, 800.0, 2000.0, 5000.0]:
                if time.time() > end_time - 0.5:
                    break
                try:
                    def obj_lbfgs(params, b=beta_val):
                        X = params.reshape(N, 3)
                        X = normalize(X)
                        facets, circumcenters, radii = compute_hull_data(X)
                        if facets is None:
                            return 1e10
                        max_r = np.max(radii)
                        if max_r < 1e-12:
                            return 0.0
                        exp_terms = np.exp(b * (radii - max_r))
                        soft_max = (1.0 / b) * np.log(np.sum(exp_terms)) + max_r
                        return soft_max

                    res = minimize(obj_lbfgs, pts.flatten(), method='L-BFGS-B',
                                   options={'maxiter': 100, 'ftol': 1e-17, 'gtol': 1e-16, 'maxls': 200})
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
        perturbed = best_pts + rng.normal(0, 0.05, (N, 3))
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
        perturbed2 = best_pts + rng.normal(0, 0.025, (N, 3))
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

    # Final facet-targeted local search
    if time.time() < deadline - 5.0:
        try:
            facet_pts, facet_rad = facet_targeted_local_search(best_pts, n_iters=200, time_budget=5.0)
            if facet_rad < best_radius:
                best_radius = facet_rad
                best_pts = facet_pts.copy()
        except Exception:
            pass

    # Final high-precision L-BFGS polish with multiple beta values
    if time.time() < deadline - 1.0:
        for beta_val in [1000.0, 3000.0, 8000.0, 20000.0]:
            if time.time() > deadline - 0.5:
                break
            try:
                def final_objective(params, b=beta_val):
                    X = params.reshape(N, 3)
                    X = normalize(X)
                    facets, circumcenters, radii = compute_hull_data(X)
                    if facets is None:
                        return 1e10
                    max_r = np.max(radii)
                    if max_r < 1e-12:
                        return 0.0
                    exp_terms = np.exp(b * (radii - max_r))
                    soft_max = (1.0 / b) * np.log(np.sum(exp_terms)) + max_r
                    return soft_max

                res = minimize(final_objective, best_pts.flatten(), method='L-BFGS-B',
                               options={'maxiter': 150, 'ftol': 1e-18, 'gtol': 1e-17, 'maxls': 300})
                new_pts = normalize(res.x.reshape(N, 3))
                current_radius = compute_covering_radius(new_pts)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = new_pts.copy()
            except Exception:
                pass

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = fibonacci_initial(0.0)
        best_radius = compute_covering_radius(best_pts)

    best_pts = normalize(best_pts)
    return best_pts
