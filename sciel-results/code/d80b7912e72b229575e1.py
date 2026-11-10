"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def compute_score(points):
        n = len(points)
        if n < 2:
            return 0.0
        # Compute all pairwise distances using cdist for efficiency
        dist = cdist(points, points)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        # Compute minimum distance to boundary
        radii = np.linalg.norm(points, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        return min(min_pair_dist / 2.0, min_boundary_dist)

    def smooth_min(a, b, beta=50.0):
        """Smooth approximation of min(a, b) using log-sum-exp trick."""
        m = np.minimum(a, b)
        return m - np.logaddexp(-beta * (a - m), -beta * (b - m)) / beta

    # Generate initial hexagonal lattice
    def gen_hex_lattice(spacing):
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing * 0.5:
                    pts.append((x, y))
        return np.array(pts)

    # Initial spacing estimate
    initial_spacing = 2.0 * np.sqrt(0.907 / N)

    # Generate initial hexagonal lattice
    spacing = initial_spacing
    while True:
        pts = gen_hex_lattice(spacing)
        if len(pts) >= N:
            break
        spacing *= 0.98

    # Select N points closest to center
    radii = np.hypot(pts[:, 0], pts[:, 1])
    points = pts[np.argsort(radii, kind="stable")[:N]].copy()

    # Force-directed optimization with better parameters
    best_score = compute_score(points)
    best_points = points.copy()

    # Iterative optimization
    step_size = spacing * 0.1
    lr_decay = 0.9995
    max_iter = 2000

    for iteration in range(max_iter):
        if time.time() > deadline - 30:  # Leave time for final polish
            break

        # Compute forces
        n = len(points)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)

        # Avoid division by zero
        dist[dist < 1e-10] = 1e-10

        # Mask diagonal to avoid self-interaction
        mask = ~np.eye(n, dtype=bool)
        
        # Only consider close pairs for efficiency
        close_mask = mask & (dist < 5 * spacing)
        inv_dist_sq = np.zeros_like(dist)
        inv_dist_sq[close_mask] = 1.0 / (dist[close_mask] ** 2)
        
        # Repulsion forces
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)

        # Boundary constraint force - stronger
        radii = np.linalg.norm(points, axis=1)
        boundary_force = np.zeros_like(points)
        boundary_margin = 0.1
        over = radii > (1.0 - boundary_margin)
        if np.any(over):
            mag = (radii[over] - (1.0 - boundary_margin)) * 5.0
            boundary_force[over] = -mag[:, np.newaxis] * points[over] / radii[over, np.newaxis]
            
        forces += boundary_force

        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_norms)
        if max_force > 1e-10:
            forces /= max_force

        # Update positions
        points += step_size * forces

        # Project back inside unit circle with margin
        radii = np.linalg.norm(points, axis=1)
        mask_out = radii > 1.0
        if np.any(mask_out):
            points[mask_out] /= radii[mask_out, np.newaxis]

        # Compute score
        current_score = compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()

        # Anneal step size
        step_size *= lr_decay

        # If step size gets too small, we're converged
        if step_size < 1e-6:
            break

    # Local optimization using scipy for the best pairs
    if time.time() < deadline - 20:
        n = len(best_points)
        # Get the k closest pairs
        diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        # Find the critical pairs (smallest distances)
        flat_dist = dist.flatten()
        k = min(30, n * (n - 1) // 2)
        smallest_indices = np.argpartition(flat_dist, k)[:k]
        
        # Extract pairs
        pairs = []
        for idx in smallest_indices:
            i, j = np.unravel_index(idx, (n, n))
            if i < j:
                pairs.append((i, j))
        
        # Also check boundary violations
        radii = np.linalg.norm(best_points, axis=1)
        boundary_violators = np.where(radii > 0.95)[0]
        
        # Use scipy to optimize with smooth surrogate
        def objective(x):
            pts = x.reshape(-1, 2)
            # Compute pairwise distances
            d = cdist(pts, pts)
            np.fill_diagonal(d, np.inf)
            
            # Smooth minimum of all pair distances / 2
            # Use log-sum-exp for numerical stability
            beta = 100.0
            d_half = d / 2.0
            m = np.min(d_half)
            # soft-min: m - log(sum(exp(-beta*(d_half - m))))/beta
            exp_terms = np.exp(-beta * (d_half - m))
            soft_min_pairs = m - np.log(np.sum(exp_terms)) / beta
            
            # Boundary distances
            r = np.linalg.norm(pts, axis=1)
            bdist = 1.0 - r
            m_b = np.min(bdist)
            exp_terms_b = np.exp(-beta * (bdist - m_b))
            soft_min_boundary = m_b - np.log(np.sum(exp_terms_b)) / beta
            
            # Smooth min of the two
            return smooth_min(soft_min_pairs, soft_min_boundary, beta)
        
        def objective_neg(x):
            return -objective(x)

        x0 = best_points.flatten()
        
        # Try multiple optimization strategies
        best_local_points = best_points.copy()
        best_local_score = compute_score(best_points)
        
        # Strategy 1: L-BFGS-B with bounds
        bounds = [(-1.0, 1.0)] * (2 * n)
        try:
            result1 = minimize(objective_neg, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10})
            if result1.success or result1.fun < objective_neg(x0):
                new_points = result1.x.reshape(-1, 2)
                # Project back inside
                radii = np.linalg.norm(new_points, axis=1)
                mask_out = radii > 1.0
                if np.any(mask_out):
                    new_points[mask_out] /= radii[mask_out, np.newaxis]
                new_score = compute_score(new_points)
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_local_points = new_points
        except Exception:
            pass
        
        # Strategy 2: Nelder-Mead on critical points only (if time allows)
        if time.time() < deadline - 10:
            # Identify the most critical points (those involved in smallest distances or near boundary)
            crit_points = set()
            for i, j in pairs:
                crit_points.add(i)
                crit_points.add(j)
            for idx in boundary_violators:
                crit_points.add(idx)
            
            if 0 < len(crit_points) < n:
                crit_indices = sorted(crit_points)
                n_crit = len(crit_indices)
                
                def crit_objective(x):
                    new_pts = best_local_points.copy()
                    new_pts[crit_indices] = x.reshape(-1, 2)
                    return -compute_score(new_pts)
                
                x0_crit = best_local_points[crit_indices].flatten()
                try:
                    result2 = minimize(crit_objective, x0_crit, method='Nelder-Mead',
                                      options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                    if result2.success or result2.fun < crit_objective(x0_crit):
                        new_pts = best_local_points.copy()
                        new_pts[crit_indices] = result2.x.reshape(-1, 2)
                        # Project back inside
                        radii = np.linalg.norm(new_pts, axis=1)
                        mask_out = radii > 1.0
                        if np.any(mask_out):
                            new_pts[mask_out] /= radii[mask_out, np.newaxis]
                        new_score = compute_score(new_pts)
                        if new_score > best_local_score:
                            best_local_score = new_score
                            best_local_points = new_pts
                except Exception:
                    pass
        
        if best_local_score > best_score:
            best_score = best_local_score
            best_points = best_local_points

    # Final high-precision polish pass
    if time.time() < deadline - 5:
        n = len(best_points)
        # Use very tight tolerance optimization on a subset of most critical points
        diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        # Find the 10 smallest distances
        flat_dist = dist.flatten()
        k_small = min(10, n * (n - 1) // 2)
        smallest_indices = np.argpartition(flat_dist, k_small)[:k_small]
        
        critical_points = set()
        for idx in smallest_indices:
            i, j = np.unravel_index(idx, (n, n))
            critical_points.add(i)
            critical_points.add(j)
        
        # Add boundary-critical points
        radii = np.linalg.norm(best_points, axis=1)
        boundary_crit = np.where(radii > 0.93)[0]
        for idx in boundary_crit:
            critical_points.add(idx)
        
        if 0 < len(critical_points) < n:
            crit_indices = sorted(critical_points)
            n_crit = len(crit_indices)
            
            def polish_objective(x):
                new_pts = best_points.copy()
                new_pts[crit_indices] = x.reshape(-1, 2)
                return -compute_score(new_pts)
            
            x0_polish = best_points[crit_indices].flatten()
            try:
                result_polish = minimize(polish_objective, x0_polish, method='Nelder-Mead',
                                        options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14,
                                                  'adaptive': True})
                if result_polish.success or result_polish.fun < polish_objective(x0_polish):
                    new_pts = best_points.copy()
                    new_pts[crit_indices] = result_polish.x.reshape(-1, 2)
                    # Project back inside
                    radii = np.linalg.norm(new_pts, axis=1)
                    mask_out = radii > 1.0
                    if np.any(mask_out):
                        new_pts[mask_out] /= radii[mask_out, np.newaxis]
                    new_score = compute_score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_pts
            except Exception:
                pass

    return best_points
# EVOLVE-BLOCK-END
