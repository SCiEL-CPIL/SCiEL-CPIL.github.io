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
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        radii = np.linalg.norm(points, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        return min(min_pair_dist / 2.0, min_boundary_dist)

    def smooth_min(a, b, beta=50.0):
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

    # Ring-based initialization
    def gen_ring_config():
        """Generate ring-based initial configuration."""
        r_est = np.sqrt(0.907 / N)
        
        pts = [(0.0, 0.0)]  # Center
        
        best_ring_config = None
        best_ring_score = 0.0
        
        for n_rings in range(5, 15):
            radii = np.linspace(0.3, 0.95, n_rings)
            
            weights = radii
            weights = weights / weights.sum()
            counts = np.round(weights * (N - 1)).astype(int)
            diff = (N - 1) - counts.sum()
            if diff > 0:
                order = np.argsort(-radii)
                for i in range(diff):
                    counts[order[i % n_rings]] += 1
            elif diff < 0:
                order = np.argsort(radii)
                for i in range(-diff):
                    if counts[order[i % n_rings]] > 1:
                        counts[order[i % n_rings]] -= 1
            
            ring_pts = [pts[0]]
            for k in range(n_rings):
                r_k = radii[k]
                n_k = counts[k]
                if n_k <= 0:
                    continue
                offset = k * np.pi / max(n_k, 1)
                angles = offset + 2 * np.pi * np.arange(n_k) / n_k
                for ang in angles:
                    x = r_k * np.cos(ang)
                    y = r_k * np.sin(ang)
                    if np.hypot(x, y) <= 1.0:
                        ring_pts.append((x, y))
            
            if len(ring_pts) >= N:
                ring_pts = ring_pts[:N]
                ring_arr = np.array(ring_pts)
                s = compute_score(ring_arr)
                if s > best_ring_score:
                    best_ring_score = s
                    best_ring_config = ring_arr
        
        if best_ring_config is not None:
            return best_ring_config
        return None

    # Try ring-based first, fallback to hex
    points = gen_ring_config()
    if points is None:
        initial_spacing = 2.0 * np.sqrt(0.907 / N)
        spacing = initial_spacing
        while True:
            pts = gen_hex_lattice(spacing)
            if len(pts) >= N:
                break
            spacing *= 0.98
        radii = np.hypot(pts[:, 0], pts[:, 1])
        points = pts[np.argsort(radii, kind="stable")[:N]].copy()

    best_score = compute_score(points)
    best_points = points.copy()

    # Force-directed optimization
    spacing = 2.0 * np.sqrt(0.907 / N)
    step_size = spacing * 0.1
    lr_decay = 0.9995
    max_iter = 2000

    for iteration in range(max_iter):
        if time.time() > deadline - 30:
            break

        n = len(points)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        dist[dist < 1e-10] = 1e-10
        mask = ~np.eye(n, dtype=bool)
        close_mask = mask & (dist < 5 * spacing)
        inv_dist_sq = np.zeros_like(dist)
        inv_dist_sq[close_mask] = 1.0 / (dist[close_mask] ** 2)
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)

        radii = np.linalg.norm(points, axis=1)
        boundary_force = np.zeros_like(points)
        boundary_margin = 0.1
        over = radii > (1.0 - boundary_margin)
        if np.any(over):
            mag = (radii[over] - (1.0 - boundary_margin)) * 5.0
            boundary_force[over] = -mag[:, np.newaxis] * points[over] / radii[over, np.newaxis]
        forces += boundary_force

        force_norms = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_norms)
        if max_force > 1e-10:
            forces /= max_force

        points += step_size * forces

        radii = np.linalg.norm(points, axis=1)
        mask_out = radii > 1.0
        if np.any(mask_out):
            points[mask_out] /= radii[mask_out, np.newaxis]

        current_score = compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()

        step_size *= lr_decay
        if step_size < 1e-6:
            break

    # Phase 2: Fine-tuning with gradient-based approach (from inspiration)
    if time.time() < deadline - 15:
        points = best_points.copy()
        polish_step = 0.005
        polish_decay = 0.999
        polish_deadline = deadline - 5

        for step in range(5000):
            if time.time() > polish_deadline:
                break

            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)

            min_pair_dist = np.min(dists)
            norms = np.linalg.norm(points, axis=1)
            min_boundary_dist = np.min(1.0 - norms)

            current_score = min(min_pair_dist / 2.0, min_boundary_dist)

            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()

            # Use top 20 closest pairs for precision
            flat_dists = dists.flatten()
            k = min(20, N * (N - 1) // 2)
            part_idx = np.argpartition(flat_dists, k)[:k]

            pair_grad = np.zeros_like(points)
            for idx in part_idx:
                i = int(idx // N)
                j = int(idx % N)
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-6:
                    continue
                vec = (points[i] - points[j]) / d
                pair_grad[i] += vec
                pair_grad[j] -= vec

            # Boundary gradient for points close to boundary
            boundary_grad = np.zeros_like(points)
            close_to_boundary = norms > (1.0 - current_score - 0.002)
            for i in np.where(close_to_boundary)[0]:
                if norms[i] > 1e-6:
                    boundary_grad[i] = -points[i] / norms[i]

            # Combine based on which constraint is tighter
            pair_score = min_pair_dist / 2.0
            bound_score = min_boundary_dist

            if pair_score <= bound_score:
                w_pair = 1.0
                w_bound = 0.2
            else:
                w_pair = 0.2
                w_bound = 1.0

            grad = w_pair * pair_grad + w_bound * boundary_grad

            g_norm = np.linalg.norm(grad)
            if g_norm > 1e-6:
                grad = grad / g_norm

            points += polish_step * grad

            # Keep inside
            norms_new = np.linalg.norm(points, axis=1)
            outside = norms_new > 0.999
            if np.any(outside):
                points[outside] = points[outside] * (0.999 / norms_new[outside, np.newaxis])

            polish_step = max(1e-5, polish_step * polish_decay)

    # Phase 3: Symmetry-enforced ring model optimization (new strategy)
    if time.time() < deadline - 25:
        # Identify ring structure from current best points
        radii_sorted = np.sort(np.linalg.norm(best_points, axis=1))
        
        # Cluster radii into rings
        ring_members = []
        current_ring = [0]
        
        for i in range(1, len(radii_sorted)):
            if radii_sorted[i] - radii_sorted[i-1] > 0.02:
                ring_members.append(current_ring)
                current_ring = [i]
            else:
                current_ring.append(i)
        ring_members.append(current_ring)
        
        # Build ring parameters
        n_rings = len(ring_members)
        if n_rings > 1:
            ring_r = np.array([np.mean(radii_sorted[m]) for m in ring_members])
            ring_counts = np.array([len(m) for m in ring_members])
            
            # Initial angular offsets
            ring_offsets = np.zeros(n_rings)
            for k, members in enumerate(ring_members):
                if len(members) > 0:
                    angles = np.arctan2(best_points[members, 1], best_points[members, 0])
                    ring_offsets[k] = np.mean(angles)
            
            def ring_objective(params):
                # params: [ring_r (n_rings), ring_offsets (n_rings)]
                new_r = params[:n_rings]
                new_off = params[n_rings:]
                
                # Ensure radii are within bounds
                new_r = np.clip(new_r, 0.05, 0.98)
                
                new_pts = np.zeros((N, 2))
                idx = 0
                for k in range(n_rings):
                    n_k = ring_counts[k]
                    angles = new_off[k] + 2 * np.pi * np.arange(n_k) / n_k
                    for j in range(n_k):
                        new_pts[idx, 0] = new_r[k] * np.cos(angles[j])
                        new_pts[idx, 1] = new_r[k] * np.sin(angles[j])
                        idx += 1
                
                # Project inside
                r = np.linalg.norm(new_pts, axis=1)
                mask_out = r > 1.0
                if np.any(mask_out):
                    new_pts[mask_out] /= r[mask_out, np.newaxis]
                
                return -compute_score(new_pts)
            
            x0 = np.concatenate([ring_r, ring_offsets])
            bounds = [(0.05, 0.98)] * n_rings + [(-np.pi, np.pi)] * n_rings
            
            try:
                result_ring = minimize(ring_objective, x0, method='L-BFGS-B', bounds=bounds,
                                       options={'maxiter': 100, 'gtol': 1e-8})
                if result_ring.fun < ring_objective(x0):
                    new_r = np.clip(result_ring.x[:n_rings], 0.05, 0.98)
                    new_off = result_ring.x[n_rings:]
                    
                    new_pts = np.zeros((N, 2))
                    idx = 0
                    for k in range(n_rings):
                        n_k = ring_counts[k]
                        angles = new_off[k] + 2 * np.pi * np.arange(n_k) / n_k
                        for j in range(n_k):
                            new_pts[idx, 0] = new_r[k] * np.cos(angles[j])
                            new_pts[idx, 1] = new_r[k] * np.sin(angles[j])
                            idx += 1
                    
                    r = np.linalg.norm(new_pts, axis=1)
                    mask_out = r > 1.0
                    if np.any(mask_out):
                        new_pts[mask_out] /= r[mask_out, np.newaxis]
                    
                    new_score = compute_score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_pts
            except Exception:
                pass

    # Phase 4: Final local optimization with scipy (from current program)
    if time.time() < deadline - 15:
        n = len(best_points)
        diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        flat_dist = dist.flatten()
        k = min(30, n * (n - 1) // 2)
        smallest_indices = np.argpartition(flat_dist, k)[:k]
        pairs = []
        for idx in smallest_indices:
            i, j = np.unravel_index(idx, (n, n))
            if i < j:
                pairs.append((i, j))
        
        radii = np.linalg.norm(best_points, axis=1)
        boundary_violators = np.where(radii > 0.95)[0]
        
        def objective(x):
            pts = x.reshape(-1, 2)
            d = cdist(pts, pts)
            np.fill_diagonal(d, np.inf)
            beta = 100.0
            d_half = d / 2.0
            m = np.min(d_half)
            exp_terms = np.exp(-beta * (d_half - m))
            soft_min_pairs = m - np.log(np.sum(exp_terms)) / beta
            r = np.linalg.norm(pts, axis=1)
            bdist = 1.0 - r
            m_b = np.min(bdist)
            exp_terms_b = np.exp(-beta * (bdist - m_b))
            soft_min_boundary = m_b - np.log(np.sum(exp_terms_b)) / beta
            return smooth_min(soft_min_pairs, soft_min_boundary, beta)
        
        def objective_neg(x):
            return -objective(x)

        x0 = best_points.flatten()
        best_local_points = best_points.copy()
        best_local_score = compute_score(best_points)
        
        bounds = [(-1.0, 1.0)] * (2 * n)
        try:
            result1 = minimize(objective_neg, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10})
            if result1.success or result1.fun < objective_neg(x0):
                new_points = result1.x.reshape(-1, 2)
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
        
        if time.time() < deadline - 8:
            crit_points = set()
            for i, j in pairs:
                crit_points.add(i)
                crit_points.add(j)
            for idx in boundary_violators:
                crit_points.add(idx)
            
            if 0 < len(crit_points) < n:
                crit_indices = sorted(crit_points)
                
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

    # Final polish
    if time.time() < deadline - 5:
        n = len(best_points)
        diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        flat_dist = dist.flatten()
        k_small = min(10, n * (n - 1) // 2)
        smallest_indices = np.argpartition(flat_dist, k_small)[:k_small]
        critical_points = set()
        for idx in smallest_indices:
            i, j = np.unravel_index(idx, (n, n))
            critical_points.add(i)
            critical_points.add(j)
        radii = np.linalg.norm(best_points, axis=1)
        boundary_crit = np.where(radii > 0.93)[0]
        for idx in boundary_crit:
            critical_points.add(idx)
        
        if 0 < len(critical_points) < n:
            crit_indices = sorted(critical_points)
            
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
