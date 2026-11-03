"""Rewritten program: Lennard-Jones particle dynamics with multi-phase annealing and vectorized gradient."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    time_limit = 145.0  # 5s safety margin

    def _elapsed():
        return time.time() - start_time

    def _compute_radius(c):
        """Compute the minimum radius (score) for centers c of shape (N,3)."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        dist_walls = np.minimum(c + 0.5, 0.5 - c)
        min_wall = np.min(dist_walls)
        return min(min_pair, min_wall)

    def _vectorized_gradient(c):
        """
        Compute the gradient for maximizing the minimum radius.
        Returns (G, r_curr) where G is the gradient direction.
        """
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, np.inf)
        dist_walls = np.minimum(c + 0.5, 0.5 - c)
        r_curr = min(np.min(d) / 2.0, np.min(dist_walls))

        G = np.zeros_like(c)
        eps = 1e-8
        active = d < 2 * r_curr + eps

        # Vectorized pair forces
        # For active pairs, add unit vector from j to i for particle i, and opposite for j
        # diff[i,j] = c[i] - c[j]
        # We want to push i away from j: add diff[i,j]/d[i,j] to G[i]
        # And push j away from i: add -diff[i,j]/d[i,j] to G[j]
        
        # Create mask for upper triangle active pairs
        i_idx, j_idx = np.where(active)
        # Only take i < j to avoid double counting
        mask_upper = i_idx < j_idx
        i_idx = i_idx[mask_upper]
        j_idx = j_idx[mask_upper]
        
        for i, j in zip(i_idx, j_idx):
            vec = diff[i, j]  # c[i] - c[j]
            norm = d[i, j]
            if norm > 1e-12:
                G[i] += vec / norm
                G[j] -= vec / norm

        # Wall forces
        for i in range(N):
            for k in range(3):
                if dist_walls[i, k] < r_curr + eps:
                    if c[i, k] > 0:
                        G[i, k] -= 1.0
                    else:
                        G[i, k] += 1.0

        norm_G = np.linalg.norm(G)
        if norm_G > 1e-15:
            G /= norm_G
        return G, r_curr

    def _lj_forces(c, sigma, k_spring, anchors):
        """
        Compute forces on all particles using vectorized operations.
        """
        n = len(c)
        forces = np.zeros_like(c)

        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, np.inf)

        cutoff = 2.5 * sigma
        mask = (d < cutoff) & (d > 1e-12)

        s_over_d = sigma / d
        rep_mag = 10.0 * (s_over_d ** 12)
        rep_mag = np.where(mask, rep_mag, 0.0)

        force_vec = rep_mag[:, :, np.newaxis] * diff / d[:, :, np.newaxis]
        forces += np.sum(force_vec, axis=1)

        spring = -k_spring * (c - anchors)
        forces += spring

        for axis in range(3):
            dist_left = c[:, axis] + 0.5
            too_close_left = dist_left < 0.05
            forces[too_close_left, axis] += 50.0 * (0.05 - dist_left[too_close_left])
            dist_right = 0.5 - c[:, axis]
            too_close_right = dist_right < 0.05
            forces[too_close_right, axis] -= 50.0 * (0.05 - dist_right[too_close_right])

        return forces

    def _generate_bcc_anchors(n_points):
        """Generate BCC lattice anchor points centered in the cube."""
        m = 3
        pts = []
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    pts.append([i, j, k])
                    pts.append([i + 0.5, j + 0.5, k + 0.5])
        pts = np.array(pts, dtype=float)
        pts -= pts.mean(axis=0)
        max_ext = np.max(np.abs(pts))
        pts *= 0.4 / max_ext
        if len(pts) >= n_points:
            dists_from_origin = np.sum(pts * pts, axis=1)
            idx = np.argsort(dists_from_origin)[:n_points]
            pts = pts[idx]
        else:
            extra = np.random.uniform(-0.3, 0.3, (n_points - len(pts), 3))
            pts = np.vstack([pts, extra])
        return pts

    def _generate_fcc_anchors(n_points):
        """Generate FCC-like lattice anchor points."""
        m = 3
        pts = []
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    pts.append([i, j, k])
                    pts.append([i + 0.5, j + 0.5, k])
                    pts.append([i + 0.5, j, k + 0.5])
                    pts.append([i, j + 0.5, k + 0.5])
        pts = np.array(pts, dtype=float)
        pts -= pts.mean(axis=0)
        max_ext = np.max(np.abs(pts))
        pts *= 0.4 / max_ext
        if len(pts) >= n_points:
            dists_from_origin = np.sum(pts * pts, axis=1)
            idx = np.argsort(dists_from_origin)[:n_points]
            pts = pts[idx]
        else:
            extra = np.random.uniform(-0.3, 0.3, (n_points - len(pts), 3))
            pts = np.vstack([pts, extra])
        return pts

    def _generate_hcp_anchors(n_points):
        """Generate HCP-like lattice anchor points."""
        m = 3
        pts = []
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    # HCP: simple hexagonal arrangement
                    pts.append([i, j, k])
                    pts.append([i + 0.5, j + 0.5, k + 0.5])
                    pts.append([i + 0.5, j, k])
                    pts.append([i, j + 0.5, k])
        pts = np.array(pts, dtype=float)
        pts -= pts.mean(axis=0)
        max_ext = np.max(np.abs(pts))
        pts *= 0.4 / max_ext
        if len(pts) >= n_points:
            dists_from_origin = np.sum(pts * pts, axis=1)
            idx = np.argsort(dists_from_origin)[:n_points]
            pts = pts[idx]
        else:
            extra = np.random.uniform(-0.3, 0.3, (n_points - len(pts), 3))
            pts = np.vstack([pts, extra])
        return pts

    def _generate_grid_anchors(n_points):
        """Simple cubic grid anchors."""
        m = int(np.ceil(n_points ** (1.0/3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        if len(pts) >= n_points:
            dists = np.sum(pts * pts, axis=1)
            idx = np.argsort(dists)[:n_points]
            pts = pts[idx]
        else:
            extra = np.random.uniform(-0.3, 0.3, (n_points - len(pts), 3))
            pts = np.vstack([pts, extra])
        return pts

    def _generate_random_anchors(n_points):
        return np.random.uniform(-0.4, 0.4, (n_points, 3))

    def _refine_from_dynamics(c, anchors, start_frac, end_frac, n_dyn, n_grad):
        """Run dynamics then gradient refinement within a time window."""
        # Phase: Dynamics
        sigma = 0.2
        k_spring = 4.0
        dt = 0.001
        for it in range(n_dyn):
            if _elapsed() > time_limit * end_frac:
                break
            F = _lj_forces(c, sigma, k_spring, anchors)
            v = F * dt
            c += v
            c = np.clip(c, -0.49, 0.49)
            sigma *= 0.9999
            k_spring *= 1.0002

        # Phase: Gradient refinement
        step = 0.001
        for it in range(n_grad):
            if _elapsed() > time_limit * end_frac:
                break
            G, r_curr = _vectorized_gradient(c)
            c_new = c + step * G
            c_new = np.clip(c_new, -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            if r_new > r_curr + 1e-12:
                c = c_new
                step = min(step * 1.02, 0.003)
            else:
                step *= 0.85
                if step < 1e-10:
                    break
        return c, _compute_radius(c)

    # Generate diverse anchor sets
    anchor_sets = [
        _generate_bcc_anchors(N),
        _generate_fcc_anchors(N),
        _generate_hcp_anchors(N),
        _generate_grid_anchors(N),
        _generate_random_anchors(N),
        _generate_bcc_anchors(N) + np.random.normal(0, 0.05, (N, 3)),
    ]

    best_r = -1.0
    best_centers = None

    # Multi-phase annealing for each anchor set
    for anchor_idx, anchors in enumerate(anchor_sets):
        if _elapsed() > time_limit * 0.85:
            break

        c = anchors.copy()
        c = np.clip(c, -0.45, 0.45)

        # Phase 1: High temperature exploration
        T = 0.05
        sigma = 0.25
        k_spring = 2.0
        dt = 0.002
        n_iter_phase1 = 2000

        for it in range(n_iter_phase1):
            if _elapsed() > time_limit * 0.4:
                break
            F = _lj_forces(c, sigma, k_spring, anchors)
            v = F * dt
            c += v
            c = np.clip(c, -0.49, 0.49)
            T *= 0.999
            sigma *= 0.9995
            k_spring *= 1.0005

        # Phase 2: Medium temperature refinement
        sigma = 0.2
        k_spring = 5.0
        dt = 0.001
        n_iter_phase2 = 3000

        for it in range(n_iter_phase2):
            if _elapsed() > time_limit * 0.7:
                break
            F = _lj_forces(c, sigma, k_spring, anchors)
            v = F * dt
            c += v
            c = np.clip(c, -0.49, 0.49)
            sigma *= 0.9998
            k_spring *= 1.0003

        # Phase 3: Fine relaxation - gradient ascent on radius
        c, r_final = _refine_from_dynamics(c, anchors, 0.7, 0.9, 1000, 3000)
        if r_final > best_r:
            best_r = r_final
            best_centers = c.copy()

        # Try a perturbation restart from this solution
        if _elapsed() < time_limit * 0.85:
            c_pert = best_centers + np.random.normal(0, 0.015, (N, 3))
            c_pert = np.clip(c_pert, -0.45, 0.45)
            c_pert, r_pert = _refine_from_dynamics(c_pert, anchors, 0.80, 0.95, 1000, 2000)
            if r_pert > best_r:
                best_r = r_pert
                best_centers = c_pert.copy()

    # Final high-precision polish pass
    if best_centers is not None and _elapsed() < time_limit * 0.97:
        c = best_centers.copy()
        step = 0.0005
        for it in range(5000):
            if _elapsed() > time_limit * 0.99:
                break
            G, r_curr = _vectorized_gradient(c)
            c_new = c + step * G
            c_new = np.clip(c_new, -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            if r_new > r_curr + 1e-12:
                c = c_new
                step = min(step * 1.02, 0.002)
            else:
                step *= 0.85
                if step < 1e-10:
                    break
        r_final = _compute_radius(c)
        if r_final > best_r:
            best_r = r_final
            best_centers = c.copy()

    # Fallback
    if best_centers is None:
        m = int(np.ceil(N ** (1.0/3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centers = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]

    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    if not np.all(np.isfinite(best_centers)):
        m = int(np.ceil(N ** (1.0/3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centers = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]

    return best_centers
# EVOLVE-BLOCK-END
