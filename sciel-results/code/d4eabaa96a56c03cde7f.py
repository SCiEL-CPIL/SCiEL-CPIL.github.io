"""Program using ring-based global search followed by full-coordinate local optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin

    def compute_score(points):
        n = len(points)
        if n < 2:
            return 0.0
        # Compute all pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        # Get minimum pairwise distance (excluding diagonal)
        mask = ~np.eye(n, dtype=bool)
        min_pair_dist = np.min(dist[mask])
        # Compute minimum distance to boundary
        radii = np.linalg.norm(points, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        return min(min_pair_dist / 2.0, min_boundary_dist)

    def generate_ring_config(params):
        # params: [r0, r1, ..., r_{K-1}, n0, n1, ..., n_{K-1}]
        K = len(params) // 2
        radii = params[:K]
        counts = params[K:].astype(int)
        
        points = []
        for k in range(K):
            r = radii[k]
            n_k = counts[k]
            if n_k == 0:
                continue
            if n_k == 1:
                points.append([0.0, 0.0])
            else:
                angles = np.arange(n_k) * (2 * np.pi / n_k)
                xs = r * np.cos(angles)
                ys = r * np.sin(angles)
                for x, y in zip(xs, ys):
                    points.append([x, y])
        
        if not points:
            return np.zeros((N, 2))
            
        pts = np.array(points)
        return pts

    def ring_objective(params):
        pts = generate_ring_config(params)
        if len(pts) < 2:
            return -1.0
        
        # Compute score
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(len(pts), dtype=bool)
        min_pair_dist = np.min(dist[mask])
        radii = np.linalg.norm(pts, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        
        score = min(min_pair_dist / 2.0, min_boundary_dist)
        
        # Penalty if number of points is not close to N
        # We want exactly N points.
        # If we have more, it's fine (we'll trim). If less, it's bad.
        # But the ring config generates exactly sum(counts) points.
        # We constrain sum(counts) == N in the optimization?
        # Actually, easier to just fix counts and optimize radii?
        # Or optimize both?
        
        return -score

    # --- Phase 1: Ring-based Global Search ---
    
    # Estimate number of rings
    # Area argument: N * pi * r^2 <= pi * 1^2 => r ~ 1/sqrt(N) ~ 0.058
    # Circumference of ring k ~ 2 * pi * r_k
    # Spacing ~ 2*r
    # n_k ~ 2 * pi * r_k / (2*r) ~ pi * r_k / r
    # Sum n_k = N
    # Integrate: N ~ integral (pi * r / r) dr = pi * R^2 / (2r) ? No.
    # Let's just try K from 10 to 30.
    
    best_ring_score = 0.0
    best_ring_params = None
    
    for K in range(12, 26):
        if time.time() > deadline - 20:
            break
            
        # Initialize counts: roughly proportional to radius
        # r_k ~ sqrt(k)
        # n_k ~ k
        # Sum k = K(K+1)/2
        # Scale to N
        k_vals = np.arange(1, K + 1)
        counts = (k_vals / np.sum(k_vals)) * N
        counts = np.round(counts).astype(int)
        # Adjust to make sum N
        diff = N - np.sum(counts)
        if diff > 0:
            idx = np.argsort(counts)[::-1]
            for i in range(diff):
                counts[idx[i % K]] += 1
        elif diff < 0:
            idx = np.argsort(counts)
            for i in range(-diff):
                if counts[idx[i % K]] > 0:
                    counts[idx[i % K]] -= 1
        
        # Initialize radii
        # r_0 = 0 (if count > 0)
        # r_k ~ sqrt(sum of previous counts) / sqrt(N) ?
        # Let's use a simple heuristic: r_k = sqrt(k / K)
        radii = np.sqrt(np.arange(K) / K)
        if counts[0] > 0:
            radii[0] = 0.0
            
        # Combine params
        params = np.concatenate([radii, counts.astype(float)])
        
        # Define bounds
        # Radii: 0 to 1
        # Counts: 0 to N (integer, but scipy handles continuous, we'll round in generate)
        # Actually, optimizing counts continuously is tricky.
        # Let's fix counts and optimize radii.
        
        def obj_radii(r_params):
            pts = []
            for k in range(K):
                r = r_params[k]
                n_k = counts[k]
                if n_k == 0:
                    continue
                if n_k == 1:
                    pts.append([0.0, 0.0])
                else:
                    angles = np.arange(n_k) * (2 * np.pi / n_k)
                    xs = r * np.cos(angles)
                    ys = r * np.sin(angles)
                    for x, y in zip(xs, ys):
                        pts.append([x, y])
            
            if not pts:
                return 1.0
            pts = np.array(pts)
            
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            mask = ~np.eye(len(pts), dtype=bool)
            min_pair_dist = np.min(dist[mask])
            radii_pts = np.linalg.norm(pts, axis=1)
            min_boundary_dist = np.min(1.0 - radii_pts)
            
            score = min(min_pair_dist / 2.0, min_boundary_dist)
            return -score

        bounds = [(0.0, 1.0) for _ in range(K)]
        
        # Optimize radii
        try:
            res = minimize(obj_radii, radii, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.success or res.fun < obj_radii(radii):
                final_radii = res.x
                final_score = -res.fun
                if final_score > best_ring_score:
                    best_ring_score = final_score
                    # Reconstruct points for this best config
                    pts = []
                    for k in range(K):
                        r = final_radii[k]
                        n_k = counts[k]
                        if n_k == 0:
                            continue
                        if n_k == 1:
                            pts.append([0.0, 0.0])
                        else:
                            angles = np.arange(n_k) * (2 * np.pi / n_k)
                            xs = r * np.cos(angles)
                            ys = r * np.sin(angles)
                            for x, y in zip(xs, ys):
                                pts.append([x, y])
                    if len(pts) >= N:
                        # Select N closest to center? Or just first N?
                        # First N is fine if counts are fixed.
                        points_init = np.array(pts[:N])
                        if best_ring_params is None or len(points_init) == N:
                            best_ring_params = points_init
        except Exception:
            pass

    if best_ring_params is None:
        # Fallback to hex lattice
        spacing = 2.0 * np.sqrt(0.907 / N)
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing * 0.6:
                    pts.append((x, y))
        pts = np.array(pts)
        radii = np.hypot(pts[:, 0], pts[:, 1])
        points = pts[np.argsort(radii, kind="stable")[:N]].copy()
    else:
        points = best_ring_params.copy()

    # --- Phase 2: Full-coordinate Local Optimization ---
    
    # Gradient ascent on the score
    # Score = min( min_pair_dist/2, min_boundary_dist )
    # We want to maximize this.
    # This is non-smooth. We can use a smooth approximation or just hill climbing.
    # Let's use a simple iterative repulsion force approach again, but starting from a better config.
    
    best_score = compute_score(points)
    best_points = points.copy()
    
    step_size = 0.001
    decay = 0.99
    
    for iteration in range(2000):
        if time.time() > deadline:
            break
            
        n = len(points)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        dist[dist < 1e-10] = 1e-10
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        
        # Force: repulsion from neighbors, attraction to center (weak)
        # We want to increase the minimum distance.
        # Gradient of min distance is complex.
        # Let's use a potential energy approach: E = sum(1/d^2) + lambda * sum(r^2)
        # Minimize E.
        
        inv_dist_sq = np.zeros_like(dist)
        inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
        
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Boundary force: push inward if close to boundary
        radii = np.linalg.norm(points, axis=1)
        boundary_force = np.zeros_like(points)
        # If r > 0.9, push in
        over = radii > 0.9
        if np.any(over):
            mag = (radii[over] - 0.9) * 5.0
            boundary_force[over] = -mag[:, np.newaxis] * points[over] / radii[over, np.newaxis]
            
        forces += boundary_force
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_norms)
        if max_force > 1e-10:
            forces /= max_force
            
        # Update
        points += step_size * forces
        
        # Project back inside
        radii = np.linalg.norm(points, axis=1)
        mask_out = radii > 1.0
        if np.any(mask_out):
            points[mask_out] /= radii[mask_out, np.newaxis]
            
        current_score = compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        step_size *= decay
        
        # Early stop if no improvement for a long time
        if iteration > 100 and best_score == compute_score(best_points):
             # Check if stuck
             if np.linalg.norm(points - best_points) < 1e-6:
                 break

    return best_points
# EVOLVE-BLOCK-END
