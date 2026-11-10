"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

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

    def gen_hex_lattice(spacing):
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing * 0.6:
                    pts.append((x, y))
        return np.array(pts)

    # Initial spacing estimate
    # Area argument: N * pi * r^2 <= pi * 1^2, so r <= 1/sqrt(N)
    # Hex packing efficiency ~0.907, so r ~ sqrt(0.907/N)
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

    # Force-directed optimization
    best_score = compute_score(points)
    best_points = points.copy()

    # Iterative optimization (Vectorized)
    # We use a soft constraint approach with adaptive target radius.
    # This is inspired by the current program's "r_target" logic but applied to the hex lattice start.
    
    r_current = best_score
    r_target = r_current * 1.01
    lr = 0.01
    momentum = 0.9
    v = np.zeros_like(points)

    # Main Optimization Loop
    while time.time() < deadline - 5.0:
        # Compute forces based on violation of r_target
        n = len(points)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Mask diagonal
        np.fill_diagonal(dist, np.inf)
        
        # Pairwise repulsion if dist < 2 * r_target
        mask = dist < 2.0 * r_target
        dists_safe = np.where(dist > 1e-8, dist, 1.0)
        
        viol = np.where(mask, 2.0 * r_target - dist, 0.0)
        weights = viol / dists_safe
        weights[~mask] = 0.0
        np.fill_diagonal(weights, 0.0)
        
        forces = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary constraint
        radii = np.linalg.norm(points, axis=1)
        b_mask = radii > 1.0 - r_target
        b_viol = np.where(b_mask, radii - (1.0 - r_target), 0.0)
        b_forces = -b_viol[:, np.newaxis] * (points / np.maximum(radii, 1e-8)[:, np.newaxis])
        
        total_forces = forces + b_forces
        
        # Update
        v = momentum * v + lr * total_forces
        points += v
        
        # Project back inside
        new_radii = np.linalg.norm(points, axis=1)
        max_r = 1.0 - r_target
        outside = new_radii > max_r
        if np.any(outside):
            points[outside] *= (max_r / new_radii[outside])[:, np.newaxis]
            v[outside] = 0.0

        # Evaluate
        current_score = compute_score(points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            r_current = current_score
            r_target = r_current * 1.005
            lr = max(0.001, lr * 0.98)
        else:
            # If we stuck, relax the target or reset if too stuck
            r_target = r_current * 0.995
            if r_target < best_score * 0.95:
                points = best_points.copy()
                v = np.zeros_like(points)
                r_target = best_score * 1.001
                lr = 0.005
            else:
                pass

    # Final Polish
    if time.time() < deadline - 1.0:
        r_target = best_score * 1.0001
        lr = 0.0005
        momentum = 0.9
        v = np.zeros_like(points)
        
        for _ in range(50):
            if time.time() > deadline:
                break
            n = len(points)
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            
            mask = dist < 2.0 * r_target
            dists_safe = np.where(dist > 1e-8, dist, 1.0)
            viol = np.where(mask, 2.0 * r_target - dist, 0.0)
            weights = viol / dists_safe
            weights[~mask] = 0.0
            np.fill_diagonal(weights, 0.0)
            
            forces = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
            
            radii = np.linalg.norm(points, axis=1)
            b_mask = radii > 1.0 - r_target
            b_viol = np.where(b_mask, radii - (1.0 - r_target), 0.0)
            b_forces = -b_viol[:, np.newaxis] * (points / np.maximum(radii, 1e-8)[:, np.newaxis])
            
            total_forces = forces + b_forces
            v = momentum * v + lr * total_forces
            points += v
            
            new_radii = np.linalg.norm(points, axis=1)
            max_r = 1.0 - r_target
            outside = new_radii > max_r
            if np.any(outside):
                points[outside] *= (max_r / new_radii[outside])[:, np.newaxis]
                v[outside] = 0.0
                
        final_score = compute_score(points)
        if final_score > best_score:
            best_score = final_score
            best_points = points.copy()

    return best_points
# EVOLVE-BLOCK-END
