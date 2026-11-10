"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

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
    step_size = spacing * 0.05
    for iteration in range(500):
        if time.time() > deadline:
            break

        # Compute forces
        n = len(points)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        dist[dist < 1e-10] = 1e-10
        
        # Mask diagonal to avoid self-interaction
        mask = ~np.eye(n, dtype=bool)
        inv_dist_sq = np.zeros_like(dist)
        inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
        
        # Repulsion forces
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)

        # Boundary constraint force
        radii = np.linalg.norm(points, axis=1)
        boundary_force = np.zeros_like(points)
        over = radii > 0.5
        if np.any(over):
            mag = (radii[over] - 0.5) * 2.0
            boundary_force[over] = -mag[:, np.newaxis] * points[over] / radii[over, np.newaxis]
            
        forces += boundary_force

        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_norms)
        if max_force > 1e-10:
            forces /= max_force

        # Update positions
        points += step_size * forces

        # Project back inside unit circle
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
        step_size *= 0.999

    return best_points
# EVOLVE-BLOCK-END
