"""Rewritten program: Lennard-Jones style dynamics + gradient polish for 300 equal circles in unit disk."""
import numpy as np
import time

N = 300


def _compute_radius(points):
    """Compute the maximum common radius for the given centres."""
    # Distance from each centre to the boundary
    dists = 1.0 - np.hypot(points[:, 0], points[:, 1])
    # Pairwise distances
    # For 300 points, 300*299/2 = 44850 pairs
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)
    return min(min_pair_dist / 2.0, np.min(dists))


def _hex_init(n):
    """Generate a hexagonal lattice initial configuration, clipped to the unit disk."""
    # Start with a reasonably dense hex lattice
    # For 300 circles in unit disk, rough spacing ~ 2/sqrt(300) ~ 0.115
    spacing = 0.12
    while True:
        k = int(1.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n:
            # Sort by radius to prefer central points
            idx = np.argsort(radius[mask], kind="stable")[:n]
            return points[idx]
        spacing *= 0.97


def _lj_step(points, dt, n):
    """One step of Lennard-Jones-like dynamics with wall repulsion."""
    # Compute forces
    forces = np.zeros_like(points)
    
    # Pairwise repulsion: use a soft repulsive potential
    # F_ij = k * (1/d^2 - 1/d0^2) * (r_i - r_j)/d for d < d0
    # Simpler: just repel when too close
    # Use a smooth potential: U = 1/d^2 for d < some threshold
    # Force = 2/d^3 * r_hat
    
    # For efficiency, compute all pairs
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(d, np.inf)
    
    # Only consider pairs that are close
    # Use a cutoff
    cutoff = 0.3
    mask = d < cutoff
    # Avoid division by zero
    d_safe = np.where(mask, d, 1.0)
    # Force magnitude: proportional to 1/d^3
    f_mag = np.zeros_like(d)
    f_mag[mask] = 2.0 / (d_safe[mask] ** 3)
    # Force direction: diff / d
    f_dir = diff / d_safe[:, :, np.newaxis]
    # Force on i from j: f_mag * f_dir
    # Sum over j
    forces += np.sum(f_mag[:, :, np.newaxis] * f_dir, axis=1)
    
    # Wall repulsion: push points back if they're too close to boundary
    dist_to_wall = 1.0 - np.hypot(points[:, 0], points[:, 1])
    wall_threshold = 0.1
    wall_mask = dist_to_wall < wall_threshold
    if np.any(wall_mask):
        # Force proportional to how close to wall
        wall_force_mag = np.zeros(n)
        wall_force_mag[wall_mask] = 10.0 * (1.0 - dist_to_wall[wall_mask] / wall_threshold) ** 2
        # Direction: away from center (radially outward)
        r_hat = points / np.hypot(points[:, 0], points[:, 1])[:, np.newaxis]
        forces[wall_mask] += wall_force_mag[wall_mask, np.newaxis] * r_hat[wall_mask]
    
    return forces


def _dynamics_optimize(points, max_time=60.0):
    """Optimize using Lennard-Jones style dynamics with damping."""
    start_time = time.time()
    n = len(points)
    velocities = np.zeros_like(points)
    
    # Adaptive time step
    dt = 0.001
    damping = 0.95
    
    best_radius = _compute_radius(points)
    best_points = points.copy()
    
    for step in range(50000):
        if time.time() - start_time > max_time:
            break
        
        forces = _lj_step(points, dt, n)
        
        # Update velocities with damping
        velocities = velocities * damping + forces * dt
        
        # Update positions
        points = points + velocities * dt
        
        # Keep points inside the disk (hard constraint)
        r = np.hypot(points[:, 0], points[:, 1])
        mask = r > 1.0
        if np.any(mask):
            points[mask] = points[mask] / r[mask, np.newaxis]
            velocities[mask] *= 0.0  # Kill velocity at wall
        
        # Check radius periodically
        if step % 100 == 0:
            current_radius = _compute_radius(points)
            if current_radius > best_radius:
                best_radius = current_radius
                best_points = points.copy()
        
        # Adaptive dt: reduce if forces are large
        max_force = np.max(np.hypot(forces[:, 0], forces[:, 1]))
        if max_force > 10.0:
            dt *= 0.5
        elif max_force < 0.1:
            dt = min(dt * 1.1, 0.005)
    
    return best_points, best_radius


def _gradient_polish(points, max_time=30.0):
    """Polish the configuration using gradient-based optimization."""
    start_time = time.time()
    n = len(points)
    
    def objective(p):
        """Negative radius (to minimize)."""
        dists = 1.0 - np.hypot(p[:, 0], p[:, 1])
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        return -min(min_pair_dist / 2.0, np.min(dists))
    
    # Use a simple gradient descent with numerical gradients
    # Since N=300, 600 variables, numerical gradients are expensive
    # Use a smarter approach: only optimize the closest pairs
    
    best_radius = _compute_radius(points)
    best_points = points.copy()
    
    lr = 0.001
    for step in range(20000):
        if time.time() - start_time > max_time:
            break
        
        # Compute current radius
        dists = 1.0 - np.hypot(points[:, 0], points[:, 1])
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        current_radius = min(min_pair_dist / 2.0, np.min(dists))
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Find the closest pair and the point closest to wall
        min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
        min_wall_idx = np.argmin(dists)
        
        # Gradient: push closest pair apart, push wall point inward
        grad = np.zeros_like(points)
        
        # For closest pair
        i, j = min_pair_idx
        if i != j:
            direction = points[i] - points[j]
            d_ij = np.linalg.norm(direction)
            if d_ij > 1e-10:
                direction = direction / d_ij
                grad[i] += direction
                grad[j] -= direction
        
        # For wall point
        r = np.linalg.norm(points[min_wall_idx])
        if r > 1e-10:
            direction = points[min_wall_idx] / r
            grad[min_wall_idx] -= direction  # Push inward
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        # Update
        points = points + lr * grad
        
        # Keep inside disk
        r = np.hypot(points[:, 0], points[:, 1])
        mask = r > 1.0
        if np.any(mask):
            points[mask] = points[mask] / r[mask, np.newaxis]
        
        # Decrease learning rate
        lr *= 0.999
    
    return best_points, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    # Phase 1: Initialize with hex lattice
    points = _hex_init(N)
    
    # Phase 2: Dynamics optimization
    if time.time() < deadline - 40.0:
        points, r1 = _dynamics_optimize(points, max_time=40.0)
    
    # Phase 3: Gradient polish
    if time.time() < deadline - 25.0:
        points, r2 = _gradient_polish(points, max_time=25.0)
    
    # Final check: ensure validity
    # Clip any points outside the disk
    r = np.hypot(points[:, 0], points[:, 1])
    mask = r > 1.0
    if np.any(mask):
        points[mask] = points[mask] / r[mask, np.newaxis]
    
    # Ensure exactly N points
    if len(points) != N:
        if len(points) < N:
            # Add more points
            extra = _hex_init(N - len(points))
            points = np.vstack([points, extra[:N - len(points)]])
        else:
            points = points[:N]
    
    return points
