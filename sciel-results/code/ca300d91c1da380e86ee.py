"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(points):
    """Compute the packing score: min of pairwise distances/2 and (1 - |c_i|)."""
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to boundary
    norms = np.sqrt(np.sum(points * points, axis=-1))
    min_boundary = np.min(1.0 - norms)
    
    return min(min_pair, min_boundary)


def _penalized_score(points, beta):
    """Smooth surrogate: penalize violations of constraints."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Pairwise: want dists >= 2r, penalize if dists < 2r
    # Use soft-min: log-sum-exp of (2r - dist)^2 for dist < 2r
    # Simpler: sum of max(0, 2r - dist)^2
    pair_viol = np.maximum(0, 2.0 - dists) ** 2
    pair_penalty = np.sum(pair_viol)
    
    # Boundary: want norms <= 1-r, penalize if norms > 1-r
    norms = np.sqrt(np.sum(points * points, axis=-1))
    bound_viol = np.maximum(0, norms - 1.0) ** 2
    bound_penalty = np.sum(bound_viol)
    
    # Base score: approximate r as min(min_pair, min_boundary)
    min_pair = np.min(dists) / 2.0
    min_boundary = np.min(1.0 - norms)
    base = min(min_pair, min_boundary)
    
    return base - beta * (pair_penalty + bound_penalty)


def _hex_lattice_init(n=50):
    """Generate a hexagonal lattice initial configuration."""
    spacing = 0.24
    points = []
    for j in range(-3, 4):
        for i in range(-3, 4):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 0.85**2:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) > n:
        # Take the n closest to center
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii)[:n]
        points = points[idx]
    elif len(points) < n:
        # Add random points
        while len(points) < n:
            theta = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(0, 0.9)
            points = np.vstack([points, [[r*np.cos(theta), r*np.sin(theta)]]])
    
    return points[:n]


def _pso_optimize(init_points, n_particles=20, n_iter=200, deadline=None):
    """Particle Swarm Optimization for circle packing."""
    N = init_points.shape[0]
    dim = N * 2
    
    # Initialize particles
    particles = np.zeros((n_particles, dim))
    velocities = np.zeros((n_particles, dim))
    pbest = np.zeros((n_particles, dim))
    pbest_score = np.full(n_particles, -np.inf)
    
    global_best = init_points.flatten().copy()
    global_score = _score(init_points)
    
    for i in range(n_particles):
        if i == 0:
            particles[i] = init_points.flatten()
        else:
            # Perturb initial configuration
            particles[i] = init_points.flatten() + np.random.normal(0, 0.05, dim)
        
        s = _score(particles[i].reshape(N, 2))
        if s > pbest_score[i]:
            pbest_score[i] = s
            pbest[i] = particles[i]
        if s > global_score:
            global_score = s
            global_best = particles[i].copy()
    
    w = 0.5  # Inertia weight
    c1 = 1.5  # Cognitive parameter
    c2 = 1.5  # Social parameter
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        for i in range(n_particles):
            r1 = np.random.uniform(0, 1, dim)
            r2 = np.random.uniform(0, 1, dim)
            
            velocities[i] = w * velocities[i] + c1 * r1 * (pbest[i] - particles[i]) + c2 * r2 * (global_best - particles[i])
            
            # Clip velocity
            max_vel = 0.1
            velocities[i] = np.clip(velocities[i], -max_vel, max_vel)
            
            particles[i] += velocities[i]
            
            # Project into unit disk (soft constraint)
            pts = particles[i].reshape(N, 2)
            norms = np.hypot(pts[:, 0], pts[:, 1])
            mask = norms > 0.95
            if np.any(mask):
                scale = 0.95 / norms[mask]
                pts[mask] *= scale[:, np.newaxis]
            particles[i] = pts.flatten()
            
            s = _score(pts)
            if s > pbest_score[i]:
                pbest_score[i] = s
                pbest[i] = particles[i]
            if s > global_score:
                global_score = s
                global_best = particles[i].copy()
        
        # Anneal: reduce exploration over time
        w = max(0.3, w * 0.995)
    
    return global_best.reshape(N, 2), global_score


def _local_polish(points, n_iter=500, deadline=None):
    """Gradient-based local optimization using repulsive forces."""
    N = points.shape[0]
    points = points.copy()
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline - 2.0:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, 1.0)
        
        # Repulsive force: push apart if too close
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                if d < 0.3:
                    force_mag = 0.01 / max(d, 0.001)
                    direction = (points[i] - points[j]) / max(d, 0.001)
                    forces[i] += force_mag * direction
                    forces[j] -= force_mag * direction
        
        # Boundary attraction: pull inward if too close to edge
        norms = np.hypot(points[:, 0], points[:, 1])
        for i in range(N):
            if norms[i] > 0.9:
                force_mag = 0.01 * (norms[i] - 0.9)
                direction = -points[i] / max(norms[i], 0.001)
                forces[i] += force_mag * direction
        
        # Update positions
        points += forces * 0.5
        
        # Project into unit disk
        norms = np.hypot(points[:, 0], points[:, 1])
        mask = norms > 0.98
        if np.any(mask):
            scale = 0.98 / norms[mask]
            points[mask] *= scale[:, np.newaxis]
    
    return points


def construct():
    """Construct optimal packing of 50 circles in unit circle."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial configuration from hexagonal lattice
    init_points = _hex_lattice_init(N)
    best_points = init_points.copy()
    best_score = _score(best_points)
    
    # PSO optimization
    pso_points, pso_score = _pso_optimize(init_points, n_particles=15, n_iter=150, deadline=deadline)
    if pso_score > best_score:
        best_points = pso_points
        best_score = pso_score
    
    # Local polish
    polished_points = _local_polish(best_points, n_iter=300, deadline=deadline)
    polished_score = _score(polished_points)
    if polished_score > best_score:
        best_points = polished_points
        best_score = polished_score
    
    # Final check: ensure all points are strictly inside
    norms = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = norms >= 1.0
    if np.any(mask):
        scale = 0.999 / norms[mask]
        best_points[mask] *= scale[:, np.newaxis]
    
    return best_points
# EVOLVE-BLOCK-END
