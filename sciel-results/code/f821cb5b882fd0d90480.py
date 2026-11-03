import numpy as np
import time

N = 300
_DEADLINE = 110.0  # safety margin before 120s kill


def _compute_radius(points):
    """Compute the largest common radius for given points."""
    n = len(points)
    if n == 0:
        return 0.0
    # Boundary constraint
    dist_center = np.linalg.norm(points, axis=1)
    r_boundary = np.min(1.0 - dist_center)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    return min(r_boundary, r_pair)


def _hex_lattice_init(seed=42):
    """Generate a hexagonal lattice initialization inside the unit disk."""
    rng = np.random.RandomState(seed)
    spacing = 0.058
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        points = np.array(rows)
        if len(points) >= N:
            # Select N closest to center
            radii = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(radii, kind='stable')[:N]
            return points[idx]
        spacing *= 0.95


def _gradient_descent(points, n_iters=2000, lr=0.0001, beta=0.9, deadline=110.0):
    """Simple gradient descent to maximize the minimum radius."""
    points = points.copy()
    n = len(points)
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        # Compute distances to all other points
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Find the minimum distance pair for each point
        min_dists = np.min(dists, axis=1)
        
        # Boundary distance
        boundary_dist = 1.0 - np.linalg.norm(points, axis=1)
        
        # The limiting constraint for each point is either its nearest neighbor or the boundary
        # We want to move points away from their closest constraint
        
        # For each point, determine which constraint is active
        # If boundary is closer than half the min pair distance, use boundary
        # Otherwise use pair distance
        
        # Compute gradient: move away from nearest neighbor or toward center if boundary is limiting
        grad = np.zeros_like(points)
        
        for i in range(n):
            # Find nearest neighbor
            min_dist = min_dists[i]
            min_j = np.argmin(dists[i])
            
            # Direction away from nearest neighbor
            dir_nn = points[i] - points[min_j]
            dir_nn_norm = np.linalg.norm(dir_nn)
            if dir_nn_norm > 1e-10:
                dir_nn /= dir_nn_norm
            
            # Boundary constraint
            bdist = boundary_dist[i]
            
            if bdist < min_dist / 2.0:
                # Boundary is limiting: move toward center
                grad[i] = -points[i] / max(np.linalg.norm(points[i]), 1e-10)
            else:
                # Pair is limiting: move away from nearest neighbor
                grad[i] = dir_nn
        
        # Update
        points += lr * grad
        
        # Clip to stay inside
        norms = np.linalg.norm(points, axis=1)
        for i in range(n):
            if norms[i] > 0.999:
                points[i] *= 0.999 / norms[i]
    
    return points


def _simulated_annealing(points, n_iters=5000, t_start=0.02, t_end=1e-5, deadline=110.0):
    """Simulated annealing to escape local minima."""
    points = points.copy()
    n = len(points)
    current_r = _compute_radius(points)
    best_points = points.copy()
    best_r = current_r
    
    n_steps = max(n_iters, 1)
    
    for it in range(n_steps):
        if time.time() > deadline:
            break
        
        # Temperature schedule
        progress = it / n_steps
        t = t_start * (t_end / t_start) ** progress
        
        # Pick a random point to move
        i = np.random.randint(n)
        
        # Compute current local constraints
        diff = points[i] - points
        dists = np.linalg.norm(diff, axis=1)
        dists[i] = np.inf
        min_dist = np.min(dists)
        min_j = np.argmin(dists)
        bdist = 1.0 - np.linalg.norm(points[i])
        
        # Determine the active constraint and direction
        if bdist < min_dist / 2.0:
            # Boundary limiting: random move toward center
            direction = -points[i] / max(np.linalg.norm(points[i]), 1e-10)
        else:
            # Pair limiting: random move away from nearest neighbor
            direction = points[i] - points[min_j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
            else:
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
        
        # Add some randomness
        angle = np.random.uniform(-0.5, 0.5)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        direction = R @ direction
        
        # Move
        step = t
        new_point = points[i] + step * direction
        
        # Clip to boundary
        new_norm = np.linalg.norm(new_point)
        if new_norm > 0.999:
            new_point *= 0.999 / new_norm
        
        # Try the move
        old_point = points[i].copy()
        points[i] = new_point
        new_r = _compute_radius(points)
        
        # Accept or reject
        if new_r > current_r or np.random.rand() < np.exp((new_r - current_r) / max(t, 1e-10)):
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_points = points.copy()
        else:
            points[i] = old_point
    
    return best_points


def construct():
    start_time = time.time()
    
    # Try multiple seeds and optimizations
    best_points = None
    best_r = 0.0
    
    seeds = [42, 123, 456, 789, 1000]
    
    for seed in seeds:
        if time.time() - start_time > _DEADLINE:
            break
        
        # Initialize with hexagonal lattice
        points = _hex_lattice_init(seed)
        
        # Gradient descent phase
        points = _gradient_descent(points, n_iters=1000, lr=0.0001, deadline=_DEADLINE - 5)
        r = _compute_radius(points)
        
        # Simulated annealing phase
        if time.time() - start_time < _DEADLINE - 10:
            points = _simulated_annealing(points, n_iters=3000, t_start=0.015, t_end=1e-6, deadline=_DEADLINE - 5)
            r = _compute_radius(points)
        
        # Final gradient polish
        if time.time() - start_time < _DEADLINE - 2:
            points = _gradient_descent(points, n_iters=500, lr=0.00005, deadline=_DEADLINE - 1)
            r = _compute_radius(points)
        
        if r > best_r:
            best_r = r
            best_points = points.copy()
    
    if best_points is None:
        # Fallback
        best_points = _hex_lattice_init(42)
    
    return best_points
