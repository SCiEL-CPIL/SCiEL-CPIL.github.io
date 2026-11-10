"""Optimized 50-circle packing using grid-based neighbor lookup and relaxation."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _initial_hexagonal():
    """Generate an initial hexagonal lattice configuration clipped to unit disk."""
    # Start with a hexagonal packing
    # For 50 circles, we need a reasonable spacing
    # Optimal radius is ~0.1258, so spacing ~0.2516
    # Let's start with a slightly larger spacing and refine
    spacing = 0.26
    
    # Generate hexagonal lattice points
    points = []
    r_max = 1.0 - spacing / 2.0
    
    # Determine grid extent
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= r_max:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than N points, select the N closest to origin
    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists, kind='stable')[:N]
        points = points[idx]
    
    # If we have fewer than N points, add random points
    if len(points) < N:
        while len(points) < N:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 0.9)
            points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
    
    return points[:N]


def _grid_neighbors(points, cell_size):
    """
    Build a grid-based collision map for O(1) neighbor lookup.
    Returns a dict mapping (ix, iy) -> list of point indices.
    """
    grid = {}
    for i, (x, y) in enumerate(points):
        ix = int(np.floor(x / cell_size))
        iy = int(np.floor(y / cell_size))
        key = (ix, iy)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)
    return grid


def _get_nearby_indices(grid, x, y, cell_size):
    """Get indices of points in neighboring grid cells."""
    ix = int(np.floor(x / cell_size))
    iy = int(np.floor(y / cell_size))
    nearby = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            key = (ix + dx, iy + dy)
            if key in grid:
                nearby.extend(grid[key])
    return nearby


def _energy_and_grad(points, cell_size=0.3):
    """
    Compute the energy (sum of squared violations) and gradient.
    Energy = sum of (min_dist_violation^2 + boundary_violation^2)
    We want to maximize the minimum radius, which is equivalent to
    minimizing violations.
    """
    n = len(points)
    energy = 0.0
    grad = np.zeros_like(points)
    
    # Build grid for neighbor lookup
    grid = _grid_neighbors(points, cell_size)
    
    # For each point, check neighbors in nearby grid cells
    for i in range(n):
        x_i, y_i = points[i]
        r_i = np.hypot(x_i, y_i)
        
        # Boundary constraint: point must be inside unit circle with margin
        # We'll use a soft penalty for being too close to boundary
        # Actually, let's think about this differently.
        
        # Get nearby points
        nearby_idx = _get_nearby_indices(grid, x_i, y_i, cell_size)
        
        for j in nearby_idx:
            if j <= i:
                continue
            x_j, y_j = points[j]
            dx = x_i - x_j
            dy = y_i - y_j
            dist = np.hypot(dx, dy)
            
            if dist < 1e-10:
                # Coincident points - push apart randomly
                angle = np.random.uniform(0, 2 * np.pi)
                grad[i, 0] += np.cos(angle) * 10.0
                grad[i, 1] += np.sin(angle) * 10.0
                grad[j, 0] -= np.cos(angle) * 10.0
                grad[j, 1] -= np.sin(angle) * 10.0
                energy += 100.0
            else:
                # We want dist >= 2*r where r is the common radius
                # Instead of fixing r, we can think of this as:
                # maximize r such that all dist >= 2r and all |c_i| <= 1-r
                # 
                # Let's use a different formulation:
                # For each pair, the "slack" is dist - 2r
                # For each point, the "slack" is (1 - |c_i|) - r
                # We want to maximize r.
                #
                # Alternative: minimize sum of (2r - dist)^2 for dist < 2r
                # and ((1-|c_i|) - r)^2 for (1-|c_i|) < r
                #
                # But we don't know r a priori. Let's use a simpler approach:
                # Minimize the maximum violation.
                # 
                # Actually, let's just use a repulsive force model:
                # Force = k / dist^2 for dist < threshold
                threshold = 0.3
                if dist < threshold:
                    force = 1.0 / (dist * dist)
                    grad[i, 0] += dx / dist * force
                    grad[i, 1] += dy / dist * force
                    grad[j, 0] -= dx / dist * force
                    grad[j, 1] -= dy / dist * force
                    energy += (threshold - dist) ** 2
    
    # Boundary forces: push points inward if too close to boundary
    for i in range(n):
        x_i, y_i = points[i]
        r_i = np.hypot(x_i, y_i)
        if r_i > 0.95:  # Close to boundary
            # Push inward
            if r_i > 1e-10:
                force = (r_i - 0.95) * 5.0
                grad[i, 0] -= x_i / r_i * force
                grad[i, 1] -= y_i / r_i * force
                energy += (r_i - 0.95) ** 2
    
    return energy, grad


def _compute_radius(points):
    """Compute the valid packing radius."""
    n = len(points)
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(points[i, 0] - points[j, 0], points[i, 1] - points[j, 1])
            if d < min_dist:
                min_dist = d
    
    min_boundary = np.min(1.0 - np.hypot(points[:, 0], points[:, 1]))
    
    return min(min_dist / 2.0, min_boundary)


def _relax(points, max_iters=500, cell_size=0.3):
    """
    Relax the configuration using gradient descent with grid-based neighbor lookup.
    """
    points = points.copy()
    lr = 0.01
    
    for it in range(max_iters):
        energy, grad = _energy_and_grad(points, cell_size)
        
        # Gradient descent step
        points -= lr * grad
        
        # Clip to unit disk
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii > 0.999
        if np.any(mask):
            points[mask, 0] *= 0.999 / radii[mask]
            points[mask, 1] *= 0.999 / radii[mask]
        
        # Decay learning rate
        lr *= 0.995
    
    return points


def construct():
    """
    Construct the optimal packing of 50 equal circles in the unit disk.
    Returns array of shape (50, 2) with circle centers.
    """
    # Try multiple initial configurations
    best_points = None
    best_radius = -1.0
    
    np.random.seed(42)
    
    # Initial hexagonal configuration
    points = _initial_hexagonal()
    
    # Relax using gradient-based optimization
    points = _relax(points, max_iters=300, cell_size=0.3)
    
    radius = _compute_radius(points)
    best_points = points.copy()
    best_radius = radius
    
    # Try a few more random initializations
    for trial in range(3):
        # Random initialization with some structure
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.random.uniform(-0.1, 0.1, N)
        radii = np.random.uniform(0.2, 0.8, N)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        points = _relax(points, max_iters=200, cell_size=0.3)
        radius = _compute_radius(points)
        
        if radius > best_radius:
            best_radius = radius
            best_points = points.copy()
    
    # Final polish with smaller steps
    best_points = _relax(best_points, max_iters=200, cell_size=0.25)
    
    # Ensure all points are strictly inside the unit circle
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    max_r = np.max(radii)
    if max_r > 0.9999:
        best_points *= 0.9999 / max_r
    
    return best_points
# EVOLVE-BLOCK-END
