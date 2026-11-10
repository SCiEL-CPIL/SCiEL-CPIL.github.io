"""Program for packing 50 equal circles in a unit circle.
Uses a grid-based neighbor lookup with physics-inspired relaxation."""
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree

def _energy_and_grad(pos, container_radius=1.0):
    """Compute repulsive energy and gradient for circle packing.
    Uses a soft repulsive potential to avoid hard constraints.
    """
    n = len(pos)
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    # Avoid division by zero
    dist = np.maximum(dist, 1e-10)
    
    # Container constraint: distance from origin
    r_origin = np.sqrt(np.sum(pos**2, axis=1))
    
    # We want to maximize the minimum of (min_pair_dist/2, min_container_margin)
    # Use a repulsive potential that pushes apart when too close
    # Target: all pairs should be at least 2*r apart, and each circle should be at most (1-r) from center
    
    # Use a smooth potential: for each pair, if dist < d_target, push apart
    # For container, if r_origin > 1 - r, push inward
    
    # We'll use a two-stage approach: first get a good configuration via relaxation,
    # then optimize r directly
    
    # Repulsive force between pairs (soft spring-like)
    # Energy = sum over pairs of max(0, d_min - dist)^2 + sum over i of max(0, r_origin_i - (1-r))^2
    # But we don't know r yet. Let's use a different approach.
    
    # Actually, let's just do a direct optimization where we place points and compute the achievable radius.
    # For the relaxation, we use a repulsive potential with a fixed "desired" separation.
    
    # Let's use: for each pair, if dist < desired_sep, add repulsive energy
    # desired_sep can be tuned; we'll use it as a parameter
    
    return None  # Placeholder

def _relaxation(pos, n_iters=5000, learning_rate=0.001, desired_sep=0.25):
    """Physics-inspired relaxation using grid-based neighbor lookup."""
    n = len(pos)
    pos = pos.copy().astype(np.float64)
    
    # Grid cell size for neighbor lookup
    cell_size = desired_sep * 1.5
    grid_res = int(np.ceil(2.0 / cell_size)) + 2
    grid_size = grid_res * 2 + 1
    
    def build_grid():
        """Build a grid mapping cell indices to point indices."""
        grid = {}
        for i in range(n):
            # Map position to grid cell
            cx = int((pos[i, 0] + 1.0) / cell_size)
            cy = int((pos[i, 1] + 1.0) / cell_size)
            cx = max(0, min(cx, grid_res - 1))
            cy = max(0, min(cy, grid_res - 1))
            key = (cx, cy)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)
        return grid
    
    def get_neighbors(i, grid):
        """Get neighbor indices within one cell distance."""
        cx = int((pos[i, 0] + 1.0) / cell_size)
        cy = int((pos[i, 1] + 1.0) / cell_size)
        cx = max(0, min(cx, grid_res - 1))
        cy = max(0, min(cy, grid_res - 1))
        neighbors = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                key = (cx + dx, cy + dy)
                if key in grid:
                    neighbors.extend(grid[key])
        # Remove self
        if i in neighbors:
            neighbors.remove(i)
        return neighbors
    
    for it in range(n_iters):
        grid = build_grid()
        forces = np.zeros_like(pos)
        
        for i in range(n):
            neighbors = get_neighbors(i, grid)
            
            # Pairwise repulsive forces
            for j in neighbors:
                if j <= i:
                    continue
                diff = pos[i] - pos[j]
                dist = np.sqrt(np.sum(diff**2))
                if dist < desired_sep and dist > 1e-10:
                    # Repulsive force proportional to (desired_sep - dist)
                    force_mag = (desired_sep - dist)
                    force = force_mag * diff / dist
                    forces[i] += force
                    forces[j] -= force
            
            # Container constraint: push inward if too close to boundary
            r_origin = np.sqrt(np.sum(pos[i]**2))
            if r_origin > 1.0 - desired_sep / 2:
                # Push toward center
                force_mag = (r_origin - (1.0 - desired_sep / 2))
                force = -force_mag * pos[i] / max(r_origin, 1e-10)
                forces[i] += force
        
        # Update positions
        pos += learning_rate * forces
        
        # Project back inside container
        r_origins = np.sqrt(np.sum(pos**2, axis=1))
        for i in range(n):
            if r_origins[i] > 1.0:
                pos[i] *= (1.0 / r_origins[i])
        
        # Decay learning rate
        learning_rate *= 0.999
    
    return pos

def _compute_radius(pos):
    """Compute the maximum valid radius for the given configuration."""
    n = len(pos)
    if n == 0:
        return 0.0
    
    # Container margin
    r_origins = np.sqrt(np.sum(pos**2, axis=1))
    container_margin = np.min(1.0 - r_origins)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    return min(container_margin, min_pair_dist / 2.0)

def construct():
    """Construct 50 circle centers in the unit circle to maximize common radius."""
    np.random.seed(42)
    
    # Initial configuration: random points in the unit disk
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Also try a hexagonal lattice initialization as a second candidate
    spacing = 0.25
    hex_points = []
    k = 6
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.sqrt(x**2 + y**2) <= 1.0 - spacing / 2:
                hex_points.append([x, y])
    hex_points = np.array(hex_points)
    if len(hex_points) >= N:
        # Sort by distance from center and take N
        r_hex = np.sqrt(np.sum(hex_points**2, axis=1))
        idx = np.argsort(r_hex, kind='stable')[:N]
        hex_points = hex_points[idx]
    
    # Run relaxation on both initializations
    best_pos = None
    best_radius = -1
    
    for init_pos in [pos, hex_points]:
        # Try multiple desired_sep values
        for desired_sep in [0.24, 0.25, 0.26]:
            relaxed = _relaxation(init_pos.copy(), n_iters=3000, learning_rate=0.0005, desired_sep=desired_sep)
            r = _compute_radius(relaxed)
            if r > best_radius:
                best_radius = r
                best_pos = relaxed.copy()
    
    # Final refinement: local optimization using scipy
    if best_pos is not None:
        # Use a simple local search to fine-tune
        def objective(x):
            p = x.reshape(-1, 2)
            # Penalize going outside container
            r_orig = np.sqrt(np.sum(p**2, axis=1))
            container_pen = np.sum(np.maximum(0, r_orig - 1.0)**2)
            # Penalize overlapping
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
            # We want to maximize min radius, so minimize negative of it
            # But for optimization, use a smooth surrogate
            # Minimize sum of max(0, 2*r_target - dist)^2 + sum of max(0, r_orig - (1-r_target))^2
            # This is complex; let's just do a direct local search
            return -_compute_radius(p) + 10.0 * container_pen
        
        # Local search
        x0 = best_pos.flatten()
        result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8})
        if result.fun < objective(x0):
            new_pos = result.x.reshape(-1, 2)
            r_new = _compute_radius(new_pos)
            if r_new > best_radius:
                best_pos = new_pos
                best_radius = r_new
    
    # Ensure all points are strictly inside the container
    r_origins = np.sqrt(np.sum(best_pos**2, axis=1))
    for i in range(N):
        if r_origins[i] >= 1.0:
            best_pos[i] *= 0.999
    
    return best_pos
# EVOLVE-BLOCK-END
