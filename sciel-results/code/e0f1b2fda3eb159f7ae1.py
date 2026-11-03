"""Force-directed packing: continuous Lennard-Jones + wall repulsion with spatial hash."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START

def _spatial_hash(centers, cell_size):
    """Build a spatial hash grid for O(N) neighbor lookups.
    Returns dict: (ix, iy) -> list of indices
    """
    grid = {}
    inv = 1.0 / cell_size
    for i, c in enumerate(centers):
        ix = int((c[0] + 0.5) * inv)
        iy = int((c[1] + 0.5) * inv)
        key = (ix, iy)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)
    return grid

def _neighbor_indices(centers, i, grid, cell_size, search_radius):
    """Get indices of points within search_radius of point i."""
    inv = 1.0 / cell_size
    cx, cy = centers[i]
    ix = int((cx + 0.5) * inv)
    iy = int((cy + 0.5) * inv)
    n_cells = int(math.ceil(search_radius / cell_size)) + 1
    result = []
    for dx in range(-n_cells, n_cells + 1):
        for dy in range(-n_cells, n_cells + 1):
            key = (ix + dx, iy + dy)
            if key in grid:
                for j in grid[key]:
                    if j != i:
                        result.append(j)
    return result

def _force_directed(centers, max_iters=5000, deadline=None):
    """
    Minimize a potential energy:
      E = sum_{i<j, |rij|<rcut} [4*eps*((sig/rij)^12 - (sig/rij)^6)] + wall_repulsion
    + wall repulsion: for each point, repel from walls if within margin.
    
    Use adaptive step size (like Langevin cooling).
    """
    centers = centers.copy()
    # Optimal packing: nearest neighbor distance = 2r. 
    # For N=200 in unit square, r ~ 0.0366, so d ~ 0.0732
    # Set sigma slightly less than expected nearest distance to create repulsion
    sigma = 0.070  # slightly less than 2*0.0366 = 0.0732
    eps = 1.0
    rcut = sigma * 1.5  # cutoff for LJ
    cell_size = rcut
    
    # Wall repulsion parameters
    wall_margin = 0.05  # start repelling within 0.05 of wall
    wall_strength = 100.0
    
    # Adaptive step
    step = 0.002
    best_score = _score(centers)
    best_centers = centers.copy()
    
    start_time = time.time()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if it % 200 == 0:
            # Periodically recompute best
            s = _score(centers)
            if s > best_score:
                best_score = s
                best_centers = centers.copy()
        
        # Build spatial hash
        grid = _spatial_hash(centers, cell_size)
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        for i in range(N):
            neighbors = _neighbor_indices(centers, i, grid, cell_size, rcut)
            for j in neighbors:
                if j <= i:  # count each pair once
                    continue
                rij_vec = centers[i] - centers[j]
                rij = np.linalg.norm(rij_vec)
                if rij < 1e-10:
                    # Perturb
                    forces[i] += np.random.normal(0, 0.001, 2)
                    forces[j] -= np.random.normal(0, 0.001, 2)
                    continue
                if rij < rcut:
                    # Lennard-Jones force magnitude: 24*eps*(2*(sig/rij)^12 - (sig/rij)^6)/rij
                    sr6 = (sigma / rij) ** 6
                    sr12 = sr6 * sr6
                    f_mag = 24.0 * eps * (2.0 * sr12 - sr6) / rij
                    forces[i] += f_mag * rij_vec / rij
                    forces[j] -= f_mag * rij_vec / rij
        
        # Wall repulsion
        for i in range(N):
            x, y = centers[i]
            # Distance to each wall
            dist_left = x + 0.5
            dist_right = 0.5 - x
            dist_bottom = y + 0.5
            dist_top = 0.5 - y
            
            if dist_left < wall_margin:
                forces[i, 0] += wall_strength * (wall_margin - dist_left)
            if dist_right < wall_margin:
                forces[i, 0] -= wall_strength * (wall_margin - dist_right)
            if dist_bottom < wall_margin:
                forces[i, 1] += wall_strength * (wall_margin - dist_bottom)
            if dist_top < wall_margin:
                forces[i, 1] -= wall_strength * (wall_margin - dist_top)
        
        # Normalize forces to avoid blowup
        f_norms = np.linalg.norm(forces, axis=1)
        max_f = np.max(f_norms)
        if max_f > 1e-10:
            forces = forces / max_f * 0.1  # scale to reasonable magnitude
        
        # Update positions
        centers += step * forces
        # Clamp to square (with small margin to avoid exactly on boundary)
        centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Adaptive step: reduce if score drops too much
        if it % 100 == 0 and it > 0:
            s = _score(centers)
            if s < best_score * 0.95:
                step *= 0.8
            elif s > best_score:
                best_score = s
                best_centers = centers.copy()
                step = min(step * 1.1, 0.01)
    
    # Final score
    final_score = _score(centers)
    if final_score > best_score:
        best_score = final_score
        best_centers = centers.copy()
    
    return best_centers, best_score

def _score(centers):
    """Compute the radius score: min of half min-distance and min boundary distance."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    boundary_margin = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary_margin)
    if min_boundary <= 0:
        return 0.0
    # Efficient min pairwise distance using spatial hash
    # For 200 points, O(N^2) is 40k ops, fine
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r = min(min_dist / 2.0, min_boundary)
    return r

def _hexagonal_seed(n):
    """Generate a hexagonal lattice seed with proper spacing."""
    # For 200 circles, try different row/column configs
    best = None
    best_score = 0.0
    
    for n_rows in range(13, 17):
        n_cols = int(np.ceil(n / n_rows))
        if n_rows * n_cols < n:
            continue
        
        # Spacing: fit n_cols points in width 1.0
        dx = 1.0 / (n_cols + 0.5)  # extra margin
        dy = dx * np.sqrt(3) / 2
        
        # Check if n_rows rows fit
        total_height = (n_rows - 1) * dy + dy  # approximate
        if total_height > 1.0:
            continue
        
        points = []
        for row in range(n_rows):
            y = -0.5 + (row + 0.5) * (1.0 / n_rows)
            offset = 0 if row % 2 == 0 else dx / 2
            for col in range(n_cols):
                x = -0.5 + (col + 0.5) * dx + offset
                points.append((x, y))
        
        points = np.array(points[:n])
        
        # Verify inside
        if np.all(np.abs(points) <= 0.5 - 1e-8):
            s = _score(points)
            if s > best_score:
                best_score = s
                best = points.copy()
    
    if best is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best = grid[:n]
        best_score = _score(best)
    
    return best, best_score

def _square_grid(n):
    """Generate a square grid of n points in the unit square centered at origin."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    # Start from hexagonal seed
    centers, score = _hexagonal_seed(N)
    best_centers = centers.copy()
    best_score = score
    
    # Run force-directed optimization
    centers, score = _force_directed(centers, max_iters=8000, deadline=deadline)
    if score > best_score:
        best_score = score
        best_centers = centers.copy()
    
    # If time remains, try from square grid too
    if time.time() < deadline - 10:
        sq = _square_grid(N)
        centers2, score2 = _force_directed(sq, max_iters=4000, deadline=deadline)
        if score2 > best_score:
            best_score = score2
            best_centers = centers2.copy()
    
    return best_centers

# EVOLVE-BLOCK-END
