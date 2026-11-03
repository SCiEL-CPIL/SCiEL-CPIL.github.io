"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(centers):
    """Compute the radius score: min of half min-distance and min boundary distance."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint: centers must be inside [-0.5, 0.5]^2
    boundary_margin = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary_margin)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    # Use efficient computation
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r = min(min_dist / 2.0, min_boundary)
    return r

def _hexagonal_lattice(n, angle_deg=0):
    """Generate a hexagonal lattice packing, rotated."""
    angle = np.radians(angle_deg)
    # Optimal hexagonal packing density is pi / (2*sqrt(3)) ~ 0.9069
    # Area per circle ~ pi*r^2. Total area = N * pi * r^2.
    # Square area = 1. So N * pi * r^2 <= 1 * density_eff.
    # Let's just generate a grid that fits.
    
    # Try to fit N points in a hexagonal lattice within [-0.5, 0.5]^2
    # The unit cell of a hexagonal lattice has area (sqrt(3)/2) * d^2 where d is distance between centers.
    # Actually, let's just use a rectangular grid with hexagonal offset.
    
    # Estimate number of rows and columns
    # Aspect ratio of square is 1. Hexagonal lattice aspect ratio for optimal packing is 1.
    # Let's try to find integer rows and cols that minimize the required square size.
    
    best_points = None
    best_size = float('inf')
    
    # Try different row counts
    for rows in range(10, 20):
        cols = int(np.ceil(n / rows))
        if rows * cols < n:
            continue
            
        # Spacing
        # Horizontal spacing dx, vertical spacing dy = dx * sqrt(3)/2
        # Total width = cols * dx + dx/2 (for offset) ? No, width is roughly cols*dx.
        # To fit in 1.0 width, dx <= 1.0 / cols.
        # To fit in 1.0 height, rows * dy <= 1.0 => rows * dx * sqrt(3)/2 <= 1.0 => dx <= 1.0 / (rows * sqrt(3)/2).
        
        dx_w = 1.0 / cols
        dx_h = 1.0 / (rows * np.sqrt(3) / 2)
        dx = min(dx_w, dx_h)
        
        dy = dx * np.sqrt(3) / 2
        
        points = []
        for r in range(rows):
            y = -0.5 + (r + 0.5) * dy
            if abs(y) > 0.5:
                continue
            offset = 0 if r % 2 == 0 else dx / 2
            for c in range(cols):
                x = -0.5 + (c + 0.5) * dx + offset
                if abs(x) <= 0.5 and abs(y) <= 0.5:
                    points.append([x, y])
        
        if len(points) >= n:
            points = points[:n]
            points = np.array(points)
            
            # Rotate
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rotated = np.zeros_like(points)
            rotated[:, 0] = points[:, 0] * cos_a - points[:, 1] * sin_a
            rotated[:, 1] = points[:, 0] * sin_a + points[:, 1] * cos_a
            
            # Check if inside
            if np.all(np.abs(rotated) <= 0.5):
                # Calculate score to see which is better
                s = _score(rotated)
                if best_points is None or s > _score(best_points):
                    best_points = rotated
                    best_size = s
                    
    if best_points is not None:
        return best_points
    return None

def _construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # 1. Initialize with best hexagonal lattice
    best_centers = None
    best_score = 0.0
    
    for angle in range(0, 90, 5):
        if time.time() > deadline - 5:
            break
        pts = _hexagonal_lattice(N, angle)
        if pts is not None:
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_centers = pts.copy()
                
    if best_centers is None:
        # Fallback to random if lattice fails
        np.random.seed(0)
        best_centers = np.random.uniform(-0.45, 0.45, (N, 2))
        best_score = _score(best_centers)
        
    # 2. Repulsion-based relaxation with spatial hash
    # We want to maximize the minimum distance.
    # This is equivalent to minimizing the sum of inverse powers of distances (Joule energy).
    # E = sum_{i<j} 1/d_ij^k + sum_i 1/(0.5 - |c_i|_inf)^k
    
    centers = best_centers.copy()
    current_energy = float('inf')
    
    # Spatial hash parameters
    # We need to check neighbors within a certain distance.
    # As the packing improves, the minimum distance increases.
    # Let's estimate the current minimum distance to set the grid cell size.
    
    def get_min_dist_and_bound(centers):
        # O(N^2) is too slow for frequent calls if we do it every step.
        # But we need it for the score and to adapt grid size.
        # We will do it less frequently or use a fast approximation.
        # For now, let's just compute it. N=200, N^2=40000. This is fast.
        dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        boundary_margin = 0.5 - np.max(np.abs(centers), axis=1)
        min_bound = np.min(boundary_margin)
        return min_dist, min_bound
        
    min_d, min_b = get_min_dist_and_bound(centers)
    current_min_constraint = min(min_d, min_b)
    
    # Grid cell size should be larger than the expected minimum distance
    # to ensure we capture all neighbors.
    # Let's use a cell size that is roughly 2x the current minimum constraint.
    
    iters = 5000
    T = 0.01
    cool_rate = 0.999
    
    np.random.seed(42)
    
    for it in range(iters):
        if time.time() > deadline:
            break
            
        # Recompute min constraint every 100 steps to update grid size
        if it % 100 == 0:
            min_d, min_b = get_min_dist_and_bound(centers)
            current_min_constraint = min(min_d, min_b)
            if current_min_constraint < 1e-6:
                # Stuck or invalid, add noise
                centers += np.random.normal(0, 0.01, centers.shape)
                centers = np.clip(centers, -0.5, 0.5)
                continue
            
        cell_size = max(2.0 * current_min_constraint, 0.01)
        
        # Build spatial hash
        # Map (x, y) to (col, row)
        # Coordinates are in [-0.5, 0.5]
        # Shift to [0, 1]
        coords = centers + 0.5
        cols = (coords[:, 0] / cell_size).astype(int)
        rows = (coords[:, 1] / cell_size).astype(int)
        
        # Clamp indices to valid grid range
        max_col = int(1.0 / cell_size) + 1
        max_row = int(1.0 / cell_size) + 1
        cols = np.clip(cols, 0, max_col)
        rows = np.clip(rows, 0, max_row)
        
        # Dictionary of lists
        grid = {}
        for i in range(N):
            key = (cols[i], rows[i])
            if key not in grid:
                grid[key] = []
            grid[key].append(i)
        
        # Compute forces/energy
        # We will use a repulsive force F = 1/d^2 * direction
        # And a boundary repulsive force.
        
        forces = np.zeros_like(centers)
        energy = 0.0
        
        # Pairwise repulsion
        for i in range(N):
            c_i = centers[i]
            col_i = cols[i]
            row_i = rows[i]
            
            # Check 3x3 neighborhood
            for dc in range(-1, 2):
                for dr in range(-1, 2):
                    key = (col_i + dc, row_i + dr)
                    if key in grid:
                        for j in grid[key]:
                            if i == j:
                                continue
                            c_j = centers[j]
                            diff = c_i - c_j
                            d_sq = np.dot(diff, diff)
                            if d_sq < 1e-12:
                                # Random jitter
                                diff = np.random.normal(0, 1e-6, 2)
                                d_sq = np.dot(diff, diff)
                                
                            d = np.sqrt(d_sq)
                            # Force magnitude
                            f_mag = 1.0 / (d_sq * d) # Gradient of 1/d^2 is -2/d^3 * dir. We want to push apart, so force on i is +dir/d^3
                            forces[i] += diff * f_mag
                            energy += 1.0 / d_sq
                            
        # Boundary repulsion
        # Distance to boundary: 0.5 - max(|x|, |y|)
        # Force pushes towards center if close to boundary.
        for i in range(N):
            x, y = centers[i]
            d_bound = 0.5 - max(abs(x), abs(y))
            if d_bound < current_min_constraint * 2: # Only if close
                # Gradient of 1/d_bound^2
                # If x is limiting (|x| > |y|), force is -sign(x) * f
                # If y is limiting, force is -sign(y) * f
                if abs(x) >= abs(y):
                    f_mag = 1.0 / (d_bound * d_bound * d_bound)
                    forces[i, 0] -= np.sign(x) * f_mag
                    energy += 1.0 / (d_bound * d_bound)
                else:
                    f_mag = 1.0 / (d_bound * d_bound * d_bound)
                    forces[i, 1] -= np.sign(y) * f_mag
                    energy += 1.0 / (d_bound * d_bound)
                    
        # Update positions
        # Step size should be adaptive
        # Let's use a fixed small step size scaled by T
        step_size = 0.01 * T
        centers += forces * step_size
        
        # Clip to square
        centers = np.clip(centers, -0.5, 0.5)
        
        # Cooling
        T *= cool_rate
        
        # Check score periodically
        if it % 50 == 0:
            s = _score(centers)
            if s > best_score:
                best_score = s
                best_centers = centers.copy()
                
    return best_centers

def construct():
    return _construct()
# EVOLVE-BLOCK-END
