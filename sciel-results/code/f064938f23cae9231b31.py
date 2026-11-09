"""New approach: Vectorized Simulated Annealing for circle packing."""

import numpy as np

N = 50


def construct():
    """
    Return the centers of 50 equal circles inside the unit disk.
    Uses Simulated Annealing to maximize the minimum radius.
    """
    
    def hex_lattice_init():
        """Create initial hexagonal lattice points within unit circle."""
        points = []
        # Hexagonal lattice with spacing ~ 2*0.125 = 0.25
        # We need to place 50 points
        spacing = 0.25
        sqrt3 = np.sqrt(3)
        
        # Generate hex lattice points
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = i * spacing + (j % 2) * spacing / 2
                y = j * spacing * sqrt3 / 2
                if x**2 + y**2 < 0.85**2:  # Leave margin
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            # If not enough, use a denser lattice
            spacing = 0.22
            points = []
            for j in range(-12, 13):
                for i in range(-12, 13):
                    x = i * spacing + (j % 2) * spacing / 2
                    y = j * spacing * sqrt3 / 2
                    if x**2 + y**2 < 0.85**2:
                        points.append([x, y])
            points = np.array(points)
        
        # Select N points closest to center for better packing
        dists = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        idx = np.argsort(dists)[:N]
        return points[idx]

    def compute_radius(pts):
        """Compute the packing radius for a set of points."""
        # Vectorized pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        min_dist = np.min(dist_mat)
        
        # Distance to boundary
        dists_origin = np.sqrt(np.sum(pts**2, axis=1))
        max_dist_origin = np.max(dists_origin)
        
        r = min(min_dist / 2.0, 1.0 - max_dist_origin)
        return r

    # Initialize
    np.random.seed(42)
    pts = hex_lattice_init()
    
    # Simulated Annealing
    T = 0.1  # Initial temperature
    T_min = 1e-6
    alpha = 0.999  # Cooling rate
    max_iters = 50000
    
    current_r = compute_radius(pts)
    best_pts = pts.copy()
    best_r = current_r
    
    for i in range(max_iters):
        if T < T_min:
            break
            
        # Pick a random point to move
        idx = np.random.randint(0, N)
        
        # Generate a random move
        # Scale step size with temperature
        step_size = 0.05 * T / 0.1 + 0.001
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        
        old_pt = pts[idx].copy()
        pts[idx] = old_pt + np.array([dx, dy])
        
        # Check if point is out of bounds (optional, but helps)
        dist_origin = np.sqrt(pts[idx][0]**2 + pts[idx][1]**2)
        if dist_origin > 1.0:
            # Push back or reject
            pts[idx] = old_pt
            T *= alpha
            continue
            
        new_r = compute_radius(pts)
        
        # Acceptance criterion
        if new_r > current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_pts = pts.copy()
        else:
            # Probabilistic acceptance
            delta = new_r - current_r
            if np.random.rand() < np.exp(delta / T):
                current_r = new_r
            else:
                pts[idx] = old_pt
                
        T *= alpha
        
    return best_pts
