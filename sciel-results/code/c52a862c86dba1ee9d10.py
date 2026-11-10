"""Packing 200 equal circles in a unit square using hexagonal init + repulsive force refinement."""
import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # seconds, safety margin before 120s limit
    
    # --- Step 1: Hexagonal lattice initialization ---
    # Place circles in a hexagonal close-packed arrangement within the square
    # This gives a good starting configuration with minimal overlaps
    
    def hexagonal_init(n):
        """Generate n points in hexagonal lattice pattern inside [-0.5, 0.5]^2."""
        # Estimate grid dimensions
        # Hexagonal packing: rows offset by half spacing
        # Try different row counts to fit n points
        
        best_pts = None
        best_spacing = -1
        
        for rows in range(10, 30):
            cols = int(np.ceil(n / rows))
            # Spacing in x and y
            # In hexagonal packing, vertical spacing = s * sqrt(3)/2
            # Horizontal spacing = s
            # We need cols * s <= 1 and rows * s * sqrt(3)/2 <= 1
            # So s <= 1/cols and s <= 2/(rows * sqrt(3))
            s_x = 1.0 / cols
            s_y = 2.0 / (rows * np.sqrt(3))
            s = min(s_x, s_y)
            
            pts = []
            for i in range(rows):
                y = -0.5 + s * np.sqrt(3) / 2 * (i + 0.5)
                offset = 0.0 if i % 2 == 0 else s / 2
                for j in range(cols):
                    x = -0.5 + s * (j + 0.5) + offset
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                        pts.append([x, y])
            
            if len(pts) >= n:
                pts = np.array(pts[:n])
                # Compute min distance
                from scipy.spatial.distance import pdist
                min_d = pdist(pts).min()
                if min_d > best_spacing:
                    best_spacing = min_d
                    best_pts = pts
        
        if best_pts is None:
            # Fallback: simple grid
            m = int(np.ceil(np.sqrt(n)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            best_pts = grid[:n]
        
        return best_pts
    
    centers = hexagonal_init(N)
    
    # Compute initial score
    def compute_score(pts):
        """Compute the radius for a given configuration."""
        if len(pts) < 2:
            return 0.0
        # Min distance between pairs
        from scipy.spatial.distance import pdist
        min_pair_dist = pdist(pts).min()
        # Min distance to boundary
        min_boundary = np.min(np.minimum(0.5 - np.abs(pts), 0.5))
        return min(min_pair_dist / 2.0, min_boundary)
    
    best_score = compute_score(centers)
    best_centers = centers.copy()
    
    # --- Step 2: Repulsive force-based refinement ---
    # Use a simple force-directed approach where circles repel each other
    # and are attracted to stay within the square
    
    def refine(pts, max_time=100.0):
        """Refine the configuration using repulsive forces."""
        pts = pts.astype(np.float64).copy()
        n = len(pts)
        
        # Adaptive step size
        step = 0.001
        decay = 0.999
        min_step = 1e-8
        
        # Force parameters
        repel_strength = 1.0
        boundary_strength = 10.0
        soft_radius = 0.05  # soft constraint radius for repulsion
        
        elapsed = time.time() - start_time
        
        while step > min_step and (time.time() - start_time) < max_time:
            forces = np.zeros_like(pts)
            
            # Pairwise repulsion (only for close pairs to save time)
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            
            # Find pairs within a certain distance
            pairs = tree.query_pairs(r=0.15)
            
            for i, j in pairs:
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random small perturbation
                    diff = np.random.randn(2) * 1e-10
                    dist = np.linalg.norm(diff)
                
                # Repulsive force: stronger when closer
                if dist < soft_radius:
                    force_mag = repel_strength * (soft_radius - dist) / soft_radius
                else:
                    force_mag = repel_strength * (soft_radius - dist) / (dist + 1e-10) * 0.1
                
                if dist > 1e-10:
                    force = force_mag * diff / dist
                    forces[i] += force
                    forces[j] -= force
            
            # Boundary forces: push points back into the square
            for i in range(n):
                for k in range(2):
                    if pts[i, k] > 0.5 - 0.01:
                        forces[i, k] -= boundary_strength * (pts[i, k] - (0.5 - 0.01))
                    elif pts[i, k] < -0.5 + 0.01:
                        forces[i, k] += boundary_strength * ((-0.5 + 0.01) - pts[i, k])
            
            # Update positions
            pts += step * forces
            
            # Clamp to square (hard constraint)
            pts = np.clip(pts, -0.5, 0.5)
            
            # Decay step size
            step *= decay
            
            # Check score periodically
            if int((time.time() - start_time) * 10) % 10 == 0:
                current_score = compute_score(pts)
                if current_score > best_score:
                    best_score = current_score
                    best_centers = pts.copy()
        
        return pts, best_score, best_centers
    
    # Run refinement
    centers, best_score, best_centers = refine(centers, max_time=100.0)
    
    # --- Step 3: Local optimization with Nelder-Mead on subsets ---
    # Try to improve further by optimizing small subsets of points
    
    from scipy.optimize import minimize
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try Nelder-Mead on the full configuration
        # This is expensive, so we'll try a limited version
        
        def negative_score(pts):
            """Negative of the score for minimization."""
            pts = pts.reshape(-1, 2)
            if np.any(np.abs(pts) > 0.5 + 1e-10):
                return 1.0  # Invalid
            from scipy.spatial.distance import pdist
            min_pair_dist = pdist(pts).min()
            min_boundary = np.min(np.minimum(0.5 - np.abs(pts), 0.5))
            return -min(min_pair_dist / 2.0, min_boundary)
        
        # Optimize with Nelder-Mead
        x0 = best_centers.flatten()
        try:
            result = minimize(negative_score, x0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < -best_score:
                new_pts = result.x.reshape(-1, 2)
                new_score = -result.fun
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_pts
        except:
            pass
        
        elapsed = time.time() - start_time
    
    # Final check
    if time.time() - start_time > deadline:
        # Return best found so far
        return best_centers
    
    return best_centers
# EVOLVE-BLOCK-END
