"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    # Start with a hexagonal lattice arrangement, then refine with projected gradient ascent
    # on the minimum pairwise distance (a convex-ish objective in practice for local search).
    
    # 1. Initialize with a dense hexagonal lattice
    # For 200 circles, a good starting arrangement is a hexagonal packing.
    # We'll create a hexagonal grid that fits in the unit square [-0.5, 0.5]^2.
    
    def hex_lattice_init(n):
        """Create initial hexagonal lattice positions for n circles."""
        # Try different spacings to fit n circles in the square
        best_pos = None
        best_min_dist = 0
        
        # Hexagonal lattice: rows offset by half the spacing
        # Spacing s between centers
        # Try a range of spacings
        for s in np.linspace(0.03, 0.05, 40):
            # Generate hex lattice points
            points = []
            # Number of rows needed
            # In a hex lattice, vertical spacing is s * sqrt(3)/2
            vert_spacing = s * np.sqrt(3) / 2
            num_rows = int(1.0 / vert_spacing) + 2
            num_cols = int(1.0 / s) + 2
            
            for row in range(num_rows):
                y = -0.5 + vert_spacing * (row + 0.5)
                if y > 0.5:
                    break
                offset = 0 if row % 2 == 0 else s / 2
                for col in range(num_cols):
                    x = -0.5 + s * (col + 0.5) + offset
                    if x > 0.5:
                        break
                    points.append([x, y])
            
            points = np.array(points)
            if len(points) < n:
                continue
            
            # Keep the n points closest to center or just take first n
            # Better: select n points that maximize minimum distance
            # For now, just take first n
            points = points[:n]
            
            # Compute min pairwise distance
            if len(points) < 2:
                continue
            from scipy.spatial.distance import pdist
            d = pdist(points)
            min_d = d.min()
            
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = points.copy()
        
        if best_pos is None:
            # Fallback: square grid
            m = int(np.ceil(np.sqrt(n)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            best_pos = grid[:n]
        
        return best_pos
    
    pos = hex_lattice_init(N)
    pos = pos.astype(np.float64)
    
    # 2. Refine with projected gradient ascent on min pairwise distance
    # Objective: maximize min_{i<j} ||p_i - p_j||
    # This is equivalent to minimizing -min_{i<j} ||p_i - p_j||
    # But min of distances is non-smooth. We use a smooth approximation or
    # alternatively, we can use a different approach: maximize the minimum distance
    # by iteratively pushing apart the closest pairs.
    
    # Use a simple iterative method: at each step, find the closest pair and
    # push them apart, while respecting boundary constraints.
    
    def boundary_constraint(p):
        """Project p onto the feasible region [-0.5, 0.5]^2."""
        return np.clip(p, -0.5, 0.5)
    
    # Iterative refinement
    max_iters = 500
    lr = 1e-4
    
    for it in range(max_iters):
        # Compute all pairwise distances
        from scipy.spatial.distance import cdist
        D = cdist(pos, pos)
        np.fill_diagonal(D, np.inf)
        
        # Find the minimum distance pair
        min_idx = np.unravel_index(np.argmin(D), D.shape)
        min_dist = D[min_idx]
        
        if min_dist < 1e-12:
            break
        
        # Compute gradient of min distance w.r.t. positions
        # The gradient of ||p_i - p_j|| w.r.t. p_i is (p_i - p_j) / ||p_i - p_j||
        # and w.r.t. p_j is (p_j - p_i) / ||p_i - p_j||
        # We want to maximize min_dist, so we move in the direction of the gradient
        
        i, j = min_idx
        diff = pos[i] - pos[j]
        norm = np.linalg.norm(diff)
        if norm < 1e-12:
            # Perturb randomly
            diff = np.random.randn(2) * 1e-6
            norm = np.linalg.norm(diff)
        grad_i = diff / norm
        grad_j = -grad_i
        
        # Update positions
        pos[i] += lr * grad_i
        pos[j] += lr * grad_j
        
        # Project back to feasible region
        pos = boundary_constraint(pos)
        
        # Also, to avoid getting stuck, add a small repulsive force from all other pairs
        # This helps escape local minima
        if it % 10 == 0:
            # Compute repulsive forces from all pairs with small distances
            D = cdist(pos, pos)
            np.fill_diagonal(D, np.inf)
            # For pairs with distance < 2 * min_dist, add repulsion
            threshold = 2 * min_dist
            for a in range(N):
                for b in range(a + 1, N):
                    if D[a, b] < threshold:
                        d_ab = pos[a] - pos[b]
                        n_ab = np.linalg.norm(d_ab)
                        if n_ab > 1e-12:
                            force = (d_ab / n_ab) * (1.0 / n_ab) * lr * 0.1
                            pos[a] += force
                            pos[b] -= force
            pos = boundary_constraint(pos)
    
    # 3. Final projection to ensure all points are in [-0.5, 0.5]^2
    pos = np.clip(pos, -0.5, 0.5)
    
    # 4. Ensure no coincident points
    from scipy.spatial.distance import pdist
    d = pdist(pos)
    if d.min() < 1e-10:
        # Perturb coincident points
        for a in range(N):
            for b in range(a + 1, N):
                if np.linalg.norm(pos[a] - pos[b]) < 1e-10:
                    pos[b] += np.random.randn(2) * 1e-6
                    pos = np.clip(pos, -0.5, 0.5)
    
    return pos
# EVOLVE-BLOCK-END
