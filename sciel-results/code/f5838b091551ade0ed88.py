# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from itertools import combinations

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    def score(points):
        """Compute R = (d_min / d_max)^2"""
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        return (d_min / d_max) ** 2

    # Generate the 19 sites of a hexagonal cluster
    # Center at (0,0)
    # Sites are (i, j) in a hex grid.
    # Let's use axial coordinates or just cartesian.
    # A 19-site hexagon is a hexagon of radius 2 (side length 3).
    # The sites are:
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1) -- wait, hex grid coords are tricky.
    # Let's use standard cartesian coordinates for a regular hexagonal lattice.
    # Basis vectors: a1 = (1, 0), a2 = (0.5, sqrt(3)/2)
    
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    
    sites = []
    # Radius 2 hexagon includes all (i,j) such that max(|i|, |j|, |i+j|) <= 2?
    # No, that's for cube coords.
    # In axial coords (q,r), distance is (|q| + |r| + |q+r|)/2.
    # We want distance <= 2.
    # Let's just list them manually or generate via loops.
    # q from -2 to 2, r from -2 to 2.
    # Check distance.
    
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Convert axial to cartesian
            # x = (q + r/2) * size
            # y = (sqrt(3)/2 * r) * size
            # Distance in grid units: (|q| + |r| + |q+r|)/2
            dist_grid = (abs(q) + abs(r) + abs(q+r)) / 2.0
            if dist_grid <= 2.0:
                x = q + r/2.0
                y = np.sqrt(3)/2 * r
                sites.append(np.array([x, y]))
                
    sites = np.array(sites)
    # There should be 1 + 6 + 12 = 19 sites.
    # Let's verify count.
    if len(sites) != 19:
        # Fallback if generation failed
        # Just use a grid
        x = np.arange(4)
        y = np.arange(4)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack((xx.ravel(), yy.ravel()))
        
    # We need to choose 16 out of 19. This is equivalent to removing 3.
    # We will iterate over all combinations of 3 indices to remove.
    
    best_score = -1.0
    best_points = None
    top_candidates = [] # List of (score, subset_indices)
    
    all_indices = list(range(19))
    
    # Precompute distances between all 19 sites for the regular hexagon
    # dists_19[i, j] = distance between site i and site j
    dists_19 = pdist(sites, 'euclidean')
    # Convert to square matrix for easy indexing
    # pdist returns condensed form. We can use squareform or just index carefully.
    # Let's use squareform for simplicity, 19x19 is small.
    from scipy.spatial.distance import squareform
    dist_matrix = squareform(dists_19)
    
    for removed in combinations(range(19), 3):
        # Create mask of kept indices
        mask = np.ones(19, dtype=bool)
        for idx in removed:
            mask[idx] = False
        
        kept_indices = np.where(mask)[0]
        
        # Extract sub-matrix
        sub_dist = dist_matrix[np.ix_(kept_indices, kept_indices)]
        
        d_min = np.min(sub_dist)
        d_max = np.max(sub_dist)
        
        if d_max < 1e-10:
            continue
            
        ratio = d_min / d_max
        score_val = ratio ** 2
        
        if score_val > best_score:
            best_score = score_val
            best_points = sites[kept_indices].copy()
            
        # Store top candidates for refinement
        # We will keep the top 10 distinct subsets
        top_candidates.append((score_val, kept_indices))
        
    # Sort top candidates
    top_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Refine the top candidates by optimizing lattice parameters
    # Parameters: r (ratio of |a2| to |a1|), theta (angle of a2)
    # We fix |a1| = 1.
    # a1 = (1, 0)
    # a2 = (r * cos(theta), r * sin(theta))
    
    def lattice_score(params, kept_indices):
        r, theta = params
        # Ensure r > 0 and theta in (0, pi)
        if r < 0.5 or r > 2.0 or theta < 0.1 or theta > np.pi - 0.1:
            return 0.0
            
        a1 = np.array([1.0, 0.0])
        a2 = np.array([r * np.cos(theta), r * np.sin(theta)])
        
        # Regenerate sites with these lattice vectors
        # The sites are generated from axial coordinates (q, r_axial)
        # Note: 'r' in params is different from 'r_axial' in the loop above.
        # Let's rename the loop variable to i_axial, j_axial to avoid confusion.
        
        new_sites = []
        for q in range(-2, 3):
            for r_axial in range(-2, 3):
                dist_grid = (abs(q) + abs(r_axial) + abs(q+r_axial)) / 2.0
                if dist_grid <= 2.0:
                    # Cartesian coord = q * a1 + r_axial * a2
                    pos = q * a1 + r_axial * a2
                    new_sites.append(pos)
                    
        new_sites = np.array(new_sites)
        
        # Select kept points
        pts = new_sites[kept_indices]
        
        dists = pdist(pts, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-10:
            return 0.0
            
        return -(d_min / d_max) ** 2 # Minimize negative score

    # Refine top 5 candidates
    for score_val, kept_indices in top_candidates[:5]:
        if time.time() > deadline - 10:
            break
            
        # Initial guess: regular hexagon (r=1, theta=pi/3)
        x0 = np.array([1.0, np.pi/3])
        
        try:
            res = minimize(
                lattice_score, 
                x0, 
                args=(kept_indices,),
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200}
            )
            
            if res.success or res.fun < 0:
                # Calculate final score
                r_opt, theta_opt = res.x
                a1 = np.array([1.0, 0.0])
                a2 = np.array([r_opt * np.cos(theta_opt), r_opt * np.sin(theta_opt)])
                
                new_sites = []
                for q in range(-2, 3):
                    for r_axial in range(-2, 3):
                        dist_grid = (abs(q) + abs(r_axial) + abs(q+r_axial)) / 2.0
                        if dist_grid <= 2.0:
                            pos = q * a1 + r_axial * a2
                            new_sites.append(pos)
                            
                new_sites = np.array(new_sites)
                pts = new_sites[kept_indices]
                
                s = score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
                    
        except Exception:
            pass

    if best_points is None:
        # Fallback
        x = np.arange(4)
        y = np.arange(4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack((xx.ravel(), yy.ravel()))

    return best_points

# EVOLVE-BLOCK-END
