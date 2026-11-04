import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 11
MIN_DIST = np.sqrt(3)

def construct():
    # Normal vectors for the 6 sides of a regular hexagon (flat-top orientation)
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    def enclosing_hexagon_side_length(points):
        """Compute the minimum side length of a regular hexagon that encloses the given points.
        
        For a regular hexagon centered at origin with side length L, the distance from center
        to each side is L * sqrt(3) / 2. The constraint is that for each of the 6 normal directions,
        the maximum projection of any point onto that normal must be <= L * sqrt(3) / 2.
        """
        # Center the points
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # Project onto all 6 normals
        projections = np.abs(centered @ normals.T)
        max_proj = np.max(projections)
        
        # Side length L such that max_proj = L * sqrt(3) / 2
        return max_proj / (np.sqrt(3) / 2)
    
    def cost_func(params):
        """Cost function: minimize enclosing hexagon size, penalize overlaps heavily."""
        points = params.reshape(-1, 2)
        
        # Calculate enclosing hexagon side length
        L_enc = enclosing_hexagon_side_length(points)
        
        # Calculate overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2
        
        # Strong penalty for overlaps to ensure validity
        return L_enc + 1000.0 * penalty
    
    def is_valid(points, tol=1e-6):
        """Check if all pairwise distances are >= MIN_DIST."""
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j])
                if d < MIN_DIST - tol:
                    return False
        return True
    
    # Generate initial configurations
    best_L = float('inf')
    best_params = None
    
    def hex_grid_init():
        """Generate hexagonal lattice points and select N closest to center."""
        pts = []
        for q in range(-4, 5):
            for r in range(-4, 5):
                x = 1.5 * q
                y = np.sqrt(3)/2 * q + np.sqrt(3) * r
                pts.append([x, y])
        pts = np.array(pts)
        dists = np.linalg.norm(pts, axis=1)
        indices = np.argsort(dists)[:N]
        return pts[indices]
    
    def random_init(scale=3.0):
        """Random initialization within a circle."""
        return np.random.uniform(-scale, scale, (N, 2))
    
    # Multiple restarts with different initializations
    for restart in range(80):
        if restart == 0:
            # Hexagonal lattice initialization
            init_pts = hex_grid_init()
            # Add small noise to break symmetry
            init_pts += np.random.normal(0, 0.05, init_pts.shape)
        elif restart == 1:
            # Tighter hexagonal lattice
            pts = []
            for q in range(-3, 4):
                for r in range(-3, 4):
                    x = 1.5 * q
                    y = np.sqrt(3)/2 * q + np.sqrt(3) * r
                    pts.append([x, y])
            pts = np.array(pts)
            dists = np.linalg.norm(pts, axis=1)
            indices = np.argsort(dists)[:N]
            init_pts = pts[indices]
            init_pts += np.random.normal(0, 0.1, init_pts.shape)
        elif restart < 10:
            # Random with varying scales
            scale = 2.0 + 0.5 * (restart - 1)
            init_pts = random_init(scale)
        else:
            # Random
            init_pts = random_init(3.0)
        
        initial_params = init_pts.flatten()
        
        # Use Nelder-Mead for optimization
        res = minimize(cost_func, initial_params, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
        
        final_pts = res.x.reshape(-1, 2)
        
        # Check validity
        if is_valid(final_pts):
            L_enc = enclosing_hexagon_side_length(final_pts)
            if L_enc < best_L:
                best_L = L_enc
                best_params = res.x.copy()
    
    # If no valid solution found, try more aggressive search
    if best_params is None:
        # Fallback: use a larger hexagonal arrangement
        pts = []
        for q in range(-4, 5):
            for r in range(-4, 5):
                x = 1.5 * q
                y = np.sqrt(3)/2 * q + np.sqrt(3) * r
                pts.append([x, y])
        pts = np.array(pts)
        dists = np.linalg.norm(pts, axis=1)
        indices = np.argsort(dists)[:N]
        best_params = pts[indices].flatten()
        best_L = enclosing_hexagon_side_length(pts[indices])
    
    # Refine with local optimization starting from best solution
    if best_params is not None:
        res = minimize(cost_func, best_params, method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        final_pts = res.x.reshape(-1, 2)
        if is_valid(final_pts):
            L_enc = enclosing_hexagon_side_length(final_pts)
            if L_enc < best_L:
                best_L = L_enc
                best_params = res.x.copy()
    
    # Construct the output
    inner_pts = best_params.reshape(-1, 2)
    
    # Center the points at the origin
    centroid = np.mean(inner_pts, axis=0)
    inner_pts_centered = inner_pts - centroid
    
    inner = [[float(x), float(y), 0.0] for x, y in inner_pts_centered]
    
    # Add a small safety margin to the side length
    outer_side = best_L + 0.01
    
    return inner, (0.0, 0.0), outer_side, 0.0
