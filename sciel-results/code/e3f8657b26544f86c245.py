import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 11
MIN_DIST = np.sqrt(3)

def construct():
    # Normal vectors for the 6 sides of a regular hexagon (flat-top orientation)
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    def compute_enclosing_hexagon(points):
        """Compute the minimum enclosing regular hexagon for a set of points."""
        # Center the points
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # For each rotation angle, compute the enclosing hexagon side length
        best_L = float('inf')
        best_angle = 0.0
        
        # Try rotations in small steps
        for rot in np.linspace(0, np.pi/3, 20):
            rot_matrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
            rotated_normals = normals @ rot_matrix.T
            
            # Project points onto each normal
            projections = np.abs(points_centered @ rotated_normals.T)
            max_proj = np.max(projections)
            L = max_proj / (np.sqrt(3) / 2)
            
            if L < best_L:
                best_L = L
                best_angle = rot
        
        return best_L, best_angle, centroid
    
    def cost_func(params):
        """Cost function: minimize convex hull area + overlap penalty."""
        points = params.reshape(-1, 2)
        
        # Compute convex hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume  # In 2D, volume is area
        except Exception:
            # Degenerate case
            hull_area = 100.0
        
        # Overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2
        
        # Combined cost: minimize area, heavily penalize overlaps
        return hull_area + 50.0 * penalty
    
    # Random restarts to find a good local minimum
    best_area = float('inf')
    best_params = None
    
    # Initial guess: hexagonal lattice pattern
    def get_hex_lattice_init():
        pts = []
        for q in range(-3, 4):
            for r in range(-3, 4):
                x = 1.5 * q
                y = np.sqrt(3)/2 * q + np.sqrt(3) * r
                pts.append([x, y])
        pts = np.array(pts)
        dists = np.linalg.norm(pts, axis=1)
        indices = np.argsort(dists)[:N]
        return pts[indices]
    
    for restart in range(80):
        # Random initialization
        if restart == 0:
            init_pts = get_hex_lattice_init()
            init_pts += np.random.normal(0, 0.05, init_pts.shape)
        elif restart < 10:
            # Perturbed lattice
            init_pts = get_hex_lattice_init()
            init_pts += np.random.normal(0, 0.3, init_pts.shape)
        else:
            init_pts = np.random.uniform(-3, 3, (N, 2))
        
        initial_params = init_pts.flatten()
        
        # Use Nelder-Mead for optimization
        res = minimize(cost_func, initial_params, method='Nelder-Mead', 
                       options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        final_pts = res.x.reshape(-1, 2)
        
        # Check for overlaps
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(final_pts[i] - final_pts[j])
                if d < MIN_DIST - 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Compute convex hull area
            try:
                hull = ConvexHull(final_pts)
                hull_area = hull.volume
            except Exception:
                continue
            
            if hull_area < best_area:
                best_area = hull_area
                best_params = res.x
    
    # If no valid solution found, fallback to a safe large solution
    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Compute the enclosing hexagon for the best cluster
    inner_pts = best_params.reshape(-1, 2)
    L_enc, hex_angle, centroid = compute_enclosing_hexagon(inner_pts)
    
    # The inner hexagons are unit hexagons (side 1), so they extend sqrt(3)/2 from center
    # The enclosing hexagon must contain all inner hexagons, so we need to account for their size
    # Distance from inner hex center to its vertices is 1.0 (circumradius = side length)
    # So we need L_enc to be large enough to contain the inner hexagons
    
    # Actually, we need to check: does the enclosing hexagon contain all inner hexagons?
    # The inner hexagons have vertices at distance 1.0 from their centers
    # We need to verify that the enclosing hexagon contains all these vertices
    
    # For now, add a safety margin: the inner hexagons extend 1.0 from center
    # So the enclosing hexagon side should be at least L_enc + 1.0
    # But this is conservative. Let's be more precise.
    
    # The vertices of each inner hexagon are at distance 1.0 from its center
    # We need to find the maximum projection of all inner hexagon vertices onto the hexagon normals
    
    def compute_precise_enclosing_hexagon(points):
        """Compute the precise enclosing hexagon that contains all inner hexagons."""
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        best_L = float('inf')
        best_angle = 0.0
        
        for rot in np.linspace(0, np.pi/3, 30):
            rot_matrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
            rotated_normals = normals @ rot_matrix.T
            
            max_proj = 0.0
            for i in range(N):
                # For each inner hexagon, compute the max projection of its vertices
                # Inner hexagon vertices are at angles 0, 60, 120, 180, 240, 300 degrees from its center
                hex_angles = np.linspace(0, 2*np.pi, 7)[:-1]
                vertices = np.array([[np.cos(a), np.sin(a)] for a in hex_angles])
                vertices_centered = vertices + points_centered[i]
                
                projections = np.abs(vertices_centered @ rotated_normals.T)
                max_proj = max(max_proj, np.max(projections))
            
            L = max_proj / (np.sqrt(3) / 2)
            
            if L < best_L:
                best_L = L
                best_angle = rot
        
        return best_L, best_angle, centroid
    
    L_enc, hex_angle, centroid = compute_precise_enclosing_hexagon(inner_pts)
    
    # Construct the output
    inner = [[float(x), float(y), 0.0] for x, y in inner_pts]
    
    outer_side = L_enc
    
    return inner, (0.0, 0.0), outer_side, 0.0
