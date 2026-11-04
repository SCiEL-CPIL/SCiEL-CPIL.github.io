import numpy as np
import time

N = 11
MIN_DIST = np.sqrt(3)

def construct():
    # Normal vectors for the 6 sides of a regular hexagon (flat-top orientation)
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    def min_enclosing_hexagon_side(points):
        """
        Compute the minimum side length of a regular hexagon (centered at origin)
        that encloses all points. Also tries rotating the hexagon to minimize side length.
        Returns (min_side, best_rotation)
        """
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        best_L = float('inf')
        best_rot = 0.0
        
        # Try different rotations of the enclosing hexagon
        # The hexagon has 6-fold symmetry, so we only need to check 0 to 30 degrees
        for rot_deg in np.linspace(0, 30, 31):
            rot = np.radians(rot_deg)
            cos_r, sin_r = np.cos(rot), np.sin(rot)
            rot_normals = np.array([[cos_r * normals[k,0] - sin_r * normals[k,1],
                                     sin_r * normals[k,0] + cos_r * normals[k,1]] for k in range(6)])
            
            projections = np.abs(points_centered @ rot_normals.T)
            max_proj = np.max(projections)
            L = max_proj / (np.sqrt(3) / 2)
            
            if L < best_L:
                best_L = L
                best_rot = rot_deg
        
        return best_L, best_rot

    def cost_func(params):
        # params is a 1D array of 22 elements: [x1, y1, x2, y2, ..., x11, y11]
        points = params.reshape(-1, 2)
        
        # Center the points at the origin
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Calculate the side length L of the smallest enclosing regular hexagon
        # We use a fixed orientation for the cost function to make it smooth,
        # but we'll optimize rotation separately later if needed.
        # For now, use fixed normals (0 degrees rotation for hexagon)
        projections = np.abs(points_centered @ normals.T)
        max_proj = np.max(projections)
        L_enc = max_proj / (np.sqrt(3) / 2)
        
        # Calculate overlap penalty
        # We want distance >= MIN_DIST
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points_centered[i] - points_centered[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2
        
        # Combine costs
        return L_enc + 100.0 * penalty

    # Random restarts to find a good local minimum
    best_L = float('inf')
    best_params = None
    
    # Initial guess: place points in a hexagonal grid pattern roughly
    for restart in range(50):
        # Random initialization
        if restart == 0:
            # Hexagonal lattice initialization
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
        else:
            init_pts = np.random.uniform(-3, 3, (N, 2))
            
        initial_params = init_pts.flatten()
        
        # Use Nelder-Mead for optimization
        from scipy.optimize import minimize
        res = minimize(cost_func, initial_params, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.success:
            final_pts = res.x.reshape(-1, 2)
            centroid = np.mean(final_pts, axis=0)
            final_pts_centered = final_pts - centroid
            
            # Check for overlaps
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(final_pts_centered[i] - final_pts_centered[j])
                    if d < MIN_DIST - 1e-5:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Compute the true minimum enclosing hexagon side length with optimal rotation
                L_enc, best_rot = min_enclosing_hexagon_side(final_pts)
                
                if L_enc < best_L:
                    best_L = L_enc
                    best_params = res.x

    # If no valid solution found, fallback to a safe large solution
    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Construct the output
    inner_pts = best_params.reshape(-1, 2)
    centroid = np.mean(inner_pts, axis=0)
    inner_pts_centered = inner_pts - centroid
    
    # Find the optimal rotation for the outer hexagon
    L_enc, best_rot_deg = min_enclosing_hexagon_side(inner_pts)
    
    inner = [[float(x), float(y), 0.0] for x, y in inner_pts_centered]
    
    outer_side = L_enc
    
    return inner, (0.0, 0.0), outer_side, float(best_rot_deg)
