import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    N = 12
    
    # Generate 12 centers from a hexagonal tiling (pointy-top)
    # 3 columns, 4 rows
    centers = []
    for i in range(-1, 2):  # -1, 0, 1
        for j in range(-1, 3):  # -1, 0, 1, 2
            x = i * np.sqrt(3)
            y = 2 * j + (1 if i % 2 != 0 else 0)
            centers.append([x, y])
            
    centers = np.array(centers)
    centers -= np.mean(centers, axis=0)
    
    # Define the 6 vertices of a unit pointy-top hexagon
    # Angles: 90, 150, 210, 270, 330, 30 degrees
    angles = np.deg2rad([90, 150, 210, 270, 330, 30])
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Pre-compute rotated vertices for each inner hexagon
    # We will optimize: global rotation (theta), and global translation (tx, ty)
    # Each inner hexagon has the same rotation theta and is placed at its center + (tx, ty)
    
    def calc_L(params):
        theta, tx, ty = params
        rot = np.deg2rad(theta)
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        
        max_L = 0.0
        for c_center in centers:
            # Rotate center
            rc = R @ c_center
            rc[0] += tx
            rc[1] += ty
            
            # Rotate vertices
            rv = R @ verts.T
            # Add center
            rv = rv.T + rc
            
            # Calculate L_req for each vertex
            # For a regular hexagon with side length L, centered at origin, pointy-top:
            # The boundary is defined by:
            # |y| <= L
            # |x| <= L * sqrt(3) / 2  =>  |x| / (sqrt(3)/2) <= L  =>  2/sqrt(3) * |x| <= L
            # |y + x/sqrt(3)| <= L
            # |y - x/sqrt(3)| <= L
            
            x = rv[:, 0]
            y = rv[:, 1]
            
            L1 = np.abs(y)
            L2 = (2 / np.sqrt(3)) * np.abs(x)
            L3 = (2 / np.sqrt(3)) * np.abs(y + x / np.sqrt(3))
            L4 = (2 / np.sqrt(3)) * np.abs(y - x / np.sqrt(3))
            
            L_max = np.max(np.maximum(np.maximum(L1, L2), np.maximum(L3, L4)))
            if L_max > max_L:
                max_L = L_max
                
        return max_L
    
    # Coarse grid search
    best_L = float('inf')
    best_params = [0, 0, 0]
    
    for theta in range(0, 180, 5):
        for tx in np.linspace(-1.5, 1.5, 7):
            for ty in np.linspace(-1.5, 1.5, 7):
                L = calc_L([theta, tx, ty])
                if L < best_L:
                    best_L = L
                    best_params = [theta, tx, ty]
    
    # Fine search around best
    for theta in np.linspace(best_params[0] - 5, best_params[0] + 5, 50):
        for tx in np.linspace(best_params[1] - 0.5, best_params[1] + 0.5, 20):
            for ty in np.linspace(best_params[2] - 0.5, best_params[2] + 0.5, 20):
                L = calc_L([theta, tx, ty])
                if L < best_L:
                    best_L = L
                    best_params = [theta, tx, ty]
    
    # Final refinement with Nelder-Mead
    result = minimize(calc_L, best_params, method='Nelder-Mead', 
                      options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 2000})
    
    if result.fun < best_L:
        best_L = result.fun
        best_params = result.x
    
    best_theta, best_tx, best_ty = best_params
    
    # Construct output
    rot_rad = np.deg2rad(best_theta)
    c, s = np.cos(rot_rad), np.sin(rot_rad)
    R = np.array([[c, -s], [s, c]])
    
    inner = []
    for c_center in centers:
        rc = R @ c_center
        rc[0] += best_tx
        rc[1] += best_ty
        inner.append([rc[0], rc[1], best_theta])
        
    return inner, (best_tx, best_ty), best_L, 0.0
