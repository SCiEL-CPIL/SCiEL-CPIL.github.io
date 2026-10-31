import numpy as np
from scipy.optimize import minimize_scalar
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    def get_hex_vertices(cx, cy):
        # Pointy-top vertices
        angles = np.deg2rad([90, 30, -30, -90, -150, 150])
        xs = cx + np.cos(angles)
        ys = cy + np.sin(angles)
        return np.column_stack((xs, ys))
    
    def axial_to_cartesian(q, r):
        return np.sqrt(3) * (q + r/2.0), 1.5 * r

    centers1 = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1), 
                (2,0), (0,2), (-2,0), (0,-2)]
    
    centers2 = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
                (2,-1), (1,-2), (-1,2), (-2,1)]
    
    centers3 = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
                (2,0), (-2,0), (1,1), (-1,-1)]

    clusters = [centers1, centers2, centers3]
    
    best_L = float('inf')
    best_theta = 0.0
    best_centers = centers1
    
    for centers in clusters:
        cart_centers = [axial_to_cartesian(q, r) for q, r in centers]
        
        all_vertices = []
        for cx, cy in cart_centers:
            verts = get_hex_vertices(cx, cy)
            all_vertices.append(verts)
        
        V = np.vstack(all_vertices)
        
        def objective(theta):
            angles = theta + np.pi/3 * np.arange(6)
            normals = np.column_stack((np.cos(angles), np.sin(angles)))
            projections = V @ normals.T
            max_proj = np.max(projections)
            L = (2.0 / np.sqrt(3.0)) * max_proj
            return L
        
        try:
            res = minimize_scalar(objective, bounds=(0, np.pi/3), method='bounded')
            if res.success:
                if res.fun < best_L:
                    best_L = res.fun
                    best_theta = res.x
                    best_centers = centers
        except Exception:
            pass

    if time.time() > deadline:
         pass
    
    inner = []
    for q, r in best_centers:
        cx, cy = axial_to_cartesian(q, r)
        inner.append([cx, cy, 0.0])
        
    return inner, (0.0, 0.0), best_L, best_theta
# EVOLVE-BLOCK-END
