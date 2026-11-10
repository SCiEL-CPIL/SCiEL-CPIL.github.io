import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses physics-based relaxation with L-BFGS-B to minimize outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = 595.0  # Safety margin
    
    def get_hex_vertices(cx, cy, side=1.0, angle_deg=30.0):
        """Get vertices of a regular hexagon."""
        angles = np.deg2rad(angle_deg) + np.deg2rad(60 * np.arange(6))
        return np.column_stack([cx + side * np.cos(angles), 
                                cy + side * np.sin(angles)])
    
    def check_separation(v1, v2, tol=1e-6):
        """Check if two hexagons are separated using SAT."""
        # Get edge normals for both hexagons
        for verts in [v1, v2]:
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                normal = normal / np.linalg.norm(normal)
                
                # Project all vertices onto normal
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Check if projections are separated
                if max1 < min2 - tol or max2 < min1 - tol:
                    return False  # Not separated (overlap)
        
        return True  # Separated
    
    def check_inside(verts, outer_verts, tol=1e-6):
        """Check if all vertices of inner hex are inside outer hex."""
        for v in verts:
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                cross = np.cross(p2 - p1, v - p1)
                if cross < -tol:
                    return False
        return True
    
    def objective(params):
        """Objective function: minimize s (outer hex side length)."""
        if time.time() > deadline:
            return 1e10
        
        s = params[0]
        centers = params[1:25].reshape(12, 2)
        
        # Check if s is too small
        if s < np.sqrt(12):
            return 1e10
        
        # Get outer hexagon vertices
        outer_verts = get_hex_vertices(0, 0, s, 30.0)
        
        # Check all inner hexagons
        for i in range(12):
            cx, cy = centers[i]
            inner_verts = get_hex_vertices(cx, cy, 1.0, 30.0)
            
            # Check if inside outer
            if not check_inside(inner_verts, outer_verts):
                # Penalty for being outside
                min_cross = np.inf
                for v in inner_verts:
                    for j in range(6):
                        p1 = outer_verts[j]
                        p2 = outer_verts[(j + 1) % 6]
                        cross = np.cross(p2 - p1, v - p1)
                        min_cross = min(min_cross, cross)
                return 1e6 + min_cross ** 2
        
        # Check separation between all pairs
        for i in range(12):
            for j in range(i + 1, 12):
                v1 = get_hex_vertices(centers[i][0], centers[i][1], 1.0, 30.0)
                v2 = get_hex_vertices(centers[j][0], centers[j][1], 1.0, 30.0)
                
                if not check_separation(v1, v2):
                    # Penalty for overlap
                    return 1e5 + np.sum((v1 - v2[0])**2)
        
        # Valid packing, return s (to minimize)
        return s
    
    # Initial configuration: flower pattern
    S_init = 1.0 + 1e-3
    centers_init = np.array([
        [0.0, 0.0],  # Center
        [np.sqrt(3) * S_init, 0],
        [np.sqrt(3) * S_init * np.cos(np.pi/3), np.sqrt(3) * S_init * np.sin(np.pi/3)],
        [np.sqrt(3) * S_init * np.cos(2*np.pi/3), np.sqrt(3) * S_init * np.sin(2*np.pi/3)],
        [np.sqrt(3) * S_init * np.cos(np.pi), np.sqrt(3) * S_init * np.sin(np.pi)],
        [np.sqrt(3) * S_init * np.cos(4*np.pi/3), np.sqrt(3) * S_init * np.sin(4*np.pi/3)],
        [np.sqrt(3) * S_init * np.cos(5*np.pi/3), np.sqrt(3) * S_init * np.sin(5*np.pi/3)],
        [3.0 * S_init * np.cos(np.pi/6), 3.0 * S_init * np.sin(np.pi/6)],
        [3.0 * S_init * np.cos(np.pi/2), 3.0 * S_init * np.sin(np.pi/2)],
        [3.0 * S_init * np.cos(5*np.pi/6), 3.0 * S_init * np.sin(5*np.pi/6)],
        [3.0 * S_init * np.cos(7*np.pi/6), 3.0 * S_init * np.sin(7*np.pi/6)],
        [3.0 * S_init * np.cos(3*np.pi/2), 3.0 * S_init * np.sin(3*np.pi/2)],
    ])
    
    s_init = 4.0 * S_init * (1.0 + 1e-3)
    
    x0 = np.concatenate([[s_init], centers_init.flatten()])
    
    # Bounds: s >= sqrt(12), centers can be anywhere
    bounds = [(np.sqrt(12), None)] + [(-100, 100)] * 24
    
    # Optimize
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
    
    if time.time() > deadline:
        # Return initial if timeout
        s_final = s_init
        centers_final = centers_init
    else:
        s_final = result.x[0]
        centers_final = result.x[1:].reshape(12, 2)
    
    # Add small margin for numerical safety
    s_final *= (1.0 + 1e-5)
    
    # Scale centers slightly to ensure gaps
    scale = 1.0 + 1e-4
    centers_final *= scale
    
    inner_hex_data = np.column_stack((centers_final, np.full(12, 30.0)))
    outer_hex_data = np.array([0.0, 0.0, 30.0])
    
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END
