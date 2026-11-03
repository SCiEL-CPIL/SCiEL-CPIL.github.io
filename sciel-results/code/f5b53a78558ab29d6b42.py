"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    sqrt3 = np.sqrt(3)
    
    def get_hex_vertices(cx, cy, side=1.0, angle_deg=0.0):
        """Get 6 vertices of a regular hexagon (flat-top) centered at (cx,cy)."""
        angle_rad = np.radians(angle_deg)
        vertices = []
        for k in range(6):
            theta = angle_rad + k * np.pi / 3
            vx = cx + side * np.cos(theta)
            vy = cy + side * np.sin(theta)
            vertices.append([vx, vy])
        return np.array(vertices)
    
    def enclosing_hexagon_side(points, angle_deg=0.0):
        """Given points and hexagon rotation, find minimal enclosing hexagon.
        Returns (cx, cy, side).
        For fixed rotation, this is: minimize s s.t. exists (cx,cy) with
        for all k in 0..5: for all p: p·n_k - c·n_k <= s*sqrt(3)/2
        where n_k = (cos(a+k*60°), sin(a+k*60°))
        """
        points = np.array(points)
        a_rad = np.radians(angle_deg)
        normals = np.array([[np.cos(a_rad + k * np.pi / 3), 
                             np.sin(a_rad + k * np.pi / 3)] for k in range(6)])
        
        # For each normal n_k, we need: max_p(p·n_k) - c·n_k <= s*sqrt(3)/2
        # and: c·n_k - min_p(p·n_k) <= s*sqrt(3)/2
        # So: max_p(p·n_k) - min_p(p·n_k) <= 2*s*sqrt(3)/2 = s*sqrt(3)
        # Thus s >= (max_p(p·n_k) - min_p(p·n_k)) / sqrt(3) for all k
        # And c·n_k = (max_p(p·n_k) + min_p(p·n_k)) / 2 for the tightest fit
        
        # The minimal s for this rotation is max_k (range_k / sqrt(3))
        # The center c must satisfy c·n_k = mid_k for all k where range_k is maximal
        # For simplicity, use the centroid
        
        projections = points @ normals.T  # (6, n)
        ranges = np.max(projections, axis=1) - np.min(projections, axis=1)
        mids = (np.max(projections, axis=1) + np.min(projections, axis=1)) / 2
        
        side = np.max(ranges) / sqrt3
        
        # Find center: solve c·n_k = mid_k for all k (least squares)
        # normals shape (6,2), mid shape (6,)
        # c = (N^T N)^{-1} N^T mid
        try:
            cx, cy = np.linalg.lstsq(normals, mids, rcond=None)[0]
        except:
            cx, cy = np.mean(points[:, 0]), np.mean(points[:, 1])
        
        return cx, cy, side
    
    def objective(params):
        """Objective: side length of enclosing hexagon.
        params: [x0,y0,x1,y1,...,x11,y11, angle_deg]
        """
        n_params = 2 * N + 1
        if len(params) != n_params:
            return 1e6
        
        centers = params[:2*N].reshape(N, 2)
        angle_deg = params[2*N]
        
        # Get all vertices
        all_verts = []
        for i in range(N):
            verts = get_hex_vertices(centers[i, 0], centers[i, 1], 1.0, angle_deg)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # Check for overlaps (simple distance check between centers)
        # Two unit hexagons overlap if center distance < sqrt(3) (flat-to-flat)
        # Actually, for flat-top hexagons, the minimum distance to avoid overlap
        # depends on relative rotation. For same rotation, it's sqrt(3).
        # For simplicity, assume all same rotation, check distance > sqrt(3)
        
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < sqrt3 - 1e-6:
                    return 1e6 + (sqrt3 - dist) * 100
        
        # Compute enclosing hexagon
        cx, cy, side = enclosing_hexagon_side(all_verts, angle_deg)
        
        return side
    
    def make_honeycomb_initial():
        """Create initial honeycomb arrangement of 12 hexagons."""
        centers = []
        # 3 rows of 4, pointy-top style for easier row packing
        # Flat-top hexagons: horizontal spacing = sqrt(3), vertical = 1.5, offset = sqrt(3)/2
        for row in range(3):
            y = row * 1.5
            offset = sqrt3 / 2 if (row % 2 == 1) else 0
            for col in range(4):
                x = offset + col * sqrt3
                centers.append([x, y])
        return np.array(centers)
    
    # Initial configuration
    init_centers = make_honeycomb_initial()
    init_params = np.concatenate([init_centers.flatten(), [0.0]])
    
    # Optimize
    best_params = init_params
    best_val = objective(init_params)
    
    # Try local search with Nelder-Mead
    if time.time() < deadline - 10:
        try:
            result = minimize(objective, init_params, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except:
            pass
    
    # Extract final configuration
    final_centers = best_params[:2*N].reshape(N, 2)
    final_angle = best_params[2*N]
    
    # Get all vertices for final enclosing hexagon
    all_verts = []
    for i in range(N):
        verts = get_hex_vertices(final_centers[i, 0], final_centers[i, 1], 1.0, final_angle)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    cx, cy, side = enclosing_hexagon_side(all_verts, final_angle)
    
    inner = [[final_centers[i, 0], final_centers[i, 1], final_angle] for i in range(N)]
    
    return inner, (cx, cy), side, final_angle
# EVOLVE-BLOCK-END

