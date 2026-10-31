# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _min_required_side_rigid(offsets, tx, ty, t_rot_deg):
    """
    Calculate the minimum side length of an axis-aligned hexagon (centered at 0,0)
    that contains the rigid cluster defined by 'offsets' (axial coords) translated by (tx,ty) and rotated by t_rot_deg.
    Offsets are in units of hex side length (1).
    Axial to Cartesian: x = 1.5*q, y = sqrt(3)*(r + q/2)
    """
    sqrt3 = np.sqrt(3)
    centers = []
    for q, r in offsets:
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        centers.append([x, y])
    centers = np.array(centers)
    
    # Apply rotation
    t_rot = np.radians(t_rot_deg)
    cos_r, sin_r = np.cos(t_rot), np.sin(t_rot)
    rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    rotated_centers = centers @ rot_mat.T
    
    # Apply translation
    translated_centers = rotated_centers + np.array([tx, ty])
    
    # Get all vertices of all hexagons
    all_vertices = []
    for cx, cy in translated_centers:
        # Angle of each hexagon is the same as the cluster rotation
        verts = _hex_vertices(cx, cy, 1.0, t_rot_deg)
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)
    
    # Calculate max projection onto 6 normals of outer hexagon
    # Outer hexagon is centered at (0,0), rotation 0.
    # Normals are at angles pi/6, 3pi/6, ...
    thetas = [(2 * k + 1) * np.pi / 6 for k in range(6)]
    max_proj = 0.0
    for vx, vy in all_vertices:
        for theta in thetas:
            proj = vx * np.cos(theta) + vy * np.sin(theta)
            if proj > max_proj:
                max_proj = proj
                
    side = max_proj / (np.sqrt(3) / 2)
    return side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Define candidate rigid shapes (axial coordinates q, r)
    # Base: Center + Ring 1 (7 hexagons)
    base = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    
    # Ring 2 coordinates (12 hexagons)
    ring2 = [
        (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), 
        (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
    ]
    
    # We need 4 more hexagons from Ring 2 to make 11.
    
    # Candidate 1: 4 contiguous (indices 0,1,2,3)
    cand1 = base + [ring2[0], ring2[1], ring2[2], ring2[3]]
    
    # Candidate 2: 4 symmetric (indices 0, 3, 6, 9 - cardinal)
    cand2 = base + [ring2[0], ring2[3], ring2[6], ring2[9]]
    
    # Candidate 3: 4 alternating (indices 0, 2, 4, 6)
    cand3 = base + [ring2[0], ring2[2], ring2[4], ring2[6]]
    
    # Candidate 4: 3 contiguous + 1 opposite (indices 0,1,2 and 6)
    cand4 = base + [ring2[0], ring2[1], ring2[2], ring2[6]]
    
    # Candidate 5: 2 contiguous + 2 opposite (indices 0,1 and 6,7)
    cand5 = base + [ring2[0], ring2[1], ring2[6], ring2[7]]
    
    candidates = [cand1, cand2, cand3, cand4, cand5]
    
    best_side = float('inf')
    best_params = None
    best_candidate_idx = 0
    
    for idx, cand in enumerate(candidates):
        if time.time() > deadline - 20:
            break
            
        # Objective function for scipy
        def objective(params):
            tx, ty, t_rot = params
            # t_rot in radians for optimization, convert to deg for _hex_vertices
            side = _min_required_side_rigid(cand, tx, ty, np.degrees(t_rot))
            return side
        
        # Initial guess: center at (0,0), rotation 0
        x0 = [0.0, 0.0, 0.0]
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
                best_candidate_idx = idx
        except Exception:
            continue
            
    # Fallback if optimization failed
    if best_params is None:
        best_params = [0.0, 0.0, 0.0]
        best_candidate_idx = 0
        best_side = _min_required_side_rigid(candidates[0], 0.0, 0.0, 0.0)
    
    # Extract best configuration
    tx, ty, t_rot_rad = best_params
    t_rot_deg = np.degrees(t_rot_rad)
    
    # Generate final inner positions
    offsets = candidates[best_candidate_idx]
    sqrt3 = np.sqrt(3)
    inner = []
    for q, r in offsets:
        x_local = 1.5 * q
        y_local = sqrt3 * (r + q / 2.0)
        
        # Rotate
        cos_r, sin_r = np.cos(t_rot_rad), np.sin(t_rot_rad)
        x_rot = x_local * cos_r - y_local * sin_r
        y_rot = x_local * sin_r + y_local * cos_r
        
        # Translate
        x_final = x_rot + tx
        y_final = y_rot + ty
        
        inner.append([x_final, y_final, t_rot_deg])
        
    # Ensure side is strictly larger than required
    best_side *= (1.0 + 1e-5)
    
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END
