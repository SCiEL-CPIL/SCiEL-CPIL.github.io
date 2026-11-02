import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: simple grid
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 3.0, 0.0

    # Initial positions for 11 hexagons
    # Center at (0,0)
    # 6 neighbors at distance sqrt(3)
    # 4 more at distance 2*sqrt(3) or similar
    
    def get_initial_positions():
        pos = []
        pos.append((0.0, 0.0))
        for i in range(6):
            angle = i * math.pi / 3
            pos.append((math.sqrt(3) * math.cos(angle), math.sqrt(3) * math.sin(angle)))
        # Add 4 more. Let's place them at (2*sqrt(3), 0), (-2*sqrt(3), 0), etc?
        # Or in the gaps.
        # Let's try a hexagonal number pattern.
        # 1 in center, 6 in ring 1, 12 in ring 2. We need 4 from ring 2.
        # Ring 2 positions are at distance 2*sqrt(3) from center, at angles 0, 30, 60...
        # Actually, the distance to the center of a hexagon in ring 2 is 2*sqrt(3) if it's aligned.
        # Let's just place them far apart initially to avoid overlap penalties.
        for i in range(4):
            angle = (i * math.pi / 2) + math.pi / 6
            pos.append((3.5 * math.cos(angle), 3.5 * math.sin(angle)))
        return np.array(pos)

    initial_pos = get_initial_positions()
    initial_theta = 0.0
    initial_phi = 0.0
    
    # Parameters: 11 x-coords, 11 y-coords, theta, phi
    x0 = np.concatenate([initial_pos[:, 0], initial_pos[:, 1], [initial_theta, initial_phi]])
    
    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10
            
        pos = params[:22].reshape(11, 2)
        theta = params[22]
        phi = params[23]
        
        # Check overlap: distance between centers must be >= sqrt(3)
        # This is valid for parallel hexagons.
        for i in range(11):
            for j in range(i + 1, 11):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < math.sqrt(3) - 1e-5:
                    # Penalty proportional to overlap
                    return 1e10 + (math.sqrt(3) - dist)
        
        # Calculate vertices of all inner hexagons
        # Hexagon vertices relative to center:
        # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        base_verts = np.array([
            [1.0, 0.0],
            [0.5, math.sqrt(3)/2],
            [-0.5, math.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -math.sqrt(3)/2],
            [0.5, -math.sqrt(3)/2]
        ])
        
        # Rotation matrix for theta
        c_t = math.cos(theta)
        s_t = math.sin(theta)
        rot_t = np.array([[c_t, -s_t], [s_t, c_t]])
        rotated_verts = base_verts @ rot_t.T
        
        # All vertices in world coordinates
        all_verts = pos[:, np.newaxis, :] + rotated_verts[np.newaxis, :, :]
        
        # Centroid of inner hexagons
        center = np.mean(pos, axis=0)
        
        # Normals of outer hexagon
        # Outer hexagon side S, apothem S*sqrt(3)/2
        # Normals are at angles phi + pi/6 + k*pi/3
        normals = []
        for k in range(6):
            angle = phi + math.pi / 6.0 + k * math.pi / 3.0
            normals.append([math.cos(angle), math.sin(angle)])
        normals = np.array(normals)
        
        # Calculate max projection
        # For each vertex, project onto each normal
        # verts: (11, 6, 2)
        # normals: (6, 2)
        # We want max | (v - c) . n |
        
        # Subtract center
        all_verts_centered = all_verts - center[np.newaxis, np.newaxis, :]
        
        # Flatten vertices
        verts_flat = all_verts_centered.reshape(-1, 2) # (66, 2)
        
        # Project
        projs = verts_flat @ normals.T # (66, 6)
        
        max_proj = np.max(np.abs(projs))
        
        # S * sqrt(3)/2 >= max_proj
        # S >= max_proj * 2 / sqrt(3)
        S = max_proj * 2.0 / math.sqrt(3)
        
        return S

    best_S = float('inf')
    best_params = x0
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_S:
            best_S = res.fun
            best_params = res.x
    except Exception:
        pass

    # Try another start point if time permits
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        # Perturb initial positions
        np.random.seed(42)
        x0_2 = x0 + np.random.normal(0, 0.5, len(x0))
        x0_2[22] = 0.0
        x0_2[23] = 0.0
        try:
            res2 = minimize(objective, x0_2, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res2.fun < best_S:
                best_S = res2.fun
                best_params = res2.x
        except Exception:
            pass

    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        # Try Powell method
        try:
            res3 = minimize(objective, best_params, method='Powell', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res3.fun < best_S:
                best_S = res3.fun
                best_params = res3.x
        except Exception:
            pass

    # Extract results
    pos = best_params[:22].reshape(11, 2)
    theta = best_params[22]
    phi = best_params[23]
    
    # Recalculate S and center
    center = np.mean(pos, axis=0)
    
    # Verify S
    base_verts = np.array([
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])
    c_t = math.cos(theta)
    s_t = math.sin(theta)
    rot_t = np.array([[c_t, -s_t], [s_t, c_t]])
    rotated_verts = base_verts @ rot_t.T
    all_verts = pos[:, np.newaxis, :] + rotated_verts[np.newaxis, :, :]
    all_verts_centered = all_verts - center[np.newaxis, np.newaxis, :]
    verts_flat = all_verts_centered.reshape(-1, 2)
    
    normals = []
    for k in range(6):
        angle = phi + math.pi / 6.0 + k * math.pi / 3.0
        normals.append([math.cos(angle), math.sin(angle)])
    normals = np.array(normals)
    
    projs = verts_flat @ normals.T
    max_proj = np.max(np.abs(projs))
    S = max_proj * 2.0 / math.sqrt(3)
    
    # Add small safety margin
    S *= 1.0001
    
    inner = []
    for x, y in pos:
        inner.append([float(x), float(y), float(math.degrees(theta))])
        
    center_out = [float(center[0]), float(center[1])]
    side = float(S)
    rotation = float(math.degrees(phi))
    
    return inner, center_out, side, rotation

