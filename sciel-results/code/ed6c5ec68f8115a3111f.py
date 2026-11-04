import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + np.cos(theta)
        y = cy + np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def _get_outer_normals(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        theta = angle_rad + np.pi/6 + i * np.pi / 3
        normals.append((np.cos(theta), np.sin(theta)))
    return np.array(normals)

def _min_enclosing_hex_side(composite_verts, outer_angle_deg, outer_center):
    normals = _get_outer_normals(outer_angle_deg)
    rel_verts = composite_verts - outer_center
    dots = rel_verts @ normals.T
    max_proj = np.max(np.abs(dots))
    return 2 * max_proj / np.sqrt(3)

def _compute_composite_verts(centers, angles):
    all_verts = []
    for i, (cx, cy) in enumerate(centers):
        verts = _hex_vertices(cx, cy, angles[i])
        all_verts.append(verts)
    return np.vstack(all_verts)

def _objective(vars, verts):
    outer_cx, outer_cy, outer_angle = vars
    outer_center = np.array([outer_cx, outer_cy])
    return _min_enclosing_hex_side(verts, outer_angle, outer_center)

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    # Define the two candidate configurations for the 11 hexagons
    # All hexagons have the same orientation (0 degrees) for tight packing
    angle = 0.0
    
    configs = []
    
    # Config 1: Outer ring missing adjacent (150, 210 missing)
    # Used: 30, 90, 270, 330
    centers1 = []
    centers1.append([0.0, 0.0])
    for k in range(6):
        a = np.deg2rad(k * 60)
        centers1.append([np.sqrt(3) * np.cos(a), np.sqrt(3) * np.sin(a)])
    for a_deg in [30, 90, 270, 330]:
        a = np.deg2rad(a_deg)
        centers1.append([3.0 * np.cos(a), 3.0 * np.sin(a)])
    configs.append(np.array(centers1))
    
    # Config 2: Outer ring missing separated (150, 330 missing)
    # Used: 30, 90, 210, 270
    centers2 = []
    centers2.append([0.0, 0.0])
    for k in range(6):
        a = np.deg2rad(k * 60)
        centers2.append([np.sqrt(3) * np.cos(a), np.sqrt(3) * np.sin(a)])
    for a_deg in [30, 90, 210, 270]:
        a = np.deg2rad(a_deg)
        centers2.append([3.0 * np.cos(a), 3.0 * np.sin(a)])
    configs.append(np.array(centers2))
    
    best_S = float('inf')
    best_inner = None
    best_center = None
    best_angle = 0.0
    
    for centers in configs:
        if time.time() > DEADLINE - 10.0:
            break
            
        angles = np.full(len(centers), angle)
        verts = _compute_composite_verts(centers, angles)
        
        # Initial guess: center at origin, angle 0
        x0 = np.array([0.0, 0.0, 0.0])
        
        # Coarse search over angle to find good starting point
        best_local_S = float('inf')
        best_local_x = x0
        for a in np.linspace(0, 30, 61):
            x_trial = np.array([0.0, 0.0, a])
            S = _objective(x_trial, verts)
            if S < best_local_S:
                best_local_S = S
                best_local_x = x_trial
        
        # Refine with Nelder-Mead
        try:
            res = minimize(_objective, best_local_x, args=(verts,), method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_local_S:
                best_local_S = res.fun
                best_local_x = res.x
        except Exception:
            pass
        
        if best_local_S < best_S:
            best_S = best_local_S
            best_inner = centers
            best_center = best_local_x[:2]
            best_angle = best_local_x[2]
            
    # Format output
    inner = []
    for i in range(len(best_inner)):
        inner.append([best_inner[i][0], best_inner[i][1], angle])
    
    return inner, tuple(best_center), best_S, best_angle

