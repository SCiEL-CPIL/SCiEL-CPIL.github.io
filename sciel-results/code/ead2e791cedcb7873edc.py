# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _generate_rigid_clusters():
    """
    Generate a list of rigid clusters (arrays of shape (12, 3) with x, y, angle).
    These are compact, non-overlapping arrangements.
    """
    clusters = []
    
    L = np.sqrt(3)
    
    # Template 1: 1-6-5 (Center + 6 + 5 from second ring)
    pts = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = np.radians(60 * k)
        pts.append([L * np.cos(angle), L * np.sin(angle), 0.0])
    
    # Second ring "between" positions (distance L*sqrt(3), angle 30+60k)
    for k in range(5):
        angle = np.radians(60 * k + 30)
        pts.append([L * np.sqrt(3) * np.cos(angle), L * np.sqrt(3) * np.sin(angle), 0.0])
    
    pts = np.array(pts)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    clusters.append(pts)
    
    # Template 2: 2-4-4-2
    pts = []
    for col in range(2):
        x = (col - 0.5) * L
        y = 0
        pts.append([x, y, 0.0])
    for col in range(4):
        x = (col - 1.5) * L
        y = L * np.sqrt(3) / 2
        pts.append([x, y, 0.0])
    for col in range(4):
        x = (col - 1.5) * L
        y = L * np.sqrt(3)
        pts.append([x, y, 0.0])
    for col in range(2):
        x = (col - 0.5) * L
        y = L * np.sqrt(3) * 1.5
        pts.append([x, y, 0.0])
        
    pts = np.array(pts)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    clusters.append(pts)
    
    # Template 3: 3-3-3-3
    pts = []
    for row in range(4):
        for col in range(3):
            x = (col - 1.0) * L
            y = row * L * np.sqrt(3) / 2
            if row % 2 == 1:
                x += L / 2
            pts.append([x, y, 0.0])
            
    pts = np.array(pts)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    clusters.append(pts)
    
    # Template 4: 2-3-4-3
    pts = []
    # Row 0: 2
    for col in range(2):
        x = (col - 0.5) * L
        y = 0
        pts.append([x, y, 0.0])
    # Row 1: 3
    for col in range(3):
        x = (col - 1.0) * L
        y = L * np.sqrt(3) / 2
        if True: # offset for triangular lattice
             x += L/2 if (1 % 2 == 1) else 0
        pts.append([x, y, 0.0])
    # Row 2: 4
    for col in range(4):
        x = (col - 1.5) * L
        y = L * np.sqrt(3)
        pts.append([x, y, 0.0])
    # Row 3: 3
    for col in range(3):
        x = (col - 1.0) * L
        y = L * np.sqrt(3) * 1.5
        if True:
             x += L/2 if (3 % 2 == 1) else 0
        pts.append([x, y, 0.0])
        
    pts = np.array(pts)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    clusters.append(pts)
    
    # Template 5: Rotated 1-6-5
    pts = clusters[0].copy()
    angle_rot = np.radians(30)
    cos_a = np.cos(angle_rot)
    sin_a = np.sin(angle_rot)
    new_x = pts[:, 0] * cos_a - pts[:, 1] * sin_a
    new_y = pts[:, 0] * sin_a + pts[:, 1] * cos_a
    pts[:, 0] = new_x
    pts[:, 1] = new_y
    pts[:, 2] += 30
    clusters.append(pts)

    return clusters

def _compute_min_s_for_cluster(cluster_verts, outer_angle_deg):
    """
    cluster_verts: (72, 2) array of all vertices of the 12 hexagons.
    outer_angle_deg: angle of the outer hexagon.
    Returns min side length s.
    """
    a_rad = np.radians(outer_angle_deg)
    normal_angles = a_rad + np.radians(30) + np.radians(60) * np.arange(6)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    projs = cluster_verts @ normals.T
    max_projs = np.max(projs, axis=0)
    
    s = 2.0 * np.max(max_projs) / np.sqrt(3)
    
    return s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 615.0
    
    clusters = _generate_rigid_clusters()
    
    best_s = float('inf')
    best_inner = None
    best_outer_angle = 0.0
    
    # Phase 1: Optimize global rotation for each rigid cluster
    for cluster in clusters:
        all_verts = np.zeros((72, 2))
        for i in range(12):
            cx, cy, ang = cluster[i]
            verts = _hex_vertices(cx, cy, 1.0, ang)
            all_verts[i*6:(i+1)*6] = verts
        
        def obj(angle_deg):
            return _compute_min_s_for_cluster(all_verts, angle_deg)
            
        res = minimize_scalar(obj, bounds=(0, 30), method='bounded', options={'xatol': 1e-10})
        
        if res.fun < best_s:
            best_s = res.fun
            best_outer_angle = res.x
            best_inner = cluster.copy()
            
    # Phase 2: Local refinement
    if best_inner is not None:
        def objective(params):
            inner = params[:36].reshape(12, 3).copy()
            outer_angle = params[36]
            
            centers = inner[:, :2]
            valid_internal = True
            for i in range(12):
                for j in range(i+1, 12):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if dist < np.sqrt(3) - 1e-4:
                        valid_internal = False
                        break
                if not valid_internal:
                    break
            
            if not valid_internal:
                return 1e10 + np.sum(np.abs(params))
                
            all_verts = np.zeros((72, 2))
            for i in range(12):
                cx, cy, ang = inner[i]
                verts = _hex_vertices(cx, cy, 1.0, ang)
                all_verts[i*6:(i+1)*6] = verts
                
            s = _compute_min_s_for_cluster(all_verts, outer_angle)
            return s
            
        x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < best_s:
                best_s = res.fun
                best_inner = res.x[:36].reshape(12, 3).copy()
                best_outer_angle = res.x[36]
        except Exception:
            pass
            
    outer = np.array([0.0, 0.0, best_outer_angle])
    
    return best_inner, outer, float(best_s)

# EVOLVE-BLOCK-END
