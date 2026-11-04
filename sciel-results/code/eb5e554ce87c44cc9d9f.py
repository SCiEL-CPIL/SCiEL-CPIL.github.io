import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 11
DEADLINE = 140.0

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _get_cluster_vertices():
    """
    Generate the vertices of the rigid 11-hexagon cluster.
    """
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3)) * np.cos(angle)
        y = (np.sqrt(3)) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0) * np.cos(angle)
        y = (3.0) * np.sin(angle)
        centers.append((x, y))
        
    all_vertices = []
    for x, y in centers:
        vs = _hex_vertices(x, y, 30.0)
        all_vertices.extend(vs)
    return np.array(all_vertices)

def _compute_min_side_for_rotation(theta_deg, hull_vertices):
    """
    Compute min side length of outer hexagon (fixed orientation) 
    to contain the cluster rotated by theta_deg.
    """
    rad = np.deg2rad(theta_deg)
    cos_t = np.cos(rad)
    sin_t = np.sin(rad)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated_vertices = hull_vertices @ R.T
    
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    max_proj = 0.0
    for n_vec in normals:
        projs = np.dot(rotated_vertices, n_vec)
        max_proj = max(max_proj, np.max(projs))
        
    return 2.0 * max_proj / np.sqrt(3.0)

def construct():
    start_time = time.time()
    
    hull_vertices = _get_cluster_vertices()
    
    def obj(theta):
        if time.time() - start_time > DEADLINE - 10:
            return float('inf')
        return _compute_min_side_for_rotation(theta, hull_vertices)
    
    try:
        res = minimize_scalar(obj, bounds=(0, 60), method='bounded', options={'xatol': 1e-8})
        best_theta = res.x
        best_side = res.fun
    except Exception:
        best_theta = 0.0
        best_side = obj(0.0)
        
    final_outer_angle = 0.0
    final_side = best_side + 1e-9
    
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3)) * np.cos(angle)
        y = (np.sqrt(3)) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0) * np.cos(angle)
        y = (3.0) * np.sin(angle)
        centers.append((x, y))
        
    inner = []
    rad = np.deg2rad(best_theta)
    cos_t = np.cos(rad)
    sin_t = np.sin(rad)
    
    for x, y in centers:
        rx = x * cos_t - y * sin_t
        ry = x * sin_t + y * cos_t
        inner.append([rx, ry, 30.0 + best_theta])
        
    return inner, (0.0, 0.0), final_side, final_outer_angle

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
