import numpy as np
import time
from scipy.optimize import minimize

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
        theta = angle_rad + i * np.pi / 3 + np.pi / 6 # Normals are offset by 30 deg from vertices
        # Wait, for a hexagon with vertices at theta, normals are at theta + 30.
        # If we define hex by "flat-top" or "pointy-top", we need to be consistent.
        # My previous analysis:
        # Inner hex rot 30: Vertices at 30, 90...
        # Normals at 0, 60, 120... (perpendicular to sides).
        # Let's verify.
        # Side between 30 and 90 deg vertices. Midpoint angle 60. Normal is 60+90=150?
        # No.
        # Let's just use the support function.
        # A regular hexagon of side S, center 0, rotation phi.
        # Vertices at phi + k*60.
        # The distance to the side (apothem) is S*sqrt(3)/2.
        # The normals to the sides are at phi + 30 + k*60.
        pass
    # Correct Normal Angles for Hex with Vertices at phi + k*60:
    # The sides are perpendicular to the radii at the midpoints of the vertices.
    # Midpoint angle: phi + 30 + k*60.
    # So normals are at phi + 30 + k*60.
    normals = []
    for i in range(6):
        theta = angle_rad + np.pi/6 + i * np.pi / 3
        normals.append((np.cos(theta), np.sin(theta)))
    return np.array(normals)

def _min_enclosing_hex_side(inner_verts, outer_angle_deg):
    # inner_verts: list of arrays of vertices (N, 6, 2)
    # outer_angle_deg: rotation of outer hex
    normals = _get_outer_normals(outer_angle_deg)
    # Project all vertices onto all normals
    # We need max |dot(v, n)|
    max_proj = 0.0
    for verts in inner_verts:
        # verts shape (6, 2)
        # dots shape (6, 6)
        dots = np.abs(verts @ normals.T)
        max_proj = max(max_proj, np.max(dots))
    
    # S * sqrt(3)/2 >= max_proj
    # S >= 2 * max_proj / sqrt(3)
    return 2 * max_proj / np.sqrt(3)

def _overlap_penalty(centers, min_dist):
    penalty = 0.0
    n = len(centers)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                penalty += (min_dist - d)**2
    return penalty

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    # Initial configuration
    eps = 1e-9
    centers = np.array([
        [0.0, 0.0],
        [np.sqrt(3)+eps, 0],
        [np.sqrt(3)*0.5+eps, np.sqrt(3)*0.866+eps], # approx
        # ... let's just use the loop from before
    ])
    
    # Re-initialize centers properly
    centers_list = []
    centers_list.append([0.0, 0.0])
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        
    centers = np.array(centers_list)
    angles = np.array([30.0] * 11)
    outer_angle = 30.0
    
    # Prepare for optimization
    # Variables: x1..x11, y1..y11, outer_angle
    x0 = np.concatenate([centers.flatten(), [outer_angle]])
    
    def objective(vars):
        if time.time() > DEADLINE:
            return 1e9
        c = vars[:22].reshape(11, 2)
        o_ang = vars[22]
        
        # Check overlap
        pen = _overlap_penalty(c, np.sqrt(3))
        if pen > 1e-6:
            return 1e6 + pen
            
        # Compute vertices
        verts_list = []
        for i in range(11):
            verts = _hex_vertices(c[i][0], c[i][1], angles[i])
            verts_list.append(verts)
            
        S = _min_enclosing_hex_side(verts_list, o_ang)
        return S
        
    # Run Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.success or res.fun < 4.0:
            best_vars = res.x
            # Extract best solution
            best_centers = best_vars[:22].reshape(11, 2)
            best_outer_angle = best_vars[22]
            
            inner = []
            for i in range(11):
                inner.append([best_centers[i][0], best_centers[i][1], angles[i]])
                
            # Calculate final side
            verts_list = []
            for i in range(11):
                verts = _hex_vertices(best_centers[i][0], best_centers[i][1], angles[i])
                verts_list.append(verts)
            side = _min_enclosing_hex_side(verts_list, best_outer_angle)
            
            return inner, (0.0, 0.0), side, best_outer_angle
    except Exception:
        pass
        
    # Fallback to current
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    # Calculate side for fallback
    verts_list = []
    for x, y in centers:
        verts = _hex_vertices(x, y, 30.0)
        verts_list.append(verts)
    side = _min_enclosing_hex_side(verts_list, 30.0)
    return inner, (0.0, 0.0), side, 30.0
