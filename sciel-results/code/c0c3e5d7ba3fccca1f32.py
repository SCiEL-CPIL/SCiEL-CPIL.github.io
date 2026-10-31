import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    
    def hex_vertices(center, rotation, size=1.0):
        """Return 6 vertices of a regular hexagon."""
        angles = rotation + np.arange(6) * np.pi / 3
        return center + size * np.column_stack([np.cos(angles), np.sin(angles)])
    
    def point_in_hexagon(p, c, theta, s):
        """Check if point p is inside hexagon with center c, rotation theta, circumradius s."""
        # Edge normals
        angles = theta + np.arange(6) * np.pi / 3
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        apothem = s * np.sqrt(3) / 2
        projections = (p - c) @ normals.T
        return np.all(projections <= apothem + 1e-9)
    
    def hexagon_contains_hexagon(inner_verts, c, theta, s):
        """Check if all inner vertices are inside outer hexagon."""
        for v in inner_verts:
            if not point_in_hexagon(v, c, theta, s):
                return False
        return True
    
    def compute_min_hexagon(points, inner_rotation=0.0):
        """Find minimum enclosing regular hexagon for given inner hexagon centers."""
        # Precompute inner hexagon vertices
        inner_verts_list = [hex_vertices(p, inner_rotation) for p in points]
        
        def objective(params):
            cx, cy, theta = params
            c = np.array([cx, cy])
            
            # Compute required s
            # For each inner vertex, find the maximum projection
            angles = theta + np.arange(6) * np.pi / 3
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            
            max_proj = 0
            for verts in inner_verts_list:
                projections = (verts - c) @ normals.T  # (6, 6)
                max_proj = max(max_proj, np.max(projections))
            
            s = 2.0 / np.sqrt(3.0) * max_proj
            return max(s, 0.0)
        
        centroid = np.mean(points, axis=0)
        best_local = float('inf')
        best_params = None
        
        for theta0 in np.linspace(0, np.pi/3, 7):
            x0 = [centroid[0], centroid[1], theta0]
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
                if res.fun < best_local:
                    best_local = res.fun
                    best_params = res.x
            except:
                pass
            
            if time.time() > deadline:
                break
        
        return best_local, best_params
    
    # Generate candidate arrangements on hex grid
    positions = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = 1.5 * q
            y = np.sqrt(3) * (r + q / 2.0)
            positions.append((x, y))
    positions = np.array(positions)
    
    dists = np.sum(positions**2, axis=1)
    order = np.argsort(dists)
    
    candidate_sets = [positions[order[:11]]]
    
    # Try some shifted versions
    for dx in [-0.5, 0, 0.5]:
        for dy in [-0.5, 0, 0.5]:
            shifted = positions + np.array([dx, dy])
            dists_s = np.sum(shifted**2, axis=1)
            order_s = np.argsort(dists_s)
            candidate_sets.append(shifted[order_s[:11]])
    
    best_side = float('inf')
    best_params = None
    best_points = None
    best_inner_rot = 0.0
    
    for pts in candidate_sets:
        if time.time() > deadline:
            break
        
        for inner_rot in [0.0, np.pi/6]:
            if time.time() > deadline:
                break
            side, params = compute_min_hexagon(pts, inner_rot)
            if side < best_side:
                best_side = side
                best_params = params
                best_points = pts
                best_inner_rot = inner_rot
    
    if best_params is None:
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
    
    cx, cy, theta = best_params
    theta_deg = np.degrees(theta)
    inner_rot_deg = np.degrees(best_inner_rot)
    
    inner = [[float(pts[0]), float(pts[1]), inner_rot_deg] for pts in best_points]
    
    return inner, (float(cx), float(cy)), float(best_side), float(theta_deg)
# EVOLVE-BLOCK-END
