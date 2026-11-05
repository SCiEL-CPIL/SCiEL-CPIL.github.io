import numpy as np
import time
from scipy.optimize import minimize

N = 11
DEADLINE = 140.0  # Safety margin before 150s kill

def _hex_vertices(cx, cy, angle, s=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with rotation angle (radians)."""
    angles = angle + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _get_all_vertices(params):
    """
    params: [cx0, cy0, a0, cx1, cy1, a1, ..., cx10, cy10, a10]
    Returns: (N, 6, 2) array of vertices
    """
    n = len(params) // 3
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        cx = params[3 * i]
        cy = params[3 * i + 1]
        angle = params[3 * i + 2]
        verts[i] = _hex_vertices(cx, cy, angle)
    return verts

def _min_dist_between_hexagons(v1, v2):
    """Compute minimum distance between vertices of two hexagons."""
    diff = v1[:, np.newaxis, :] - v2[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    return np.min(dists)

def _overlap_penalty(verts, threshold=0.01, softness=0.05):
    """
    Compute a smooth overlap penalty based on minimum distance between hexagons.
    """
    n = len(verts)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            c1 = verts[i].mean(axis=0)
            c2 = verts[j].mean(axis=0)
            center_dist = np.linalg.norm(c1 - c2)
            if center_dist < 2.5:
                min_d = _min_dist_between_hexagons(verts[i], verts[j])
                if min_d < threshold:
                    penalty += ((threshold - min_d) ** 2) / softness
    return penalty

def _objective(params):
    """
    Objective function to minimize.
    params: [cx0, cy0, a0, ..., cx10, cy10, a10, outer_angle]
    """
    if time.time() - START_TIME > DEADLINE:
        return float('inf')
        
    inner_params = params[:-1]
    outer_angle = params[-1]
    
    verts = _get_all_vertices(inner_params)
    
    # Flatten vertices to (N*6, 2)
    flat_verts = verts.reshape(-1, 2)
    
    # Normals for the outer hexagon facets
    # Vertices of unit hex are at 0, 60, ...
    # Facets are perpendicular to normals at 30, 90, 150, ...
    theta_n = np.pi / 6 + np.arange(6) * np.pi / 3 + outer_angle
    nx = np.cos(theta_n)
    ny = np.sin(theta_n)
    
    # Project all vertices onto normals
    # (N*6, 1) * (1, 6) -> (N*6, 6)
    proj_x = flat_verts[:, 0:1] * nx[np.newaxis, :]
    proj_y = flat_verts[:, 1:2] * ny[np.newaxis, :]
    projections = proj_x + proj_y
    
    max_abs_proj = np.max(np.abs(projections))
    
    # Side length S such that apothem = max_abs_proj
    # Apothem = S * sqrt(3) / 2
    S = max_abs_proj / (np.sqrt(3) / 2)
    
    # Penalty for overlaps
    penalty = _overlap_penalty(verts)
    
    return S + 10.0 * penalty

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # --- Analytic Initial Guess: The Flower Pattern ---
    # Center
    centers = [(0.0, 0.0)]
    angles = [0.0]
    
    # Ring 1: 6 hexagons at distance sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        angles.append(0.0) # All aligned
        
    # Ring 2: 4 hexagons at distance 3, in the gaps
    # Gaps are at 30, 90, 150, 210, 270, 330.
    # We pick 4 of them. Due to symmetry of the 6-ring, any 4 will work if we optimize, 
    # but typically we pick 30, 90, 150, 210 or similar.
    # Actually, for N=11, the optimal is often described as 1 + 6 + 4.
    # The 4 are placed in 4 of the 6 gaps.
    gap_angles = [30, 90, 150, 210]
    for deg in gap_angles:
        angle = np.deg2rad(deg)
        r = 3.0
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        angles.append(0.0)
        
    # Construct initial params
    params = np.zeros(3 * N + 1)
    for i in range(N):
        params[3*i] = centers[i][0]
        params[3*i+1] = centers[i][1]
        params[3*i+2] = angles[i]
    params[-1] = 0.0 # Outer angle initial guess
    
    best_params = params.copy()
    best_score = _objective(best_params)
    
    # --- Optimization ---
    # Use Powell method for robustness with non-smooth components
    try:
        res = minimize(_objective, best_params, method='Powell',
                       options={'maxiter': 1000, 'maxfev': 10000, 'xrtol': 1e-9, 'ftol': 1e-12})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass
        
    # Try Nelder-Mead for further polish
    if time.time() - START_TIME < DEADLINE - 10:
        try:
            res2 = minimize(_objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass
            
    # Extract results
    inner = []
    for i in range(N):
        cx = best_params[3 * i]
        cy = best_params[3 * i + 1]
        angle = best_params[3 * i + 2]
        inner.append([cx, cy, np.degrees(angle)])
    
    # The outer hexagon is centered at (0,0)
    # Side length is the optimized score (minus any penalty if we want raw S, but penalty should be 0)
    # Let's re-calculate S cleanly for the final result
    verts = _get_all_vertices(best_params[:-1])
    flat_verts = verts.reshape(-1, 2)
    outer_angle = best_params[-1]
    theta_n = np.pi / 6 + np.arange(6) * np.pi / 3 + outer_angle
    nx = np.cos(theta_n)
    ny = np.sin(theta_n)
    proj_x = flat_verts[:, 0:1] * nx[np.newaxis, :]
    proj_y = flat_verts[:, 1:2] * ny[np.newaxis, :]
    projections = proj_x + proj_y
    max_abs_proj = np.max(np.abs(projections))
    S = max_abs_proj / (np.sqrt(3) / 2)
    
    return inner, (0.0, 0.0), S, np.degrees(outer_angle)

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side length: {side}")
    print(f"Angle: {angle}")
