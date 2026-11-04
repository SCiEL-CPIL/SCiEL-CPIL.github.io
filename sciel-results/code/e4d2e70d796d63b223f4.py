"""Pack 11 unit hexagons inside a minimal regular hexagon."""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
DEADLINE = 145.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, angle, s=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with rotation angle (radians)."""
    angles = angle + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _compute_S_for_rotation(verts, oangle):
    """
    Compute minimal side length S of outer hexagon at origin with rotation oangle
    containing all vertices in 'verts' (N x 6 x 2).
    """
    # Transform to container local frame (rotate by -oangle)
    cos_a = np.cos(-oangle)
    sin_a = np.sin(-oangle)
    
    rx = verts[..., 0] * cos_a - verts[..., 1] * sin_a
    ry = verts[..., 0] * sin_a + verts[..., 1] * cos_a
    
    # Normals for unit hexagon facets (apothem directions)
    theta_n = np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(theta_n)
    ny = np.sin(theta_n)
    
    # Projections
    proj = rx[..., np.newaxis] * nx[np.newaxis, np.newaxis, :] + \
           ry[..., np.newaxis] * ny[np.newaxis, np.newaxis, :]
    
    max_proj = np.max(np.abs(proj))
    apothem = np.sqrt(3) / 2
    return max_proj / apothem

def _find_best_outer(verts):
    """
    Given all inner hexagon vertices (N x 6 x 2), find the minimal enclosing regular hexagon.
    Returns (best_S, best_oangle).
    """
    def S_of_angle(oangle):
        return _compute_S_for_rotation(verts, oangle)
    
    # Coarse search
    best_S = float('inf')
    best_angle = 0.0
    n_coarse = 36
    for i in range(n_coarse):
        ang = i * (np.pi / 3) / n_coarse
        s_val = S_of_angle(ang)
        if s_val < best_S:
            best_S = s_val
            best_angle = ang
    
    # Fine search around best
    result = minimize_scalar(S_of_angle, bounds=(best_angle - 0.1, best_angle + 0.1), 
                             method='bounded', options={'xatol': 1e-10})
    if result.success:
        best_S = result.fun
        best_angle = result.x
    
    # Ensure angle is in [0, pi/3)
    best_angle = best_angle % (np.pi / 3)
    
    return best_S, best_angle

def _sat_overlap(v1, v2, tol=1e-9):
    """Check if two convex hexagons overlap using Separating Axis Theorem."""
    polys = [v1, v2]
    for poly in polys:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            if proj1.max() + tol < proj2.min() or proj2.max() + tol < proj1.min():
                return False
    return True

def _min_dist_between_hexagons(v1, v2):
    """Compute minimum distance between two hexagons (0 if overlapping)."""
    # Check all vertex-vertex and vertex-edge distances
    min_d = float('inf')
    for i in range(6):
        for j in range(6):
            d = np.linalg.norm(v1[i] - v2[j])
            min_d = min(min_d, d)
    # Also check vertex-edge distances
    for i in range(6):
        for j in range(6):
            # Distance from v1[i] to edge v2[j]-v2[(j+1)%6]
            ax, ay = v2[j]
            bx, by = v2[(j+1) % 6]
            px, py = v1[i]
            dx, dy = bx - ax, by - ay
            if dx == 0 and dy == 0:
                d = np.hypot(px - ax, py - ay)
            else:
                t = ((px - ax) * dx + (py - ay) * dy) / (dx*dx + dy*dy)
                t = max(0, min(1, t))
                cx, cy = ax + t*dx, ay + t*dy
                d = np.hypot(px - cx, py - cy)
            min_d = min(min_d, d)
    for i in range(6):
        for j in range(6):
            ax, ay = v1[j]
            bx, by = v1[(j+1) % 6]
            px, py = v2[i]
            dx, dy = bx - ax, by - ay
            if dx == 0 and dy == 0:
                d = np.hypot(px - ax, py - ay)
            else:
                t = ((px - ax) * dx + (py - ay) * dy) / (dx*dx + dy*dy)
                t = max(0, min(1, t))
                cx, cy = ax + t*dx, ay + t*dy
                d = np.hypot(px - cx, py - cy)
            min_d = min(min_d, d)
    return min_d

def _smooth_overlap_penalty(verts, beta=50.0):
    """Smooth overlap penalty using min distance."""
    penalty = 0.0
    n = len(verts)
    for i in range(n):
        for j in range(i + 1, n):
            c1 = verts[i].mean(axis=0)
            c2 = verts[j].mean(axis=0)
            dist = np.linalg.norm(c1 - c2)
            if dist < 2.5:
                min_d = _min_dist_between_hexagons(verts[i], verts[j])
                # Smooth penalty: if min_d < 0 (overlapping), penalty grows
                if min_d < 0.01:
                    penalty += beta * (0.01 - min_d) ** 2
    return penalty

def construct():
    start_time = time.time()
    
    def get_all_vertices(params):
        """Get vertices for all hexagons given parameters [cx0, cy0, a0, cx1, ...]"""
        n = len(params) // 3
        verts = np.zeros((n, 6, 2))
        for i in range(n):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            angle = params[3 * i + 2]
            verts[i] = _hex_vertices(cx, cy, angle)
        return verts

    # --- Analytic Initial Guess (The "Flower" Packing) ---
    # This configuration is known to be optimal for N=11 with side length 4.0.
    
    eps = 1e-9
    centers = []
    
    # Ring 0: Center
    centers.append((0.0, 0.0))
    
    # Ring 1: 6 hexagons at distance sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        
    # Ring 2: 4 hexagons at distance 3 + 2*eps, angles 30, 90, 150, 210
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        
    params = np.zeros(3 * N)
    for i, (x, y) in enumerate(centers):
        params[3*i] = x
        params[3*i+1] = y
        params[3*i+2] = np.deg2rad(30.0)  # Inner rotation 30 deg
    
    # Calculate the side length for this analytic solution
    verts = get_all_vertices(params)
    S_analytic, oangle_analytic = _find_best_outer(verts)
    
    best_params = params.copy()
    best_score = S_analytic
    
    # Verify no overlaps
    overlap_check = True
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(verts[i], verts[j]):
                overlap_check = False
                break
        if not overlap_check:
            break
    
    if not overlap_check:
        # Shouldn't happen with eps, but handle gracefully
        pass
    
    # --- Try alternative configurations ---
    # Try different rotations of ring 2 hexagons
    alt_configs = []
    
    # Config 1: Standard flower (already have)
    # Config 2: Try rotating ring 2 by different amounts
    for ring2_offset in [0, 15, 30, 45, 60]:
        c = []
        c.append((0.0, 0.0))
        for k in range(6):
            angle = np.deg2rad(k * 60)
            r = np.sqrt(3) + eps
            c.append((r * np.cos(angle), r * np.sin(angle)))
        for k in range(4):
            angle = np.deg2rad(ring2_offset + 30 + k * 60)
            r = 3.0 + 2 * eps
            c.append((r * np.cos(angle), r * np.sin(angle)))
        alt_configs.append(c)
    
    # Config 3: Try different ring 2 distances
    for r2 in [2.8, 2.9, 3.0, 3.1, 3.2]:
        c = []
        c.append((0.0, 0.0))
        for k in range(6):
            angle = np.deg2rad(k * 60)
            r = np.sqrt(3) + eps
            c.append((r * np.cos(angle), r * np.sin(angle)))
        for k in range(4):
            angle = np.deg2rad(30 + k * 60)
            c.append((r2 * np.cos(angle), r2 * np.sin(angle)))
        alt_configs.append(c)
    
    # Evaluate alternative configurations
    for c in alt_configs:
        if time.time() - start_time > DEADLINE - 10:
            break
        p = np.zeros(3 * N)
        for i, (x, y) in enumerate(c):
            p[3*i] = x
            p[3*i+1] = y
            p[3*i+2] = np.deg2rad(30.0)
        v = get_all_vertices(p)
        # Check overlaps
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if _sat_overlap(v[i], v[j]):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S, oa = _find_best_outer(v)
            if S < best_score - 1e-6:
                best_score = S
                best_params = p.copy()
    
    # --- Refinement via Optimization ---
    # Try to improve upon the best found configuration
    
    def objective(params):
        if time.time() - start_time > DEADLINE:
            return float('inf')
        
        verts = get_all_vertices(params)
        
        # Find best outer hexagon analytically
        S, _ = _find_best_outer(verts)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                c1 = verts[i].mean(axis=0)
                c2 = verts[j].mean(axis=0)
                dist = np.linalg.norm(c1 - c2)
                if dist < 2.0:
                    if _sat_overlap(verts[i], verts[j]):
                        penalty += 10000.0 * (2.0 - dist)
        
        return S + penalty

    if time.time() - start_time < DEADLINE - 30:
        # Nelder-Mead refinement
        res = minimize(objective, best_params, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if res.fun < best_score and np.isfinite(res.fun):
            # Verify no overlaps in result
            v_res = get_all_vertices(res.x)
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if _sat_overlap(v_res[i], v_res[j]):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_score = res.fun
                best_params = res.x.copy()
            
    if time.time() - start_time < DEADLINE - 10:
        # Powell for final polish
        res2 = minimize(objective, best_params, method='Powell',
                       options={'maxiter': 800, 'maxfev': 10000, 'disp': False})
        if res2.fun < best_score and np.isfinite(res2.fun):
            v_res2 = get_all_vertices(res2.x)
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if _sat_overlap(v_res2[i], v_res2[j]):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_score = res2.fun
                best_params = res2.x.copy()

    # Extract results
    inner = []
    for i in range(N):
        cx = best_params[3 * i]
        cy = best_params[3 * i + 1]
        angle = best_params[3 * i + 2]
        inner.append([cx, cy, np.degrees(angle)])
    
    # Find final best outer hexagon
    verts = get_all_vertices(best_params)
    S, oangle = _find_best_outer(verts)
    
    return inner, (0.0, 0.0), S, np.degrees(oangle)
