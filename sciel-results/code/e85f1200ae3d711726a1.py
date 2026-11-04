"""Hexagon packing: 11 unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
DEADLINE = 140.0  # Safety margin from 150s limit

# EVOLVE-BLOCK-START

def _hex_vertices(angle_rad=0.0, center=None):
    """Return 6 vertices of a unit regular hexagon rotated by angle_rad, optionally translated by center."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = math.cos(theta)
        y = math.sin(theta)
        if center is not None:
            x += center[0]
            y += center[1]
        verts.append([x, y])
    return np.array(verts)

def _outer_hexagon_vertices(side_length, angle_rad=0.0, center=None):
    """Return 6 vertices of a regular hexagon with given side length."""
    R = side_length
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = R * math.cos(theta)
        y = R * math.sin(theta)
        if center is not None:
            x += center[0]
            y += center[1]
        verts.append([x, y])
    return np.array(verts)

def _compute_min_side_length(centers, angles, outer_angle_rad=0.0):
    """
    Compute the minimum side length S such that all inner hexagons fit inside
    the outer hexagon centered at origin with rotation outer_angle_rad.
    """
    outer_normals = np.array([
        [math.cos(outer_angle_rad + k * math.pi / 3.0), 
         math.sin(outer_angle_rad + k * math.pi / 3.0)]
        for k in range(6)
    ])
    
    max_abs_proj = 0.0
    for i in range(N):
        verts = _hex_vertices(angles[i], centers[i])
        for k in range(6):
            n = outer_normals[k]
            projs = verts @ n
            max_abs_proj = max(max_abs_proj, np.max(np.abs(projs)))
    
    S = max_abs_proj * 2.0 / math.sqrt(3.0)
    return S

def _point_in_polygon(p, poly):
    """Check if point p is inside convex polygon poly (6,2 array)."""
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
        if cross < -1e-10:
            return False
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between segments p1-p2 and p3-p4."""
    # Check if segments intersect
    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    
    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return 0.0
    
    # Check point-segment distances
    def pt_seg_dist(p, a, b):
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = max(0, min(1, t))
        proj = a + t * ab
        return np.linalg.norm(p - proj)
    
    d = min(pt_seg_dist(p1, p3, p4), pt_seg_dist(p2, p3, p4),
            pt_seg_dist(p3, p1, p2), pt_seg_dist(p4, p1, p2))
    return d

def _poly_separation(poly1, poly2):
    """
    Compute minimum distance between two convex polygons.
    Returns 0 if overlapping, positive if separated.
    """
    # Check if any vertex of one is inside the other
    for i in range(6):
        if _point_in_polygon(poly1[i], poly2):
            return 0.0
    for i in range(6):
        if _point_in_polygon(poly2[i], poly1):
            return 0.0
    
    # Check all edge pairs
    min_dist = float('inf')
    for i in range(6):
        for j in range(6):
            d = _seg_seg_dist(poly1[i], poly1[(i+1)%6], poly2[j], poly2[(j+1)%6])
            min_dist = min(min_dist, d)
    
    return min_dist

def _total_overlap_penalty(centers, angles, eps=0.01):
    """Compute total penalty for overlaps between inner hexagons."""
    polys = [_hex_vertices(angles[i], centers[i]) for i in range(N)]
    
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = _poly_separation(polys[i], polys[j])
            if d < eps:
                penalty += (eps - d) ** 2
    return penalty

def _objective(params, outer_angle_rad):
    """
    Objective function: minimize outer side length + overlap penalty.
    params: 3N array [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN]
    """
    centers = np.column_stack([params[0::3], params[1::3]])
    angles = params[2::3]
    
    S = _compute_min_side_length(centers, angles, outer_angle_rad)
    pen = _total_overlap_penalty(centers, angles)
    
    return S + 100.0 * pen

def _is_valid(centers, angles, S, outer_angle_rad=0.0, tol=1e-4):
    """Check if configuration is valid: no overlaps, all inside outer hexagon."""
    polys = [_hex_vertices(angles[i], centers[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            d = _poly_separation(polys[i], polys[j])
            if d < tol:
                return False
    
    outer_normals = np.array([
        [math.cos(outer_angle_rad + k * math.pi / 3.0), 
         math.sin(outer_angle_rad + k * math.pi / 3.0)]
        for k in range(6)
    ])
    limit = S * math.sqrt(3) / 2.0
    for i in range(N):
        verts = _hex_vertices(angles[i], centers[i])
        for k in range(6):
            n = outer_normals[k]
            projs = verts @ n
            if np.max(np.abs(projs)) > limit + tol:
                return False
    return True

def _seed_solution():
    """Generate a valid seed solution using a hexagonal lattice."""
    d = math.sqrt(3)  # Distance between centers of touching hexagons
    centers = []
    angles = []
    
    # Central hexagon
    centers.append([0, 0])
    angles.append(0.0)
    
    # 6 surrounding hexagons
    for i in range(6):
        theta = i * math.pi / 3.0
        cx = d * math.cos(theta)
        cy = d * math.sin(theta)
        centers.append([cx, cy])
        angles.append(0.0)
    
    # 4 more from second ring
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2.append((x, y))
    
    # Select 4 that are well-separated
    selected_indices = [0, 3, 6, 9]
    for idx in selected_indices:
        if idx < len(ring2):
            centers.append(list(ring2[idx]))
            angles.append(0.0)
    
    centers = np.array(centers)
    angles = np.array(angles)
    
    S = _compute_min_side_length(centers, angles)
    S *= 1.02  # Small margin
    
    return centers, angles, S

def construct():
    start_time = time.time()
    
    centers0, angles0, S0 = _seed_solution()
    best_S = S0
    best_centers = centers0.copy()
    best_angles = angles0.copy()
    best_outer_angle = 0.0
    
    # Verify seed
    if not _is_valid(best_centers, best_angles, best_S, best_outer_angle):
        d = math.sqrt(3) * 1.3
        centers_fb = []
        angles_fb = []
        centers_fb.append([0, 0])
        angles_fb.append(0.0)
        for i in range(6):
            theta = i * math.pi / 3.0
            centers_fb.append([d * math.cos(theta), d * math.sin(theta)])
            angles_fb.append(0.0)
        for i in range(4):
            theta = (2*i + 1) * math.pi / 12.0
            centers_fb.append([2*d * math.cos(theta), 2*d * math.sin(theta)])
            angles_fb.append(0.0)
        best_centers = np.array(centers_fb)
        best_angles = np.array(angles_fb)
        best_S = _compute_min_side_length(best_centers, best_angles)
        best_S *= 1.02
    
    def objective_full(params):
        """
        params: 3N + 1 array [x1, y1, a1, ..., xN, yN, aN, outer_angle]
        """
        if time.time() > DEADLINE - 5.0:
            return 1e10
        
        centers = np.column_stack([params[0::3], params[1::3]])
        angles = params[2::3]
        outer_angle = params[3*N]
        
        S = _compute_min_side_length(centers, angles, outer_angle)
        pen = _total_overlap_penalty(centers, angles)
        
        return S + 100.0 * pen
    
    # Initial parameters
    x0 = np.zeros(3*N + 1)
    for i in range(N):
        x0[3*i] = best_centers[i][0]
        x0[3*i+1] = best_centers[i][1]
        x0[3*i+2] = best_angles[i]
    x0[3*N] = best_outer_angle
    
    # Try differential evolution for global search
    if time.time() < DEADLINE - 60.0:
        bounds = [(-4, 4), (-4, 4), (-math.pi, math.pi)] * N + [(-math.pi/6, math.pi/6)]
        
        try:
            res_de = differential_evolution(objective_full, bounds,
                                            maxiter=50, popsize=15,
                                            tol=1e-4, seed=42,
                                            polish=False)
            
            if res_de.fun < best_S:
                centers_new = np.column_stack([res_de.x[0::3], res_de.x[1::3]])
                angles_new = res_de.x[2::3]
                outer_angle_new = res_de.x[3*N]
                
                S_val = _compute_min_side_length(centers_new, angles_new, outer_angle_new)
                if _is_valid(centers_new, angles_new, S_val * 1.001, outer_angle_new):
                    best_S = S_val
                    best_centers = centers_new
                    best_angles = angles_new
                    best_outer_angle = outer_angle_new
        except Exception:
            pass
    
    # Local refinement with Nelder-Mead from multiple restarts
    for restart in range(8):
        if time.time() > DEADLINE - 15.0:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            x_start = np.zeros(3*N + 1)
            for i in range(N):
                x_start[3*i] = best_centers[i][0]
                x_start[3*i+1] = best_centers[i][1]
                x_start[3*i+2] = best_angles[i]
            x_start[3*N] = best_outer_angle
            
            np.random.seed(42 + restart)
            perturbation = np.random.randn(len(x_start)) * 0.3
            x_start += perturbation
        
        try:
            res = minimize(objective_full, x_start, method='Nelder-Mead',
                           options={'xatol': 1e-7, 'fatol': 1e-7, 
                                   'maxiter': 5000, 'maxfev': 8000})
            
            if res.fun < best_S:
                centers_new = np.column_stack([res.x[0::3], res.x[1::3]])
                angles_new = res.x[2::3]
                outer_angle_new = res.x[3*N]
                
                S_val = _compute_min_side_length(centers_new, angles_new, outer_angle_new)
                if _is_valid(centers_new, angles_new, S_val * 1.001, outer_angle_new):
                    best_S = S_val
                    best_centers = centers_new
                    best_angles = angles_new
                    best_outer_angle = outer_angle_new
        except Exception:
            pass
    
    # Final tight refinement
    if time.time() < DEADLINE - 10.0:
        x_start = np.zeros(3*N + 1)
        for i in range(N):
            x_start[3*i] = best_centers[i][0]
            x_start[3*i+1] = best_centers[i][1]
            x_start[3*i+2] = best_angles[i]
        x_start[3*N] = best_outer_angle
        
        try:
            res = minimize(objective_full, x_start, method='Nelder-Mead',
                           options={'xatol': 1e-8, 'fatol': 1e-8, 
                                   'maxiter': 10000, 'maxfev': 15000})
            
            if res.fun < best_S:
                centers_new = np.column_stack([res.x[0::3], res.x[1::3]])
                angles_new = res.x[2::3]
                outer_angle_new = res.x[3*N]
                
                S_val = _compute_min_side_length(centers_new, angles_new, outer_angle_new)
                if _is_valid(centers_new, angles_new, S_val * 1.001, outer_angle_new):
                    best_S = S_val
                    best_centers = centers_new
                    best_angles = angles_new
                    best_outer_angle = outer_angle_new
        except Exception:
            pass
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], math.degrees(best_angles[i])])
    
    return inner, (0.0, 0.0), best_S, math.degrees(best_outer_angle)

# EVOLVE-BLOCK-END
