"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 330]) * np.pi / 180 + a
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _hex_vertices_correct(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + a
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _support_in_dir(vx, vy, dx, dy):
    """Max projection of vertices (vx,vy) onto direction (dx,dy)."""
    return np.max(vx * dx + vy * dy)

def _gap_between_hexes(c1x, c1y, a1, c2x, c2y, a2, dx, dy):
    """
    Gap along direction (dx,dy) between two unit hexagons.
    Positive means separated (no overlap in that direction).
    """
    a1r = np.radians(a1)
    a2r = np.radians(a2)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    
    v1x = np.cos(base_angles + a1r)
    v1y = np.sin(base_angles + a1r)
    v2x = np.cos(base_angles + a2r)
    v2y = np.sin(base_angles + a2r)
    
    s1 = np.max(v1x * dx + v1y * dy)
    s2 = np.max(v2x * (-dx) + v2y * (-dy))
    
    csep = (c2x - c1x) * dx + (c2y - c1y) * dy
    return csep - s1 - s2

def _min_gap_between_hexes(c1x, c1y, a1, c2x, c2y, a2):
    """
    Minimum gap between two unit hexagons using SAT.
    Returns the minimum gap over all 12 candidate separating directions.
    If all gaps are positive, the hexagons are disjoint.
    """
    a1r = np.radians(a1)
    a2r = np.radians(a2)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    
    # Edge normals for hex1
    v1x = np.cos(base_angles + a1r)
    v1y = np.sin(base_angles + a1r)
    # Edge normals are perpendicular to edges. For hexagon with vertices at angles theta,
    # edges go from vertex i to vertex i+1. The normal direction is at angle theta + 30.
    n1_angles = base_angles + a1r + np.pi / 6  # 30 degree offset
    n1x = np.cos(n1_angles)
    n1y = np.sin(n1_angles)
    
    v2x = np.cos(base_angles + a2r)
    v2y = np.sin(base_angles + a2r)
    n2_angles = base_angles + a2r + np.pi / 6
    n2x = np.cos(n2_angles)
    n2y = np.sin(n2_angles)
    
    min_gap = float('inf')
    
    # Check all 12 directions
    for dx, dy in list(zip(n1x, n1y)) + list(zip(n2x, n2y)):
        s1 = np.max(v1x * dx + v1y * dy)
        s2 = np.max(v2x * (-dx) + v2y * (-dy))
        csep = (c2x - c1x) * dx + (c2y - c1y) * dy
        gap = csep - s1 - s2
        min_gap = min(min_gap, gap)
    
    return min_gap

def _check_overlap(c1x, c1y, a1, c2x, c2y, a2):
    """Check if two unit hexagons overlap. Returns True if they overlap."""
    gap = _min_gap_between_hexes(c1x, c1y, a1, c2x, c2y, a2)
    return gap < -1e-9

def _min_outer_side(inner_pts, outer_angle_deg):
    """
    Compute minimum outer hexagon side length (centered at origin) 
    to contain all inner hexagons, given outer rotation.
    """
    theta = np.radians(outer_angle_deg)
    # Face normals of outer hexagon (perpendicular to edges)
    # For a regular hexagon with vertices at angles 0,60,..., the face normals are at 30,90,...
    face_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180 + theta
    normals_x = np.cos(face_angles)
    normals_y = np.sin(face_angles)
    
    # For a regular hexagon with side s, the inradius (center to face) is s*sqrt(3)/2
    # Constraint: for each vertex of each inner hex, (v - 0) . n_k <= s * sqrt(3)/2
    
    min_s = 0.0
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    
    for ix, iy, ia in inner_pts:
        iar = np.radians(ia)
        vx = ix + np.cos(base_angles + iar)
        vy = iy + np.sin(base_angles + iar)
        
        for k in range(6):
            dots = vx * normals_x[k] + vy * normals_y[k]
            max_dot = np.max(dots)
            if max_dot > 0:
                s_needed = max_dot / (np.sqrt(3) / 2)
                min_s = max(min_s, s_needed)
    
    return min_s

def _build_ring12():
    """
    Build 12 hexagons in a ring pattern: 6 in outer ring at distance d1, 
    6 in inner ring at distance d2, alternating angles.
    """
    pts = []
    # Outer ring: 6 hexagons at angles 0, 60, 120, 180, 240, 300
    d_outer = 4.0
    for i in range(6):
        angle = i * 60
        x = d_outer * np.cos(np.radians(angle))
        y = d_outer * np.sin(np.radians(angle))
        pts.append([x, y, 0.0])
    
    # Inner ring: 6 hexagons at angles 30, 90, 150, 210, 270, 330
    d_inner = 2.5
    for i in range(6):
        angle = i * 60 + 30
        x = d_inner * np.cos(np.radians(angle))
        y = d_inner * np.sin(np.radians(angle))
        pts.append([x, y, 0.0])
    
    return pts

def _build_hex6_plus6():
    """
    Build 6 hexagons in a ring around center + 6 more in a second ring.
    Actually: 1 center + 6 ring + 5 more? No, let's do 6+6.
    Two clusters of 6 hexagons each.
    """
    pts = []
    # First cluster: 6 hexagons in a ring
    d1 = 3.0
    for i in range(6):
        angle = i * 60
        x = d1 * np.cos(np.radians(angle))
        y = d1 * np.sin(np.radians(angle))
        pts.append([x, y, 0.0])
    
    # Second cluster: offset to the right
    offset_x = 8.0
    d2 = 3.0
    for i in range(6):
        angle = i * 60
        x = offset_x + d2 * np.cos(np.radians(angle))
        y = d2 * np.sin(np.radians(angle))
        pts.append([x, y, 0.0])
    
    return pts

def _build_rect3x4():
    """
    Build a 3x4 rectangular grid of hexagons.
    """
    pts = []
    spacing = 2 * np.sqrt(3)  # ~3.464
    for r in range(3):
        for c in range(4):
            x = c * spacing
            y = r * spacing * 1.5
            if r % 2 == 1:
                x += spacing / 2
            pts.append([x, y, 0.0])
    return pts

def _build_hex_lattice(rows, cols):
    """Build hexagonal lattice."""
    pts = []
    spacing = 2 * np.sqrt(3)
    for r in range(rows):
        for c in range(cols):
            x = c * spacing
            y = r * spacing * 1.5
            if r % 2 == 1:
                x += spacing / 2
            pts.append([x, y, 0.0])
    return pts

def _center_config(pts):
    """Center configuration at origin."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    return [[p[0] - cx, p[1] - cy, p[2]] for p in pts]

def _objective(x, deadline):
    """
    Objective function: minimize outer side length.
    x = [x1,y1,a1, x2,y2,a2, ..., x12,y12,a12, outer_angle]
    """
    if time.time() > deadline:
        return 1e10
    
    inner_pts = []
    for i in range(N):
        ix = x[3*i]
        iy = x[3*i+1]
        ia = np.degrees(x[3*i+2])
        inner_pts.append([ix, iy, ia])
    outer_angle = np.degrees(x[3*N])
    
    # Check non-overlap
    for i in range(N):
        for j in range(i+1, N):
            if _check_overlap(inner_pts[i][0], inner_pts[i][1], inner_pts[i][2],
                             inner_pts[j][0], inner_pts[j][1], inner_pts[j][2]):
                # Penalty proportional to overlap severity
                gap = _min_gap_between_hexes(inner_pts[i][0], inner_pts[i][1], inner_pts[i][2],
                                             inner_pts[j][0], inner_pts[j][1], inner_pts[j][2])
                return 1e6 * (1.0 + abs(gap)) + 1e10
    
    s = _min_outer_side(inner_pts, outer_angle)
    return s

def _optimize_config(config, deadline, maxiter=3000):
    """Optimize a given configuration using Nelder-Mead."""
    config_centered = _center_config(config)
    
    # Initial outer side estimate
    s0 = _min_outer_side(config_centered, 0.0)
    
    # Variables
    x0 = []
    for p in config_centered:
        x0.extend([p[0], p[1], np.radians(p[2])])
    x0.append(0.0)  # outer angle
    
    # Try multiple initial outer angles
    best_result = None
    best_fun = float('inf')
    
    for outer_angle_init in [0.0, 30.0, 15.0, -15.0]:
        x0_temp = x0.copy()
        x0_temp[-1] = np.radians(outer_angle_init)
        
        try:
            result = minimize(_objective, x0_temp, args=(deadline,),
                           method='Nelder-Mead',
                           options={'maxiter': maxiter, 'xatol': 1e-5, 'fatol': 1e-5,
                                  'adaptive': True, 'maxfev': maxiter * 2})
            
            if result.fun < best_fun and result.fun < 1e9:
                best_fun = result.fun
                best_result = result.x.copy()
                
        except Exception:
            pass
    
    if best_result is not None:
        inner_pts = []
        for i in range(N):
            ix = best_result[3*i]
            iy = best_result[3*i+1]
            ia = np.degrees(best_result[3*i+2])
            inner_pts.append([ix, iy, ia])
        outer_angle = np.degrees(best_result[3*N])
        
        # Final verification
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if _check_overlap(inner_pts[i][0], inner_pts[i][1], inner_pts[i][2],
                                 inner_pts[j][0], inner_pts[j][1], inner_pts[j][2]):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            s_final = _min_outer_side(inner_pts, outer_angle)
            return inner_pts, s_final, outer_angle
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_score = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = 40.0
    best_angle = 0.0
    
    # Multiple topology initializations
    configs = []
    
    # Ring 12 (two concentric rings of 6)
    configs.append(_build_ring12())
    
    # Hex 6+6 (two clusters)
    configs.append(_build_hex6_plus6())
    
    # Rect 3x4
    configs.append(_build_rect3x4())
    
    # Hex lattice 3x4
    configs.append(_build_hex_lattice(3, 4))
    
    # Hex lattice 4x3
    configs.append(_build_hex_lattice(4, 3))
    
    # Hex lattice 2x6
    configs.append(_build_hex_lattice(2, 6))
    
    # Hex lattice 6x2
    configs.append(_build_hex_lattice(6, 2))
    
    # Additional: compact hexagonal arrangement
    # Center + 6 ring + 5 more
    pts = [[0.0, 0.0, 0.0]]
    d1 = 2 * np.sqrt(3)
    for i in range(6):
        angle = i * 60
        x = d1 * np.cos(np.radians(angle))
        y = d1 * np.sin(np.radians(angle))
        pts.append([x, y, 0.0])
    # Add 5 more in second ring
    d2 = 4 * np.sqrt(3)
    for i in range(5):
        angle = i * 60 + 30
        x = d2 * np.cos(np.radians(angle))
        y = d2 * np.sin(np.radians(angle))
        pts.append([x, y, 0.0])
    configs.append(pts)
    
    # Optimize each config
    for config_idx, config in enumerate(configs):
        if time.time() > deadline:
            break
        
        result = _optimize_config(config, deadline, maxiter=2000)
        
        if result is not None:
            inner_pts, s, angle = result
            if s < best_score:
                best_score = s
                best_inner = inner_pts
                best_center = (0.0, 0.0)
                best_side = s
                best_angle = angle
    
    # If we have time, do a second pass with more iterations on the best config
    if best_inner is not None and time.time() < deadline - 30:
        # Re-optimize the best result with more iterations
        x0 = []
        for p in best_inner:
            x0.extend([p[0], p[1], np.radians(p[2])])
        x0.append(np.radians(best_angle))
        
        try:
            result = minimize(_objective, x0, args=(deadline,),
                           method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6,
                                  'adaptive': True, 'maxfev': 10000})
            
            if result.fun < best_score and result.fun < 1e9:
                inner_pts = []
                for i in range(N):
                    ix = result.x[3*i]
                    iy = result.x[3*i+1]
                    ia = np.degrees(result.x[3*i+2])
                    inner_pts.append([ix, iy, ia])
                outer_angle = np.degrees(result.x[3*N])
                
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if _check_overlap(inner_pts[i][0], inner_pts[i][1], inner_pts[i][2],
                                         inner_pts[j][0], inner_pts[j][1], inner_pts[j][2]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    s_final = _min_outer_side(inner_pts, outer_angle)
                    if s_final < best_score:
                        best_score = s_final
                        best_inner = inner_pts
                        best_center = (0.0, 0.0)
                        best_side = s_final
                        best_angle = outer_angle
        except Exception:
            pass
    
    # Return best result
    if best_inner is not None:
        return best_inner, best_center, best_side, best_angle
    
    # Fallback
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
