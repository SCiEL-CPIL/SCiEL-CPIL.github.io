"""Packing 11 unit hexagons into minimal regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, angle_deg):
    """Return 6 vertices of unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    verts = []
    for k in range(6):
        theta = np.pi / 3 * k
        vx = np.cos(theta)
        vy = np.sin(theta)
        # Rotate
        rx = cos_a * vx - sin_a * vy
        ry = sin_a * vx + cos_a * vy
        verts.append((cx + rx, cy + ry))
    return np.array(verts)

def _point_in_hex_dist_sq(px, py, ox, oy, side, angle_deg):
    """Squared distance-like measure: 0 if point inside, positive if outside.
    For a regular hexagon, a point is inside iff it satisfies 6 half-plane constraints.
    We return the maximum violation (squared) over all 6 sides, or 0 if inside."""
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    # Transform point to hexagon local frame
    dx = px - ox
    dy = py - oy
    lx = cos_a * dx + sin_a * dy
    ly = -sin_a * dx + cos_a * dy
    # Regular hexagon with side s: vertices at angles 0,60,...,300 at distance s
    # The hexagon is intersection of 6 half-planes: |dot(p, n_i)| <= s * inradius
    # inradius = s * sqrt(3)/2
    inradius = side * np.sqrt(3) / 2.0
    max_viol = 0.0
    for k in range(6):
        theta = np.pi / 3 * k
        nx = np.cos(theta)
        ny = np.sin(theta)
        d = lx * nx + ly * ny
        if d > inradius:
            viol = d - inradius
            if viol * viol > max_viol:
                max_viol = viol * viol
    return max_viol

def _check_validity(inner, outer_center, outer_side, outer_angle):
    """Check all inner hexagons fit inside outer hexagon. Returns max violation (0 = valid)."""
    ox, oy = outer_center
    max_viol = 0.0
    for (cx, cy, ang) in inner:
        verts = _hex_vertices(cx, cy, ang)
        for (vx, vy) in verts:
            v = _point_in_hex_dist_sq(vx, vy, ox, oy, outer_side, outer_angle)
            if v > max_viol:
                max_viol = v
    return max_viol

def _objective(params, inner_fixed=None):
    """Objective: outer_side + penalty * max_violation."""
    # params: [ox, oy, side, angle_deg]
    ox, oy, side, angle = params
    if side < 0.5:
        return 1e6
    max_viol = 0.0
    for (cx, cy, ang) in inner_fixed:
        verts = _hex_vertices(cx, cy, ang)
        for (vx, vy) in verts:
            v = _point_in_hex_dist_sq(vx, vy, ox, oy, side, angle)
            if v > max_viol:
                max_viol = v
    penalty = 1000.0 * np.sqrt(max_viol)
    return side + penalty

def _honeycomb_positions():
    """Generate 11 hexagon positions in a honeycomb-like arrangement.
    1 center + 6 at distance 2 + 4 at distance 2*sqrt(3) (corners)."""
    positions = [(0.0, 0.0, 0.0)]
    # 6 neighbors at distance 2
    for k in range(6):
        theta = np.pi / 3 * k
        x = 2.0 * np.cos(theta)
        y = 2.0 * np.sin(theta)
        positions.append((x, y, 0.0))
    # 4 more at corner positions of second ring (distance 2*sqrt(3))
    # Corners are at angles 30, 150, 270, 30+180=210... let's use 30, 150, 270, 30+240
    # Actually for hexagonal lattice, second ring corners are at distance 2*sqrt(3) at angles 30, 150, 270, 330... 
    # Let me use 4 of the 6 corner positions
    for k in [0, 1, 3, 4]:  # pick 4 out of 6 corners
        theta = np.pi / 3 * k + np.pi / 6  # offset by 30 degrees
        x = 2.0 * np.sqrt(3) * np.cos(theta)
        y = 2.0 * np.sqrt(3) * np.sin(theta)
        positions.append((x, y, 0.0))
    return positions

def construct():
    global _START_TIME, _DEADLINE
    
    # Start with honeycomb arrangement
    inner = _honeycomb_positions()
    
    # Estimate initial outer hexagon size
    # Farthest point from origin: corner hexagons at distance 2*sqrt(3) ≈ 3.464, plus circumradius 1
    # So need outer circumradius >= 3.464 + 1 = 4.464, side = circumradius for regular hex
    # Actually for regular hexagon, circumradius = side length
    # The outer hex needs to contain points at distance up to ~4.464 from center
    # For a regular hexagon with side s, the inradius is s*sqrt(3)/2 and circumradius is s
    # A point at distance r from center is inside if r <= s (in the vertex directions) or r <= s*sqrt(3)/2 (in edge-mid directions)
    # Worst case: point at distance 4.464. Need s >= 4.464 if aligned with vertex, or s >= 4.464*2/sqrt(3) ≈ 5.15 if aligned with edge
    
    # Let's start with a reasonable estimate
    ox, oy = 0.0, 0.0
    side = 5.5
    angle = 0.0
    
    # Optimize outer hexagon to minimize side while containing all inner hexagons
    x0 = np.array([ox, oy, side, angle])
    inner_fixed = inner
    
    best_result = None
    best_side = side
    
    # Try optimization
    try:
        result = minimize(_objective, x0, args=(inner_fixed,), method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        if time.time() - _START_TIME > _DEADLINE:
            break
        if result.fun < best_side:
            ox, oy, side, angle = result.x
            best_side = side
            best_result = (ox, oy, side, angle)
    except Exception:
        pass
    
    # Try different initial angles for outer hexagon
    for init_angle in [30.0, 60.0, 15.0, 45.0]:
        if time.time() - _START_TIME > _DEADLINE:
            break
        x0 = np.array([0.0, 0.0, 5.5, init_angle])
        try:
            result = minimize(_objective, x0, args=(inner_fixed,), method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if time.time() - _START_TIME > _DEADLINE:
                break
            if result.fun < best_side:
                ox, oy, side, angle = result.x
                best_side = side
                best_result = (ox, oy, side, angle)
        except Exception:
            pass
    
    # Verify validity
    if best_result is not None:
        ox, oy, side, angle = best_result
    else:
        # Fallback: compute a valid but possibly suboptimal outer hex
        # Find the minimum enclosing regular hexagon
        all_verts = []
        for (cx, cy, ang) in inner:
            verts = _hex_vertices(cx, cy, ang)
            all_verts.extend(verts.tolist())
        all_verts = np.array(all_verts)
        
        # Compute max distance from origin
        max_r = np.max(np.sqrt(all_verts[:, 0]**2 + all_verts[:, 1]**2))
        # For safety, use side = max_r * 1.2 (accounting for hexagon shape)
        side = max_r * 1.15
        ox, oy, angle = 0.0, 0.0, 0.0
    
    # Final validity check - if invalid, increase side
    viol = _check_validity(inner, (ox, oy), side, angle)
    if viol > 0:
        # Increase side until valid
        while viol > 0 and time.time() - _START_TIME < _DEADLINE:
            side *= 1.05
            viol = _check_validity(inner, (ox, oy), side, angle)
    
    # Convert to list format
    inner_list = [[x, y, a] for (x, y, a) in inner]
    
    return inner_list, (float(ox), float(oy)), float(side), float(angle)
