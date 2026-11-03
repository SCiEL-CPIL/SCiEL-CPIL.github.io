"""Packing 12 unit hexagons in a minimal regular hexagon."""
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

def _hex_verts(cx, cy, s, ang_deg):
    """Vertices of a regular hexagon of side s centered at (cx,cy) with rotation ang_deg.
    Vertex i at angle ang_deg + 60*i degrees, radius s (circumradius = side for regular hex)."""
    a = np.radians(ang_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _min_side_for_fixed(inner, center, ang_deg):
    """Minimal side length of outer regular hexagon (center, angle) enclosing all inner hexes.
    Constraint: along each of the 6 outward normals of the outer hex, the max |dot| of any
    inner vertex (relative to center) must be <= apothem = s*sqrt(3)/2."""
    cx, cy = center
    a = np.radians(ang_deg)
    # Outer hex normals are at angles ang_deg + 60*i (perpendicular to sides)
    # For a regular hexagon with vertices at angles ang+60*i, the normals (apothem directions)
    # are at angles ang + 60*i + 30? No. Let me think.
    # If vertices are at angles 0, 60, 120, 180, 240, 300 (flat-top if we rotate),
    # the sides are between consecutive vertices. The normal to a side bisects the angle.
    # Actually for a regular hexagon, the apothem direction to a side is at angle = midpoint of two adjacent vertex angles.
    # Vertex angles: a, a+60, a+120, a+180, a+240, a+300
    # Side between vertex at a and a+60: normal at a+30
    # So normals are at a+30, a+90, a+150, a+210, a+270, a+330
    normal_angles = a + np.radians(30) + np.radians(60) * np.arange(6)
    dirs = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    max_extent = 0.0
    for (ix, iy, iang) in inner:
        v = _hex_verts(ix, iy, 1.0, iang)
        rel = v - np.array([cx, cy])
        d = np.abs(rel @ dirs.T)
        max_extent = max(max_extent, d.max())
    
    # apothem = s * sqrt(3)/2 >= max_extent  =>  s >= 2*max_extent/sqrt(3)
    return 2.0 * max_extent / np.sqrt(3.0)

def _initial_packing():
    """Create a reasonable initial packing of 12 unit hexagons in a compact cluster."""
    # Hexagonal tiling: place in rows
    # Unit hex (circumradius 1) flat-to-flat distance = sqrt(3), point-to-point = 2
    # For a honeycomb of flat-top hexes:
    # Horizontal center-to-center in same row: 2 (touching at vertices? No, touching at sides)
    # Actually for flat-top hexes side by side, center distance = 2 (they share a vertical side)
    # Wait, flat-top means flat sides are top/bottom. Adjacent hexes horizontally share a slanted side.
    # Let me just use a known compact arrangement.
    
    # Place hexagons in a roughly hexagonal cluster
    positions = []
    # Center hex
    positions.append([0.0, 0.0, 0.0])
    # Ring of 6 around it (touching)
    for i in range(6):
        ang = np.radians(60 * i)
        x = 2.0 * np.cos(ang)
        y = 2.0 * np.sin(ang)
        positions.append([x, y, 0.0])
    # That's 7. Need 5 more. Add a partial second ring.
    # Second ring positions (distance ~4 from center for some)
    # Let's add 5 more around
    for i in range(5):
        ang = np.radians(60 * i + 30)
        x = 3.5 * np.cos(ang)
        y = 3.5 * np.sin(ang)
        positions.append([x, y, 0.0])
    
    return positions

def construct():
    t0 = time.time()
    inner = _initial_packing()
    
    def time_ok():
        return (time.time() - t0) < DEADLINE
    
    # Initial outer estimate
    center0 = (0.0, 0.0)
    ang0 = 0.0
    s0 = _min_side_for_fixed(inner, center0, ang0)
    best_s = s0
    best_center = center0
    best_ang = ang0
    
    # Optimize outer parameters (cx, cy, ang_deg)
    def obj(x):
        return _min_side_for_fixed(inner, (x[0], x[1]), x[2])
    
    # Multi-start
    starts = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 30.0],
        [0.0, 0.0, 60.0],
        [0.5, 0.5, 15.0],
        [-0.5, 0.5, 45.0],
        [0.0, 0.0, 15.0],
    ]
    
    for start in starts:
        if not time_ok():
            break
        try:
            res = minimize(obj, start, method='SLSQP',
                           options={'maxiter': 300, 'ftol': 1e-12})
            if res.fun < best_s:
                best_s = res.fun
                best_center = (res.x[0], res.x[1])
                best_ang = res.x[2]
        except Exception:
            pass
    
    # Polish with Nelder-Mead
    if time_ok():
        def obj2(x):
            return _min_side_for_fixed(inner, (x[0], x[1]), x[2])
        try:
            res2 = minimize(obj2, [best_center[0], best_center[1], best_ang],
                            method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
            if res2.fun < best_s:
                best_s = res2.fun
                best_center = (res2.x[0], res2.x[1])
                best_ang = res2.x[2]
        except Exception:
            pass
    
    return inner, best_center, best_s, best_ang
