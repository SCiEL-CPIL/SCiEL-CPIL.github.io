"""Kinematic closure strategy for packing 12 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3.0
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def _point_in_convex_polygon(p, polygon):
    n = len(polygon)
    sign = 0
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cross) < 1e-10:
            continue
        if cross > 0:
            if sign < 0:
                return False
            sign = 1
        else:
            if sign > 0:
                return False
            sign = -1
    return True

def _segments_intersect(p1, p2, p3, p4):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    def on_segment(p, q, r):
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and \
               min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
    if d1 == 0 and on_segment(p3, p4, p1): return True
    if d2 == 0 and on_segment(p3, p4, p2): return True
    if d3 == 0 and on_segment(p1, p2, p3): return True
    if d4 == 0 and on_segment(p1, p2, p4): return True
    return False

def _compute_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    cx, cy = outer_center
    all_verts = []
    for i, (ix, iy, iang) in enumerate(inner_positions):
        verts = _hex_vertices(ix, iy, 1.0, iang)
        all_verts.extend(verts.tolist())
    all_verts = np.array(all_verts)
    dx = all_verts[:, 0] - cx
    dy = all_verts[:, 1] - cy
    angle = -np.radians(outer_angle)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    max_proj = 0
    for k in range(6):
        theta = np.radians(30 + k * 60)
        nx = np.cos(theta)
        ny = np.sin(theta)
        proj = lx * nx + ly * ny
        max_proj = max(max_proj, np.max(proj))
    side = max_proj / (np.sqrt(3) / 2)
    return side

def _objective(params):
    scale, rot_deg = params
    positions_axial = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2)
    ]
    rot = np.radians(rot_deg)
    cos_r = np.cos(rot)
    sin_r = np.sin(rot)
    inner_positions = []
    for q, r in positions_axial:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        sx = scale * x
        sy = scale * y
        rx = sx * cos_r - sy * sin_r
        ry = sx * sin_r + sy * cos_r
        inner_positions.append((rx, ry, rot_deg))
    outer_center = (0.0, 0.0)
    outer_angle = rot_deg
    side = _compute_outer_side(inner_positions, [p[2] for p in inner_positions], outer_center, outer_angle)
    return side

def construct():
    x0 = [1.0, 0.0]
    result = minimize(_objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
    scale, rot_deg = result.x
    positions_axial = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2)
    ]
    rot = np.radians(rot_deg)
    cos_r = np.cos(rot)
    sin_r = np.sin(rot)
    inner = []
    for q, r in positions_axial:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        sx = scale * x
        sy = scale * y
        rx = sx * cos_r - sy * sin_r
        ry = sx * sin_r + sy * cos_r
        inner.append([rx, ry, rot_deg])
    outer_center = (0.0, 0.0)
    outer_angle = rot_deg
    side = _compute_outer_side(inner, [p[2] for p in inner], outer_center, outer_angle)
    return inner, outer_center, side, outer_angle
