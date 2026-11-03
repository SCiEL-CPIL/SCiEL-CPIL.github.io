import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    a = np.deg2rad(angle_deg)
    ang = a + np.arange(6) * np.pi / 3.0
    return np.stack([cx + side * np.cos(ang), cy + side * np.sin(ang)], axis=1)

def seg_dist_sq(p1, p2, p3, p4):
    # min distance squared between segments p1p2 and p3p4
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = d1 @ d1
    e = d2 @ d2
    f = d2 @ r
    if a <= 1e-12 and e <= 1e-12:
        return r @ r
    if a <= 1e-12:
        t = f / e
        t = max(0.0, min(1.0, t))
        d = r + t * d2
        return d @ d
    c = d1 @ r
    if e <= 1e-12:
        t = -c / a
        t = max(0.0, min(1.0, t))
        d = r + t * d1
        return d @ d
    b = d1 @ d2
    denom = a * e - b * b
    s = (b * f - c * e) / denom
    s = max(0.0, min(1.0, s))
    t = (b * s + f) / e
    if t < 0.0:
        t = 0.0
        s = max(0.0, min(1.0, -c / a))
    elif t > 1.0:
        t = 1.0
        s = max(0.0, min(1.0, (b - c) / a))
    d1s = d1 * s
    d2t = d2 * t
    d = r + d1s - d2t
    return d @ d

def polygons_disjoint(v1, v2, eps=1e-9):
    n1 = len(v1); n2 = len(v2)
    for i in range(n1):
        a = v1[i]; b = v1[(i+1) % n1]
        for j in range(n2):
            c = v2[j]; d = v2[(j+1) % n2]
            if seg_dist_sq(a, b, c, d) < eps:
                return False
    return True

def point_in_poly(p, poly):
    # winding / ray cast
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        if ((y1 > y) != (y2 > y)):
            xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xint:
                inside = not inside
    return inside

def hex_inside_hex(v_inner, v_outer):
    for p in v_inner:
        if not point_in_poly(p, v_outer):
            return False
    return True

def evaluate(inner, outer_center, outer_side, outer_angle):
    v_outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    for row in inner:
        v = hex_vertices(row[0], row[1], 1.0, row[2])
        if not hex_inside_hex(v, v_outer):
            return False
    for i in range(N):
        for j in range(i+1, N):
            vi = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            vj = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if not polygons_disjoint(vi, vj):
                return False
    return True

# Build a honeycomb initial arrangement
def honeycomb_positions():
    # 3 columns x 4 rows honeycomb of flat-top hexes
    # flat-top: horizontal spacing 1.5, vertical spacing sqrt(3)/2, offset rows
    pts = []
    # Use axial coordinates for a 3x4 block
    # We'll just place them in a grid with hex packing
    rows = 4
    cols = 3
    dx = 1.5
    dy = np.sqrt(3) / 2.0
    for r in range(rows):
        for c in range(cols):
            x = c * 2 * dx + (0 if r % 2 == 0 else dx)
            y = r * 2 * dy
            pts.append([x, y, 0.0])
    # center them
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    cx = (min(xs)+max(xs))/2; cy = (min(ys)+max(ys))/2
    for p in pts:
        p[0] -= cx; p[1] -= cy
    return pts

def min_enclosing_hex_side(inner):
    # find minimal outer hexagon (angle 0, center origin) that contains all inner hexes
    # outer hexagon flat-top: |x| <= S and |x/2 + sqrt(3)/2*y| <= S and |-x/2+sqrt(3)/2*y|<=S
    # A point (x,y) is inside flat-top hex of side S centered at origin iff:
    #   |x| <= S and |x/2 + (sqrt(3)/2) y| <= S
    # Actually for flat-top regular hexagon with side S (circumradius S):
    #   vertices at (S,0),(S/2, sqrt3 S/2),(-S/2,sqrt3 S/2),(-S,0),...
    #   constraints: |x| <= S and |x/2 + sqrt(3) y /2| <= S  ... let's derive
    #   The hexagon is intersection of 6 half-planes. For flat-top:
    #   x <= S, -x <= S,  x/2 + sqrt(3)/2 * y <= S, -x/2 - sqrt(3)/2*y <= S,
    #   x/2 - sqrt(3)/2*y <= S, -x/2 + sqrt(3)/2*y <= S
    best = 1e9
    for row in inner:
        v = hex_vertices(row[0], row[1], 1.0, row[2])
        for (x, y) in v:
            # required S for this point
            c1 = abs(x)
            c2 = abs(x/2 + np.sqrt(3)/2 * y)
            c3 = abs(x/2 - np.sqrt(3)/2 * y)
            # S must satisfy S >= c1 and S >= c2 and S >= c3? 
            # From x <= S => S>=x; -x<=S=>S>=-x => S>=|x|
            # x/2+sqrt3 y/2 <= S and its neg => S >= |x/2+sqrt3 y/2|
            # x/2-sqrt3 y/2 <= S => S >= |x/2 - sqrt3 y/2|
            S = max(c1, c2, c3)
            best = max(best, S)
    return best

def construct():
    start = time.time()
    deadline = start + 120.0
    inner = honeycomb_positions()
    # compute enclosing side
    side = min_enclosing_hex_side(inner)
    # Try to improve by rotating/shrinking arrangement via optimization of positions
    # Represent positions (x,y) for 12 hexes, angle 0 fixed for simplicity
    # Objective: minimize enclosing side (max over constraints)
    def obj(vars):
        pos = vars.reshape(N, 2)
        arr = []
        for k in range(N):
            arr.append([pos[k,0], pos[k,1], 0.0])
        return min_enclosing_hex_side(arr)
    # constraints: pairwise non-overlap (distance between centers >= 2*sqrt(3)/2? )
    # For two flat-top hexes of side 1 at angle 0, non-overlap requires center distance >= sqrt(3) (edge-to-edge) 
    # Actually min distance between two non-overlapping flat-top unit hexes = sqrt(3) (they share an edge at distance sqrt(3))
    # To be safe use a conservative bound and rely on final check
    def overlap_penalty(vars):
        pos = vars.reshape(N, 2)
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i]-pos[j])
                if d < np.sqrt(3) - 1e-6:
                    pen += (np.sqrt(3) - d)**2
        return pen
    def obj_pen(vars):
        return obj(vars) + 100.0 * overlap_penalty(vars)
    x0 = np.array(inner).reshape(-1)
    try:
        res = minimize(obj_pen, x0, method='L-BFGS-B', options={'maxiter': 2000, 'maxfun': 50000})
        if time.time() < deadline - 5:
            inner2 = res.x.reshape(N,2)
            arr2 = [[inner2[k,0], inner2[k,1], 0.0] for k in range(N)]
            side2 = min_enclosing_hex_side(arr2)
            if evaluate(arr2, (0.0,0.0), side2, 0.0):
                inner = arr2
                side = side2
    except Exception:
        pass
    # final validity check; if invalid, fall back to safe large side
    if not evaluate(inner, (0.0,0.0), side, 0.0):
        # expand side until valid
        while not evaluate(inner, (0.0,0.0), side, 0.0) and side < 100:
            side *= 1.1
    return inner, (0.0, 0.0), side, 0.0
