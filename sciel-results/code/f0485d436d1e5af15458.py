"""Joint Nelder-Mead optimization of 11 unit hexagon positions + outer hexagon rotation.
Uses SAT-based collision (polygon distance) and computes true minimum enclosing regular hexagon side."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
DEADLINE = 140.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _hex_vertices(cx, cy, side, angle_deg):
    a = math.radians(angle_deg)
    return [(cx + side * math.cos(a + k * math.pi / 3.0),
             cy + side * math.sin(a + k * math.pi / 3.0)) for k in range(6)]

def _seg_dist_sq(p1, p2, q1, q2):
    """Minimum squared distance between segments p1-p2 and q1-q2."""
    # Use robust segment-segment distance
    d1 = (p2[0]-p1[0], p2[1]-p1[1])
    d2 = (q2[0]-q1[0], q2[1]-q1[1])
    r = (p1[0]-q1[0], p1[1]-q1[1])
    a = d1[0]*d1[0]+d1[1]*d1[1]
    e = d2[0]*d2[0]+d2[1]*d2[1]
    f = d2[0]*r[0]+d2[1]*r[1]
    if a < 1e-14:
        s = 0.0
    else:
        s = (d1[0]*r[0]+d1[1]*r[1]) / a
        s = max(0.0, min(1.0, s))
    if e < 1e-14:
        t = 0.0
    else:
        t = (f + d1[0]*d2[0]*s + d1[1]*d2[1]*s) / e
        t = max(0.0, min(1.0, t))
    dx = r[0] + d1[0]*s - d2[0]*t
    dy = r[1] + d1[1]*s - d2[1]*t
    return dx*dx + dy*dy

def _poly_dist_sq(A, B):
    """Minimum squared distance between convex polygons A and B."""
    best = float('inf')
    nA = len(A)
    nB = len(B)
    for i in range(nA):
        a1 = A[i]
        a2 = A[(i+1) % nA]
        for j in range(nB):
            b1 = B[j]
            b2 = B[(j+1) % nB]
            d2 = _seg_dist_sq(a1, a2, b1, b2)
            if d2 < best:
                best = d2
    return best

def _point_in_polygon(p, poly):
    """Ray casting for point-in-convex-polygon test."""
    x, y = p
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _hex_separated(A, B, min_gap):
    """Check if two convex hexagons are separated by at least min_gap."""
    # Check if any vertex of A is inside B (or vice versa)
    for p in A:
        if _point_in_polygon(p, B):
            return False
    for p in B:
        if _point_in_polygon(p, A):
            return False
    # Check edge-edge proximity
    d2 = _poly_dist_sq(A, B)
    return math.sqrt(d2) >= min_gap

def _compute_min_side_length(all_verts, outer_angle_deg):
    """Compute minimum side of regular hexagon (centered at origin) containing all_verts."""
    phi = math.radians(outer_angle_deg)
    sqrt3 = math.sqrt(3.0)
    r_min = 0.0
    for k in range(6):
        alpha = phi + k * math.pi / 3.0
        nx = math.cos(alpha)
        ny = math.sin(alpha)
        max_pos = 0.0
        max_neg = 0.0
        for (px, py) in all_verts:
            proj = px * nx + py * ny
            if proj > max_pos:
                max_pos = proj
            if -proj > max_neg:
                max_neg = -proj
        r_candidate = max(max_pos, max_neg)
        if r_candidate > r_min:
            r_min = r_candidate
    return 2.0 * r_min / sqrt3

def _objective(x, min_gap=0.01):
    """x = [x0,y0, x1,y1, ..., x10,y10, outer_angle_deg]."""
    positions = x[:2*N].reshape(N, 2)
    outer_angle = x[2*N]
    
    # Build all hexagon vertex lists
    hex_verts = []
    for i in range(N):
        hex_verts.append(_hex_vertices(positions[i][0], positions[i][1], 1.0, 0.0))
    
    # Check separation
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            # Quick bbox check
            for p in hex_verts[i]:
                if _point_in_polygon(p, hex_verts[j]):
                    penalty += 100.0
                    break
            else:
                for p in hex_verts[j]:
                    if _point_in_polygon(p, hex_verts[i]):
                        penalty += 100.0
                        break
                else:
                    d2 = _poly_dist_sq(hex_verts[i], hex_verts[j])
                    d = math.sqrt(d2)
                    if d < min_gap:
                        penalty += (min_gap - d) * 1000.0
    if penalty > 0:
        return penalty + _compute_min_side_length(
            [v for hv in hex_verts for v in hv], outer_angle)
    
    all_verts = [v for hv in hex_verts for v in hv]
    return _compute_min_side_length(all_verts, outer_angle)

def _seed_solution():
    """Generate initial positions in a compact hexagonal cluster pattern."""
    sqrt3 = math.sqrt(3.0)
    positions = [(0.0, 0.0)]
    # Ring 1
    for i in range(6):
        theta = math.radians(60 * i)
        positions.append((sqrt3 * math.cos(theta), sqrt3 * math.sin(theta)))
    # Ring 2: 4 more at "corner" positions
    for i in range(4):
        theta = math.radians(60 + 60 * i)
        positions.append((2 * sqrt3 * math.cos(theta + math.pi/6),
                          2 * sqrt3 * math.sin(theta + math.pi/6)))
    return np.array(positions, dtype=float)

def _is_valid(positions, outer_angle_deg, min_gap=0.005):
    """Verify all hexagons are disjoint and inside the outer hexagon."""
    hex_verts = []
    for i in range(N):
        hex_verts.append(_hex_vertices(positions[i][0], positions[i][1], 1.0, 0.0))
    
    # Check pairwise separation
    for i in range(N):
        for j in range(i+1, N):
            for p in hex_verts[i]:
                if _point_in_polygon(p, hex_verts[j]):
                    return False
            for p in hex_verts[j]:
                if _point_in_polygon(p, hex_verts[i]):
                    return False
            d2 = _poly_dist_sq(hex_verts[i], hex_verts[j])
            if math.sqrt(d2) < min_gap:
                return False
    
    # Check all vertices inside outer hexagon
    all_verts = [v for hv in hex_verts for v in hv]
    side = _compute_min_side_length(all_verts, outer_angle_deg)
    # Re-verify: each vertex must satisfy all 6 half-plane constraints
    phi = math.radians(outer_angle_deg)
    r = side * math.sqrt(3.0) / 2.0
    for (px, py) in all_verts:
        for k in range(6):
            alpha = phi + k * math.pi / 3.0
            nx = math.cos(alpha)
            ny = math.sin(alpha)
            if px * nx + py * ny > r + 1e-9:
                return False
            if -px * nx - py * ny > r + 1e-9:
                return False
    return True

def construct():
    start = time.time()
    
    # Initial seed
    init_pos = _seed_solution()
    
    best_side = float('inf')
    best_x = None
    
    # Try multiple restarts
    n_restarts = 4
    for restart in range(n_restarts):
        if _elapsed() > DEADLINE - 20:
            break
        
        if restart == 0:
            x0 = np.append(init_pos.flatten(), 0.0)
        else:
            # Perturb from best found so far, or from seed with noise
            if best_x is not None:
                x0 = best_x + np.random.randn(2*N + 1) * 0.3
            else:
                x0 = np.append(init_pos.flatten() + np.random.randn(2*N) * 0.5,
                               np.random.uniform(0, 60))
        
        # Nelder-Mead optimization
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-4,
                                    'adaptive': True})
            if res.fun < best_side:
                best_side = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
        
        if _elapsed() > DEADLINE - 15:
            break
    
    # Final validation and result
    if best_x is not None:
        positions = best_x[:2*N].reshape(N, 2)
        outer_angle = best_x[2*N]
        hex_verts = [_hex_vertices(positions[i][0], positions[i][1], 1.0, 0.0) for i in range(N)]
        all_verts = [v for hv in hex_verts for v in hv]
        side = _compute_min_side_length(all_verts, outer_angle)
        
        # Verify
        if _is_valid(positions, outer_angle):
            inner = [[positions[i][0], positions[i][1], 0.0] for i in range(N)]
            return inner, (0.0, 0.0), side, outer_angle
    
    # Fallback: rigid cluster with grid-search rotation
    sqrt3 = math.sqrt(3.0)
    centers = [(0.0, 0.0)]
    for i in range(6):
        theta = math.radians(60 * i)
        centers.append((sqrt3 * math.cos(theta), sqrt3 * math.sin(theta)))
    for i in range(4):
        theta = math.radians(60 + 60 * i)
        centers.append((2 * sqrt3 * math.cos(theta + math.pi/6),
                        2 * sqrt3 * math.sin(theta + math.pi/6)))
    
    all_verts = []
    for (cx, cy) in centers:
        all_verts.extend(_hex_vertices(cx, cy, 1.0, 0.0))
    
    best_s = float('inf')
    best_a = 0.0
    for angle in np.linspace(0, 60, 200):
        s = _compute_min_side_length(all_verts, angle)
        if s < best_s:
            best_s = s
            best_a = angle
    
    inner = [[cx, cy, 0.0] for (cx, cy) in centers]
    return inner, (0.0, 0.0), best_s, best_a

# EVOLVE-BLOCK-START
# construct function defined above
# EVOLVE-BLOCK-END
