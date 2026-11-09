"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_DEADLINE = 145.0

def _hex_support(theta):
    """
    Support function of a regular hexagon of side 1 centered at origin.
    h(theta) = max dot product of unit vector theta with any point in hexagon.
    For a regular hexagon with circumradius 1 (side 1), the vertices are at angles k*pi/3.
    The support function is the max of cos(theta - k*pi/3) over k=0..5?
    No, the support function of a convex polygon is the max of dot(n, v_i).
    For a regular hexagon of side 1, circumradius R=1.
    Vertices: (cos(k*pi/3), sin(k*pi/3)).
    h(theta) = max_k cos(theta - k*pi/3).
    """
    # Reduce theta to [0, pi/3] due to symmetry
    theta_mod = theta % (math.pi / 3)
    # The max is achieved by the vertex closest to theta.
    # cos(theta_mod) is the projection onto the vertex at angle 0.
    # Wait, if theta is in [0, pi/3], the closest vertex is at 0 or pi/3.
    # cos(theta) vs cos(pi/3 - theta).
    # Since cos is decreasing in [0, pi/2], and theta_mod in [0, pi/3],
    # cos(theta_mod) >= cos(pi/3 - theta_mod) is not always true?
    # At theta=pi/6, cos(pi/6) = sqrt(3)/2.
    # The support function is actually the apothem divided by cos(angle to normal).
    # Let's just compute the max of the 6 dot products. It's fast enough.
    max_val = -np.inf
    for k in range(6):
        angle = k * math.pi / 3
        val = math.cos(theta - angle)
        if val > max_val:
            max_val = val
    return max_val

def _hex_separation_dist(c1, c2, a1, a2):
    """
    Estimate the minimum distance between two unit hexagons.
    Returns a signed distance: positive if separated, negative if overlapping.
    We use the support function difference along the vector connecting centers.
    This is a lower bound on the true distance, but good for optimization.
    Actually, to be safe, we should check if 0 is in the Minkowski difference.
    For optimization, a smooth approximation is better.
    
    Let v = c2 - c1. Let phi = atan2(v.y, v.x).
    The distance along phi is |v| - h1(phi - a1) - h2(phi + pi - a2).
    If this is > 0 for all phi, they are separated.
    We take the min over a few sample angles or just the critical one.
    For a smooth objective, we use the value at the center-to-center angle.
    """
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    dist = math.sqrt(dx*dx + dy*dy)
    if dist < 1e-6:
        return -1.0 # Overlapping centers
    
    phi = math.atan2(dy, dx)
    
    # Support of hex 1 in direction phi
    h1 = _hex_support(phi - a1)
    # Support of hex 2 in direction -phi (towards hex 1)
    # h2(-phi) = h2(phi + pi)
    h2 = _hex_support(-phi - a2)
    
    return dist - h1 - h2

def _boundary_dist(c, a, S):
    """
    Distance from hexagon at c with rotation a to the boundary of outer hexagon.
    Outer hexagon is centered at origin, side S, rotation 0.
    Inradius of outer hex = S * sqrt(3)/2.
    We check the distance from the outer hexagon's edges to the inner hexagon.
    Equivalent to: Is the inner hexagon inside?
    Penalty if any vertex of inner hex is outside.
    Or use support function:
    The inner hex is inside if h_inner(n) + h_outer(-n) <= 0? No.
    Inner is inside Outer if for all n, h_inner(n) <= -h_outer(-n) + c_outer?
    c_outer = 0.
    h_outer(n) is the support of the outer hex.
    Condition: h_inner(n) - h_outer(n) <= 0 for all n?
    No, h_inner(n) is max dot(n, p) for p in Inner.
    If we translate Inner by c, support is h_inner(n) + dot(n, c).
    We require h_inner(n) + dot(n, c) <= h_outer(n) for all n.
    Violation = max_n (h_inner(n) + dot(n, c) - h_outer(n)).
    We approximate this by checking the 6 normals of the outer hexagon.
    """
    r_out = S * math.sqrt(3) / 2.0
    max_viol = 0.0
    for k in range(6):
        angle = k * math.pi / 3
        # Normal vector
        nx = math.cos(angle)
        ny = math.sin(angle)
        
        # Support of inner hex in direction n
        h_in = _hex_support(angle - a) + nx * c[0] + ny * c[1]
        
        # Support of outer hex in direction n
        # For outer hex of side S, support is S * _hex_support(angle)
        h_out = S * _hex_support(angle)
        
        viol = h_in - h_out
        if viol > max_viol:
            max_viol = viol
            
    return max_viol

def _is_valid_config(inner, center, side, angle):
    """
    Strict validity check using polygon intersection.
    """
    cx, cy = center
    s_out = side
    
    def _hex_verts(cx, cy, s, rot):
        vs = np.empty((6, 2))
        for k in range(6):
            a = math.radians(60.0 * k) + rot
            vs[k, 0] = cx + s * math.cos(a)
            vs[k, 1] = cy + s * math.sin(a)
        return vs

    def _point_in_hex(px, py, hx, hy, hs, hrot):
        dx = px - hx
        dy = py - hy
        ca, sa = math.cos(-hrot), math.sin(-hrot)
        lx = ca * dx + sa * dy
        ly = -sa * dx + ca * dy
        r_in = hs * math.cos(math.radians(30))
        d1 = abs(math.cos(math.radians(30)) * lx + 0.5 * ly)
        d2 = abs(ly)
        d3 = abs(-math.cos(math.radians(30)) * lx + 0.5 * ly)
        return max(d1, d2, d3) <= r_in + 1e-9

    def _min_dist_poly(A, B):
        # Simple check: if any vertex of A is in B, or B in A, dist is 0
        def point_in_poly(pt, poly):
            # Winding number or ray casting
            inside = False
            n = len(poly)
            j = n - 1
            for i in range(n):
                if ((poly[i][1] > pt[1]) != (poly[j][1] > pt[1])) and \
                   (pt[0] < (poly[j][0] - poly[i][0]) * (pt[1] - poly[i][1]) / (poly[j][1] - poly[i][1]) + poly[i][0]):
                    inside = not inside
                j = i
            return inside

        for v in A:
            if point_in_poly(v, B):
                return 0.0
        for v in B:
            if point_in_poly(v, A):
                return 0.0
            
        min_d = float('inf')
        for i in range(6):
            ax1, ay1 = A[i]
            ax2, ay2 = A[(i+1)%6]
            for j in range(6):
                bx1, by1 = B[j]
                bx2, by2 = B[(j+1)%6]
                # Segment-segment distance
                # Approximate by sampling
                for t in [0, 0.5, 1]:
                    px = ax1 + t*(ax2-ax1)
                    py = ay1 + t*(ay2-ay1)
                    for u in [0, 0.5, 1]:
                        qx = bx1 + u*(bx2-bx1)
                        qy = by1 + u*(by2-by1)
                        d = math.sqrt((px-qx)**2 + (py-qy)**2)
                        if d < min_d:
                            min_d = d
        return min_d

    # Check boundary
    outer_verts = _hex_verts(cx, cy, s_out, angle)
    for i in range(N):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        vi = _hex_verts(xi, yi, 1.0, ai)
        for v in vi:
            if not _point_in_hex(v[0], v[1], cx, cy, s_out, angle):
                return False

    # Check overlaps
    for i in range(N):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        vi = _hex_verts(xi, yi, 1.0, ai)
        for j in range(i + 1, N):
            xj, yj, aj_deg = inner[j]
            aj = math.radians(aj_deg)
            vj = _hex_verts(xj, yj, 1.0, aj)
            
            dc = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dc > 2.5:
                continue
            
            if _min_dist_poly(vi, vj) < 1e-6:
                return False
    return True

def construct():
    t_start = time.time()
    
    def _remaining():
        return _DEADLINE - (time.time() - t_start)

    # Initial guess: Honeycomb
    # Center (0,0)
    # Ring 1: 6 hexagons at distance 2*sqrt(3)/2 = sqrt(3)? No, side 1 hexagons touching.
    # Distance between centers of touching hexagons is 2 * apothem = sqrt(3).
    # Wait, side 1. Apothem = sqrt(3)/2. Distance = 2 * sqrt(3)/2 = sqrt(3) ~ 1.732.
    # In a honeycomb, the distance is 2 * r_in? No.
    # Two hexagons of side 1 sharing an edge have center distance 2 * (sqrt(3)/2) = sqrt(3).
    
    init_pos = []
    init_pos.append((0.0, 0.0, 0.0))
    r1 = math.sqrt(3)
    for k in range(6):
        a = math.radians(60.0 * k + 30.0)
        init_pos.append((r1*math.cos(a), r1*math.sin(a), 0.0))
    
    # We need 4 more. Place them in the gaps of the second ring.
    # Second ring distance is 2 * r1 = 2*sqrt(3) ~ 3.46?
    # No, the next ring in honeycomb is at distance 2*sqrt(3) from center?
    # Center to Ring 1: sqrt(3).
    # Center to Ring 2: 2*sqrt(3).
    # We pick 4 positions in Ring 2.
    r2 = 2 * math.sqrt(3)
    # To maintain some symmetry, pick angles 0, 90, 180, 270? No, hex symmetry is 60.
    # Pick 0, 60, 120, 180? That's 4.
    # Or just randomize slightly to break symmetry if needed, but start symmetric.
    angles2 = [0, 90, 180, 270] # Arbitrary, will be optimized
    for a_deg in angles2:
        a = math.radians(a_deg)
        init_pos.append((r2*math.cos(a), r2*math.sin(a), 0.0))
        
    # Ensure we have 11
    while len(init_pos) < N:
        init_pos.append((0.0, 0.0, 0.0)) # Dummy, will be moved
    init_pos = init_pos[:N]
    
    # Initial S
    max_r = 0
    for p in init_pos:
        r = math.sqrt(p[0]**2 + p[1]**2)
        if r > max_r: max_r = r
    # Max extent of hexagon is r + 1 (circumradius)
    # Outer hex inradius must be >= max_r + 1
    # S * sqrt(3)/2 >= max_r + 1
    S_init = (max_r + 1.0) * 2.0 / math.sqrt(3)
    S_init *= 1.05
    
    p0 = np.zeros(3*N + 1)
    for i in range(N):
        p0[3*i] = init_pos[i][0]
        p0[3*i+1] = init_pos[i][1]
        p0[3*i+2] = 0.0
    p0[3*N] = S_init
    
    def objective(p):
        x = p[0:3*N:3]
        y = p[1:3*N:3]
        angles = p[2:3*N:3]
        S = p[3*N]
        
        if S < 1.0:
            return 1e6
            
        pen = 0.0
        
        # Boundary penalty
        for i in range(N):
            c = np.array([x[i], y[i]])
            viol = _boundary_dist(c, angles[i], S)
            if viol > 0:
                pen += 10.0 * viol**2
                
        # Overlap penalty
        for i in range(N):
            c1 = np.array([x[i], y[i]])
            for j in range(i+1, N):
                c2 = np.array([x[j], y[j]])
                d = _hex_separation_dist(c1, c2, angles[i], angles[j])
                if d < 0.05:
                    pen += 10.0 * (0.05 - d)**2
                    
        return 0.5 * S * S + pen

    best_val = float('inf')
    best_p = p0.copy()
    
    rng = np.random.RandomState(42)
    
    # Optimization loop
    for trial in range(10):
        if _remaining() < 10.0:
            break
            
        p_init = p0.copy()
        if trial > 0:
            # Perturb
            noise = 0.5
            p_init[0:3*N] += rng.randn(3*N) * noise
            p_init[3*N] = S_init * (1.0 + rng.rand()*0.1)
            
        try:
            # Use L-BFGS-B for better convergence on smooth-ish functions
            res = minimize(objective, p_init, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
                
            # Also try Powell if L-BFGS-B fails or gets stuck
            if _remaining() < 5.0:
                 break
                 
            res2 = minimize(objective, best_p, method='Powell',
                          options={'maxiter': 50, 'ftol': 1e-6})
            if res2.fun < best_val:
                best_val = res2.fun
                best_p = res2.x.copy()
                
        except Exception:
            pass

    x = best_p[0:3*N:3]
    y = best_p[1:3*N:3]
    angles = best_p[2:3*N:3]
    S = best_p[3*N]
    
    inner = [[float(x[i]), float(y[i]), float(math.degrees(angles[i]))] for i in range(N)]
    center = (0.0, 0.0)
    angle_out = 0.0
    
    if _is_valid_config(inner, center, S, angle_out):
        return inner, center, float(S), angle_out
    else:
        # Fallback to initial guess if optimization failed
        inner_init = [[float(p[0]), float(p[1]), float(math.degrees(p[2]))] for p in init_pos]
        return inner_init, center, float(S_init), angle_out

