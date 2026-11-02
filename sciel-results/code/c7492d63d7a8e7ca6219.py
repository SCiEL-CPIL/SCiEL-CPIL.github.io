import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0 # Safety margin

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def get_min_enclosing_hexagon_params(points, theta_deg):
    """
    Given a set of points and a fixed outer hexagon rotation theta,
    returns the minimum side length S and center (cx, cy) of the outer hexagon.
    """
    b = math.radians(theta_deg)
    sqrt3 = math.sqrt(3)
    
    # Normals for the 3 pairs of parallel sides
    # Angles: 30+b, 90+b, 150+b
    normals = []
    for i in range(3):
        theta = math.radians(30 + 60 * i) + b
        normals.append((math.cos(theta), math.sin(theta)))
        
    M = []
    m = []
    for nx, ny in normals:
        projections = [p[0]*nx + p[1]*ny for p in points]
        M.append(max(projections))
        m.append(min(projections))
        
    widths = [M[i] - m[i] for i in range(3)]
    S = max(widths) / sqrt3
    
    # Find Center
    # Constraints:
    # M[k] - S*sqrt3/2 <= C.n_k <= m[k] + S*sqrt3/2
    L = [M[i] - S * sqrt3 / 2 for i in range(3)]
    U = [m[i] + S * sqrt3 / 2 for i in range(3)]
    
    n0 = normals[0]
    n1 = normals[1]
    
    # We want C such that C.n0 is in [L0, U0] and C.n1 is in [L1, U1]
    # To minimize S, we already fixed S. We just need a valid C.
    # The intersection of the strips must be non-empty.
    # If we pick C to be the midpoint of the feasible region in the 2D plane,
    # we can solve for C using the midpoints of the intervals if the intervals are "tight".
    # However, the intervals are projections.
    # A robust way is to find the point C that minimizes the maximum violation, 
    # but since S is the max width, the intersection is guaranteed to be non-empty 
    # (it contains the centroid of the points projected appropriately).
    
    # Let's just solve for C using the midpoints of the L/U intervals for n0 and n1.
    c_proj_0 = (L[0] + U[0]) / 2
    c_proj_1 = (L[1] + U[1]) / 2
    
    det = n0[0]*n1[1] - n0[1]*n1[0]
    if abs(det) < 1e-9:
        # Fallback
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
    else:
        cx = (c_proj_0 * n1[1] - c_proj_1 * n0[1]) / det
        cy = (n0[0] * c_proj_1 - n1[0] * c_proj_0) / det
        
    return S, cx, cy

def compute_outer_hexagon(points):
    """
    Finds the minimum side length S and optimal rotation theta for the outer hexagon.
    """
    def objective(theta):
        # theta in radians
        S, _, _ = get_min_enclosing_hexagon_params(points, math.degrees(theta))
        return S
    
    # Initial scan to find a good starting point
    best_theta = 0.0
    best_S = float('inf')
    # Scan 0 to 30 degrees (pi/6)
    for i in range(61):
        t = i * (math.pi / 6) / 60.0
        S, _, _ = get_min_enclosing_hexagon_params(points, math.degrees(t))
        if S < best_S:
            best_S = S
            best_theta = t
            
    # Refine with Nelder-Mead
    # Bounds: theta is periodic with pi/3, but we can restrict to [0, pi/6] due to symmetry?
    # Actually, the hexagon has 6-fold symmetry, so period is pi/3.
    # But the orientation of the "flat" side vs "pointy" side matters for the width.
    # The width function is periodic with pi/3.
    res = minimize(objective, best_theta, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 100})
    
    opt_theta = res.x[0]
    # Ensure theta is in [0, pi/6] or similar range for consistency?
    # The function is periodic, so any valid theta works.
    # We return the S and the theta in degrees.
    S, cx, cy = get_min_enclosing_hexagon_params(points, math.degrees(opt_theta))
    
    return S, cx, cy, math.degrees(opt_theta)

def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-5):
    """
    Checks if the configuration is valid.
    1. Inner hexagons are disjoint (distance > sqrt(3) - eps).
    2. Inner hexagons are inside the outer hexagon.
    """
    centers = []
    for x, y, ang in inner:
        centers.append((x, y, ang))
        
    # 1. Disjointness
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            x1, y1, a1 = centers[i]
            x2, y2, a2 = centers[j]
            dist = math.hypot(x1-x2, y1-y2)
            # Minimum distance for non-overlapping unit hexagons is sqrt(3).
            if dist < math.sqrt(3) - eps:
                return False
            
            # If close, check vertex containment to be safe against rotation overlaps
            if dist < math.sqrt(3) + 0.1:
                verts_i = hex_vertices(x1, y1, 1.0, a1)
                verts_j = hex_vertices(x2, y2, 1.0, a2)
                
                # Check if any vertex of i is inside j
                for v in verts_i:
                    # Fast check: distance from v to center j
                    if math.hypot(v[0]-x2, v[1]-y2) < 1.0: # Bounding circle check
                         # More precise check
                         # Project v onto normals of j
                         aj = math.radians(a2)
                         r = 1.0 * math.sqrt(3) / 2
                         valid = True
                         for k in range(3):
                             theta = math.radians(30 + 60 * k) + aj
                             nx = math.cos(theta)
                             ny = math.sin(theta)
                             proj = (v[0] - x2) * nx + (v[1] - y2) * ny
                             if abs(proj) > r + 1e-9:
                                 valid = False
                                 break
                         if valid:
                             return False
                             
                for v in verts_j:
                    if math.hypot(v[0]-x1, v[1]-y1) < 1.0:
                         ai = math.radians(a1)
                         r = 1.0 * math.sqrt(3) / 2
                         valid = True
                         for k in range(3):
                             theta = math.radians(30 + 60 * k) + ai
                             nx = math.cos(theta)
                             ny = math.sin(theta)
                             proj = (v[0] - x1) * nx + (v[1] - y1) * ny
                             if abs(proj) > r + 1e-9:
                                 valid = False
                                 break
                         if valid:
                             return False

    # 2. Containment
    ox, oy = outer_center
    ao = math.radians(outer_angle)
    ro = outer_side * math.sqrt(3) / 2
    for x, y, ang in inner:
        verts = hex_vertices(x, y, 1.0, ang)
        for v in verts:
            valid = True
            for k in range(3):
                theta = math.radians(30 + 60 * k) + ao
                nx = math.cos(theta)
                ny = math.sin(theta)
                proj = (v[0] - ox) * nx + (v[1] - oy) * ny
                if abs(proj) > ro + 1e-9:
                    valid = False
                    break
            if not valid:
                return False
    return True

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    sqrt3 = math.sqrt(3)
    
    # Initial configuration: Honeycomb lattice
    # Vectors for hexagonal lattice
    v1 = np.array([sqrt3, 0])
    v2 = np.array([sqrt3/2, 1.5])
    
    centers = []
    # 3x4 grid
    for i in range(3):
        for j in range(4):
            pos = i * v1 + j * v2
            centers.append(pos)
            
    # Center at origin
    centroid = np.mean(centers, axis=0)
    centers = [c - centroid for c in centers]
    
    inner = [[float(c[0]), float(c[1]), 0.0] for c in centers]
    
    best_score = float('inf')
    best_inner = [row[:] for row in inner]
    best_outer_center = (0,0)
    best_outer_side = 0
    best_outer_angle = 0
    
    def evaluate(config):
        # config: list of [x, y, angle]
        all_points = []
        for x, y, ang in config:
            verts = hex_vertices(x, y, 1.0, ang)
            all_points.extend([tuple(v) for v in verts])
            
        # Compute optimal outer hexagon
        S, cx, cy, theta_deg = compute_outer_hexagon(all_points)
        
        # Check validity
        if not check_validity(config, (cx, cy), S, theta_deg):
            return float('inf'), None, None, None, None
            
        return S, config, (cx, cy), S, theta_deg

    # Initial evaluation
    score, cfg, oc, side, ang = evaluate(inner)
    if score < best_score:
        best_score = score
        best_inner = cfg
        best_outer_center = oc
        best_outer_side = side
        best_outer_angle = ang

    # Simulated Annealing
    T = 0.05
    cooling_rate = 0.995
    steps = 0
    max_steps = 2000
    
    while steps < max_steps:
        if time.time() > DEADLINE:
            break
            
        i = np.random.randint(0, N)
        new_inner = [row[:] for row in inner]
        
        # Move: small translation and rotation
        dx = np.random.normal(0, 0.1)
        dy = np.random.normal(0, 0.1)
        da = np.random.normal(0, 10.0) * math.pi / 180.0
        
        new_inner[i][0] += dx
        new_inner[i][1] += dy
        new_inner[i][2] += da
        
        new_score, new_cfg, new_oc, new_side, new_ang = evaluate(new_inner)
        
        if new_score < best_score:
            best_score = new_score
            best_inner = new_cfg
            best_outer_center = new_oc
            best_outer_side = new_side
            best_outer_angle = new_ang
            inner = new_cfg
        else:
            if new_score < float('inf') and np.random.rand() < math.exp(-(new_score - best_score) / T):
                inner = new_cfg
                
        T *= cooling_rate
        steps += 1
        
    # Final check to ensure we return a valid result
    if best_score == float('inf'):
        # Fallback to initial
        score, cfg, oc, side, ang = evaluate(inner)
        if score < float('inf'):
            best_inner = cfg
            best_outer_center = oc
            best_outer_side = side
            best_outer_angle = ang

    return best_inner, best_outer_center, best_outer_side, best_outer_angle

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
