import numpy as np
from scipy.optimize import minimize
import time
import math

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side length, rotation in degrees."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * math.pi / 3
        verts.append([cx + side * math.cos(a), cy + side * math.sin(a)])
    return np.array(verts)

def polygon_overlap_sat(p1, p2, eps=1e-9):
    """Check if two convex polygons overlap using SAT. Returns True if they overlap."""
    for poly in [p1, p2]:
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            # Project both polygons onto this normal
            proj1 = np.dot(p1, normal)
            proj2 = np.dot(p2, normal)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 + eps or max2 < min1 + eps:
                return False  # separating axis found
    return True  # no separating axis -> overlap

def compute_min_outer_side(inner_hexes, outer_center, outer_angle_deg):
    """
    Given inner hexagons (list of [cx, cy, angle_deg] with side=1),
    find minimum outer hexagon side length that contains all inner hexagons,
    with outer center and angle fixed.
    """
    outer_cx, outer_cy = outer_center
    # Get all vertices of all inner hexagons
    all_verts = []
    for hx, hy, ha in inner_hexes:
        verts = hex_vertices(hx, hy, 1.0, ha)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # For a regular hexagon with center (cx,cy), side s, angle theta:
    # The 6 normals (outward) are at angles theta + i*pi/3, i=0..5
    # A point (x,y) is inside if for all i: dot((x-cx, y-cy), normal_i) <= s * sqrt(3)/2 * ... 
    # Actually for regular hexagon with side s, the apothem (distance from center to side) is s*sqrt(3)/2
    # The 6 side normals are at angles theta + i*pi/3
    # Constraint: dot(p - c, n_i) <= apothem = s * sqrt(3) / 2
    # So s >= max_i(dot(p - c, n_i)) / (sqrt(3)/2) for all p, i
    # But we need s >= max over all points and all normals of dot(p-c, n_i) / (sqrt(3)/2)
    # Actually, for each normal direction, the maximum projection of any point gives the required half-width
    # s_min = max_i (max_p dot(p - c, n_i)) / (sqrt(3)/2)
    # But we also need to ensure the point is on the correct side, so we take max of positive projections
    
    theta = math.radians(outer_angle_deg)
    normals = []
    for i in range(6):
        a = theta + i * math.pi / 3
        normals.append(np.array([math.cos(a), math.sin(a)]))
    
    max_proj = 0.0
    for n in normals:
        projs = np.dot(all_verts - np.array(outer_center), n)
        max_proj = max(max_proj, projs.max())
    
    # apothem = s * sqrt(3) / 2, so s = max_proj / (sqrt(3)/2) = 2 * max_proj / sqrt(3)
    s_min = 2.0 * max_proj / math.sqrt(3.0)
    return s_min

def objective(params, outer_center, outer_angle_deg):
    """
    params: 3*N vector: [x0, y0, a0, x1, y1, a1, ...]
    Returns the outer side length (to minimize) plus penalty if any overlap.
    """
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner.append([x, y, a])
    
    # Compute outer side
    s = compute_min_outer_side(inner, outer_center, outer_angle_deg)
    
    # Check overlaps - add large penalty
    penalty = 0.0
    for i in range(N):
        vi = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for j in range(i+1, N):
            vj = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if polygon_overlap_sat(vi, vj):
                penalty += 100.0
                break
        if penalty > 0:
            break
    
    return s + penalty

def make_ring12():
    """12 hexagons in a ring around center."""
    params = np.zeros(3*N)
    for i in range(N):
        a = i * 2 * math.pi / N
        r = 3.0
        params[3*i] = r * math.cos(a)
        params[3*i+1] = r * math.sin(a)
        params[3*i+2] = 0.0
    return params

def make_hex6plus6():
    """Two clusters of 6 hexagons."""
    params = np.zeros(3*N)
    # First cluster of 6 in hexagonal arrangement
    for i in range(6):
        a = i * math.pi / 3
        r = 1.8
        params[3*i] = -3.0 + r * math.cos(a)
        params[3*i+1] = r * math.sin(a)
        params[3*i+2] = 0.0
    # Second cluster of 6
    for i in range(6):
        a = i * math.pi / 3
        r = 1.8
        params[3*(i+6)] = 3.0 + r * math.cos(a)
        params[3*(i+6)+1] = r * math.sin(a)
        params[3*(i+6)+2] = 0.0
    return params

def make_rect3x4():
    """3x4 grid arrangement."""
    params = np.zeros(3*N)
    idx = 0
    for row in range(3):
        for col in range(4):
            params[3*idx] = (col - 1.5) * 2.2
            params[3*idx+1] = (row - 1.0) * 1.9
            params[3*idx+2] = 0.0
            idx += 1
    return params

def make_scattered():
    """Random scattered positions."""
    np.random.seed(42)
    params = np.zeros(3*N)
    for i in range(N):
        params[3*i] = np.random.uniform(-5, 5)
        params[3*idx if False else 3*i+1] = np.random.uniform(-5, 5)
        params[3*i+2] = np.random.uniform(0, 60)
    return params

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_params = None
    best_score = float('inf')
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try multiple initial topologies and outer angles
    initializers = [make_ring12, make_hex6plus6, make_rect3x4, make_scattered]
    outer_angles = [0.0, 30.0, 15.0, 45.0]
    
    for outer_angle in outer_angles:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        for init_func in initializers:
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            x0 = init_func()
            
            # Optimize inner positions with fixed outer center and angle
            try:
                result = minimize(
                    objective,
                    x0,
                    args=((0.0, 0.0), outer_angle),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'maxfun': 5000}
                )
                score = result.fun
                if score < best_score and score < 100:  # 100 means overlap penalty
                    best_score = score
                    best_params = result.x.copy()
                    best_outer_center = (0.0, 0.0)
                    best_outer_angle = outer_angle
            except Exception:
                pass
    
    # Also try optimizing outer center and angle jointly for best config
    if best_params is not None and time.time() - START_TIME < TIME_LIMIT - 20:
        def joint_objective(params):
            # params: 3*N + 3 (outer cx, cy, angle)
            inner_params = params[:3*N]
            ocx = params[3*N]
            ocy = params[3*N+1]
            oangle = params[3*N+2]
            
            inner = []
            for i in range(N):
                inner.append([inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]])
            
            s = compute_min_outer_side(inner, (ocx, ocy), oangle)
            
            penalty = 0.0
            for i in range(N):
                vi = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
                for j in range(i+1, N):
                    vj = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                    if polygon_overlap_sat(vi, vj):
                        penalty += 100.0
                        break
                if penalty > 0:
                    break
            return s + penalty
        
        x0_joint = np.zeros(3*N + 3)
        x0_joint[:3*N] = best_params
        x0_joint[3*N] = 0.0
        x0_joint[3*N+1] = 0.0
        x0_joint[3*N+2] = best_outer_angle
        
        try:
            result = minimize(
                joint_objective,
                x0_joint,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 3000}
            )
            if result.fun < best_score and result.fun < 100:
                best_score = result.fun
                best_params = result.x[:3*N].copy()
                best_outer_center = (result.x[3*N], result.x[3*N+1])
                best_outer_angle = result.x[3*N+2]
        except Exception:
            pass
    
    # Build final result
    if best_params is None:
        # Fallback: return something reasonable
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = []
    for i in range(N):
        inner.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    
    # Compute final outer side
    final_side = compute_min_outer_side(inner, best_outer_center, best_outer_angle)
    
    return inner, best_outer_center, final_side, best_outer_angle
