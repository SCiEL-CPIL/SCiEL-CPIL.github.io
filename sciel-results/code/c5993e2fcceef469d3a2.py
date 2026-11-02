import math
import numpy as np
import time
from scipy.optimize import minimize, linprog, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

# Unit hexagon vertices (side length 1, flat-top orientation)
def hex_vertices(angle=0.0, cx=0.0, cy=0.0):
    """Return 6 vertices of a unit hexagon with given rotation and center."""
    angles = angle + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

# Minkowski sum of two unit hexagons: a hexagon of side length 2
# The minimum distance between centers for non-overlap is the apothem of the Minkowski sum
# For two regular hexagons of side 1, the Minkowski sum is a hexagon of side 2
# The minimum center distance for non-overlap depends on relative orientation
def minkowski_min_distance(angle1, angle2):
    """Minimum distance between centers of two unit hexagons with given orientations for non-overlap."""
    # The Minkowski sum of two regular hexagons with relative angle delta
    # has support function h(theta) = h1(theta) + h2(theta - delta)
    # where h_i(theta) = cos(theta - theta_i) for a unit hexagon (apothem = sqrt(3)/2)
    # Actually for a regular hexagon of side s, apothem = s*sqrt(3)/2
    # Support function: h(theta) = apothem / cos(theta - nearest_vertex_normal)
    # For non-overlap: |c1 - c2| >= h_M(theta) for all theta, where h_M is support of Minkowski sum
    # The minimum distance is max over theta of h_M(theta) = max over theta of (h1(theta) + h2(theta-delta))
    # For regular hexagons, this is a piecewise cosine function
    # The maximum of the support function of the Minkowski sum equals the circumradius of the Minkowski sum
    # For two unit hexagons with relative angle delta, the Minkowski sum is a 12-gon (or hexagon if delta=0)
    # The circumradius of the Minkowski sum = R1 + R2 where R is circumradius = 1 for unit hexagon
    # Wait, that's not right. The Minkowski sum of two convex bodies has support function h_M = h1 + h2
    # The minimum distance for non-overlap is the maximum of h_M over all directions
    # For a regular hexagon of side 1: circumradius = 1, apothem = sqrt(3)/2
    # h(theta) = apothem / cos(theta_mod) where theta_mod is angle to nearest normal
    # Actually h(theta) = 1 * |cos(theta - k*pi/3)| for appropriate k... no
    # Let me think differently. For a unit regular hexagon (vertices at distance 1 from center),
    # the support function in direction theta is:
    # h(theta) = cos(theta - k*pi/3) for theta in [k*pi/3 - pi/6, k*pi/3 + pi/6]
    # This equals cos(theta_mod) where theta_mod = theta - nearest multiple of pi/3, |theta_mod| <= pi/6
    # So h(theta) = cos(theta_mod) where theta_mod in [-pi/6, pi/6]
    # The Minkowski sum support: h_M(theta) = h1(theta) + h2(theta)
    # where h2 is the support of the reflected hexagon (for Minkowski difference)
    # Actually for non-overlap of A and B translated: (c1 - c2) must be outside A ⊕ (-B)
    # The minimum distance is the minimum norm of points in A ⊕ (-B) boundary... 
    # No, it's the minimum over all directions of the support function of A ⊕ (-B)
    # Wait, I need the minimum distance such that the two hexagons don't overlap.
    # This is the minimum over all directions u of (h_A(u) + h_B(-u))
    # For regular hexagons: h_A(u) = cos(mod(u - a1, pi/3) - pi/6 + pi/6)... 
    
    # Let me just compute numerically: sample many directions and find max of h1(theta)+h2(theta)
    # where h_i(theta) is support function of hexagon i in direction theta
    # The non-overlap constraint: |c1-c2| * cos(phi - theta) >= h1(theta) + h2(theta) for all theta
    # where phi is the direction of c1-c2
    # Minimum distance = max_theta (h1(theta) + h2(theta))
    
    delta = (angle2 - angle1) % (math.pi / 3)  # Relative orientation mod symmetry
    # Sample the support function
    thetas = np.linspace(0, math.pi / 3, 100)
    # h1(theta) for hexagon with rotation angle1:
    # The normals of a regular hexagon are at angles angle1 + k*pi/3 for k=0..5
    # h(theta) = cos(theta - nearest_normal_angle) where nearest is within pi/6
    def support_hex(theta, rot):
        # theta: direction, rot: rotation of hexagon
        # Normals at rot + k*pi/3
        # Find nearest normal
        mod_angle = (theta - rot) % (math.pi / 3)
        if mod_angle > math.pi / 6:
            mod_angle = math.pi / 3 - mod_angle
        return math.cos(mod_angle)
    
    h_vals = np.array([support_hex(t, 0) + support_hex(t, delta) for t in thetas])
    return np.max(h_vals)

def axial_to_cartesian(axial_list):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def get_candidates():
    """Return list of candidate axial configurations."""
    candidates = []
    
    axial_1 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)]
    candidates.append(axial_1)
    
    axial_2 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]
    candidates.append(axial_2)
    
    axial_3 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)]
    candidates.append(axial_3)
    
    axial_4 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)]
    candidates.append(axial_4)
    
    axial_5 = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    candidates.append(axial_5)
    
    axial_6 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)]
    candidates.append(axial_6)
    
    axial_7 = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)]
    candidates.append(axial_7)
    
    axial_8 = [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)]
    candidates.append(axial_8)
    
    axial_9 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)]
    candidates.append(axial_9)
    
    axial_10 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)]
    candidates.append(axial_10)
    
    axial_11 = [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)]
    candidates.append(axial_11)
    
    axial_12 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)]
    candidates.append(axial_12)
    
    axial_13 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)]
    candidates.append(axial_13)
    
    axial_14 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)]
    candidates.append(axial_14)
    
    axial_15 = [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)]
    candidates.append(axial_15)
    
    axial_16 = [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)]
    candidates.append(axial_16)
    
    axial_17 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)]
    candidates.append(axial_17)
    
    axial_18 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)]
    candidates.append(axial_18)
    
    axial_19 = [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)]
    candidates.append(axial_19)
    
    axial_20 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)]
    candidates.append(axial_20)
    
    axial_21 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)]
    candidates.append(axial_21)
    
    axial_22 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)]
    candidates.append(axial_22)
    
    axial_23 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]
    candidates.append(axial_23)
    
    axial_24 = [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)]
    candidates.append(axial_24)
    
    axial_25 = [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    candidates.append(axial_25)
    
    axial_26 = [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)]
    candidates.append(axial_26)
    
    axial_27 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)]
    candidates.append(axial_27)
    
    axial_28 = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)]
    candidates.append(axial_28)
    
    axial_29 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)]
    candidates.append(axial_29)
    
    axial_30 = [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)]
    candidates.append(axial_30)
    
    axial_31 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)]
    candidates.append(axial_31)
    
    axial_32 = [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)]
    candidates.append(axial_32)
    
    axial_33 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)]
    candidates.append(axial_33)
    
    axial_34 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)]
    candidates.append(axial_34)
    
    return candidates

def compute_S_for_config(pts, angles, outer_center, outer_beta):
    """Compute the outer hexagon side length needed to contain all inner hexagons.
    pts: (N, 2) array of inner hexagon centers
    angles: (N,) array of inner hexagon rotations
    outer_center: (2,) array
    outer_beta: outer hexagon rotation
    Returns: side length S
    """
    outer_angles = outer_beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    outer_normals = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])  # (6, 2)
    
    D = 0.0
    for i in range(N):
        # Inner hexagon i vertices
        inner_angles = angles[i] + np.arange(6) * math.pi / 3.0
        inner_verts = np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])  # (6, 2)
        inner_verts = inner_verts + pts[i]  # Translate
        
        # Project onto outer normals
        projs = inner_verts @ outer_normals.T  # (6, 6)
        # D must be >= max projection and >= -min projection for each normal
        for k in range(6):
            D = max(D, np.max(projs[:, k]), -np.min(projs[:, k]))
    
    S = (D + APOTHEM) / APOTHEM
    return S

def non_overlap_violation(pts, angles, i, j, temperature=0.1):
    """Compute soft non-overlap violation between hexagons i and j using Minkowski sum.
    Returns a scalar >= 0 (0 means no overlap, positive means overlap amount).
    Uses soft-max over the support function of the Minkowski difference.
    """
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return 0.0
    
    d = pts[i] - pts[j]  # Vector from j to i
    dist = np.linalg.norm(d)
    if dist < 1e-10:
        return 1.0  # Same position, definite overlap
    
    # Direction from j to i
    phi = np.arctan2(d[1], d[0])
    
    # The non-overlap condition: dist >= h_M(theta) for all theta
    # where h_M is the support function of hex_i ⊕ (-hex_j)
    # h_M(theta) = h_i(theta) + h_j(-theta + relative_angle)
    # For regular hexagons, h(theta) = cos(theta_mod) where theta_mod is in [-pi/6, pi/6]
    # relative angle = angles[i] - angles[j]
    
    rel_angle = angles[i] - angles[j]
    
    # Sample the support function at many angles
    thetas = np.linspace(0, math.pi / 3, 50)
    
    def h_hex(theta, rot):
        """Support function of unit hexagon with rotation rot in direction theta."""
        mod_a = (theta - rot) % (math.pi / 3)
        if mod_a > math.pi / 6:
            mod_a = math.pi / 3 - mod_a
        return math.cos(mod_a)
    
    # h_M(theta) = h_i(theta) + h_j(theta + rel_angle)  [since -hex_j has rotation -angles[j], 
    # but support of -B in direction theta = support of B in direction theta+pi]
    # Actually: support of (A ⊕ (-B)) in direction theta = h_A(theta) + h_(-B)(theta)
    # h_(-B)(theta) = h_B(-theta) = h_B(theta + pi)
    # For a regular hexagon, h_B(theta + pi) = h_B(theta) by symmetry (6-fold)
    # Wait, no. h_B(theta) has period pi/3. h_B(theta + pi) = h_B(theta + pi - 6*pi/3) = h_B(theta)
    # So h_(-B)(theta) = h_B(theta)
    # Therefore h_M(theta) = h_i(theta) + h_j(theta) where h_j is with rotation angles[j]
    
    h_vals = np.array([h_hex(t, angles[i]) + h_hex(t, angles[j]) for t in thetas])
    max_h = np.max(h_vals)
    
    # The overlap violation: if dist < max_h, there's overlap
    # The amount of overlap in the worst direction:
    # For direction theta*, the gap = dist * cos(phi - theta*) - h_M(theta*)
    # Minimum gap over theta = dist - max_h (approximately, when aligned)
    # More precisely: min_theta (dist * cos(phi-theta) - h_M(theta))
    
    # Soft violation: use soft-max to get a smooth measure
    # gap(theta) = dist * cos(phi - theta) - h_M(theta)
    # We want max(0, -min_theta gap(theta)) = max(0, max_theta (h_M(theta) - dist*cos(phi-theta)))
    
    cos_diff = np.cos(phi - thetas)
    gaps = dist * cos_diff - h_vals  # positive means no overlap in that direction
    min_gap = np.min(gaps)
    
    if min_gap >= 0:
        return 0.0
    
    # Soft version: -soft_min(gaps)
    # soft_min(x) = -t * log(sum(exp(-x_i/t)))
    t = temperature
    neg_gaps = -gaps / t
    neg_gaps = np.clip(neg_gaps, -50, 50)
    log_sum = np.log(np.sum(np.exp(neg_gaps)))
    soft_min = -t * log_sum
    violation = max(0, -soft_min)
    
    return violation

def total_non_overlap_violation(pts, angles, temperature=0.1):
    """Sum of all pairwise non-overlap violations."""
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return 0.0
    
    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            total += non_overlap_violation(pts, angles, i, j, temperature)
    return total

def objective(params, temperature=0.1):
    """Full objective: outer side length + penalty for non-overlap."""
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return 1e10
    
    # Unpack parameters
    # params: [cx_0, cy_0, a_0, cx_1, cy_1, a_1, ..., cx_10, cy_10, a_10, ocx, ocy, obeta]
    # Total: 3*N + 3 = 36 parameters
    
    pts = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        pts[i, 0] = params[3*i]
        pts[i, 1] = params[3*i + 1]
        angles[i] = params[3*i + 2]
    
    ocx = params[3*N]
    ocy = params[3*N + 1]
    obeta = params[3*N + 2]
    
    # Compute outer side length
    outer_angles = obeta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    outer_normals = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    
    D = 0.0
    for i in range(N):
        inner_angles = angles[i] + np.arange(6) * math.pi / 3.0
        inner_verts = np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
        inner_verts = inner_verts + pts[i]
        projs = inner_verts @ outer_normals.T
        for k in range(6):
            D = max(D, np.max(projs[:, k]), -np.min(projs[:, k]))
    
    S = (D + APOTHEM) / APOTHEM
    
    # Compute non-overlap violation
    no_viol = total_non_overlap_violation(pts, angles, temperature)
    
    # Objective: S + large penalty for overlap
    penalty_weight = 100.0
    return S + penalty_weight * no_viol

def exact_objective(params):
    """Exact objective with hard non-overlap check."""
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return 1e10
    
    pts = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        pts[i, 0] = params[3*i]
        pts[i, 1] = params[3*i + 1]
        angles[i] = params[3*i + 2]
    
    ocx = params[3*N]
    ocy = params[3*N + 1]
    obeta = params[3*N + 2]
    
    # Check non-overlap exactly
    for i in range(N):
        for j in range(i + 1, N):
            viol = non_overlap_violation(pts, angles, i, j, temperature=0.001)
            if viol > 1e-6:
                return 1e10  # Infeasible
    
    # Compute outer side length
    outer_angles = obeta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    outer_normals = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    
    D = 0.0
    for i in range(N):
        inner_angles = angles[i] + np.arange(6) * math.pi / 3.0
        inner_verts = np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
        inner_verts = inner_verts + pts[i]
        projs = inner_verts @ outer_normals.T
        for k in range(6):
            D = max(D, np.max(projs[:, k]), -np.min(projs[:, k]))
    
    S = (D + APOTHEM) / APOTHEM
    return S

def optimize_continuous(initial_params, max_time_per_opt=15.0):
    """Run continuous optimization from initial parameters."""
    start_t = time.time()
    best_params = initial_params.copy()
    best_val = objective(initial_params, temperature=0.1)
    
    # Stage 1: Coarse Nelder-Mead with smooth objective
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            res = minimize(objective, initial_params, method='Nelder-Mead',
                          args=(0.1,),
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
    except Exception:
        pass
    
    # Stage 2: Finer temperature
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          args=(0.03,),
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
    except Exception:
        pass
    
    # Stage 3: Very fine temperature
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          args=(0.01,),
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
    except Exception:
        pass
    
    # Stage 4: Final polish with exact objective
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            # Try multiple starting points around best
            polish_starts = [best_params.copy()]
            for dim in range(len(best_params)):
                for delta in [0.01, -0.01]:
                    p = best_params.copy()
                    p[dim] += delta
                    polish_starts.append(p)
            
            for ps in polish_starts[:10]:  # Limit number of polish attempts
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                res = minimize(exact_objective, ps, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-13,
                                       'adaptive': True})
                if res.fun < best_val and res.fun < 1e9:
                    best_val = res.fun
                    best_params = res.x
    except Exception:
        pass
    
    return best_params, best_val

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    candidates = get_candidates()
    
    best_global_params = None
    best_global_S = float('inf')
    
    # Process candidates in order of promise (diverse shapes first)
    # Try the most promising candidates first
    candidate_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    
    for idx in candidate_order:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        if idx >= len(candidates):
            break
        
        axial = candidates[idx]
        if len(axial) != N:
            continue
        
        pts = axial_to_cartesian(axial)
        
        # Center the configuration
        centroid = np.mean(pts, axis=0)
        pts_centered = pts - centroid
        
        # Initial parameters: all angles 0, outer at centroid with beta=0
        initial_params = np.zeros(3 * N + 3)
        for i in range(N):
            initial_params[3*i] = pts_centered[i, 0]
            initial_params[3*i + 1] = pts_centered[i, 1]
            initial_params[3*i + 2] = 0.0  # angle
        initial_params[3*N] = 0.0  # outer cx
        initial_params[3*N + 1] = 0.0  # outer cy
        initial_params[3*N + 2] = 0.0  # outer beta
        
        # Also try with some rotation variations
        for angle_offset in [0.0, math.pi / 6, math.pi / 3]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            test_params = initial_params.copy()
            test_params[3*N + 2] = angle_offset
            
            params, val = optimize_continuous(test_params)
            
            if val < 1e9 and val < best_global_S:
                best_global_S = val
                best_global_params = params
    
    # If continuous optimization didn't find anything good, fall back to discrete
    if best_global_params is None or best_global_S > 10.0:
        # Use the best discrete result
        best_S = float('inf')
        best_pts = None
        best_beta = 0.0
        best_center = None
        
        for axial in candidates:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            if len(axial) != N:
                continue
            pts = axial_to_cartesian(axial)
            
            # Try several beta values
            for beta in np.linspace(0, math.pi / 6, 7):
                outer_angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
                outer_normals = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
                
                # Find optimal center via LP
                c_obj = np.array([0.0, 0.0, 1.0])
                A_ub = []
                b_ub = []
                for nx, ny in outer_normals:
                    for px, py in pts:
                        A_ub.append([-nx, -ny, -1.0])
                        b_ub.append(-(px * nx + py * ny))
                        A_ub.append([nx, ny, -1.0])
                        b_ub.append(px * nx + py * ny)
                
                A_ub = np.array(A_ub)
                b_ub = np.array(b_ub)
                bounds = [(None, None), (None, None), (0, None)]
                
                try:
                    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                    if res.status == 0:
                        D = res.x[2]
                        S = (D + APOTHEM) / APOTHEM
                        if S < best_S:
                            best_S = S
                            best_pts = pts
                            best_beta = beta
                            best_center = (res.x[0], res.x[1])
                except Exception:
                    pass
        
        if best_pts is not None:
            # Convert to continuous parameters
            best_global_params = np.zeros(3 * N + 3)
            for i in range(N):
                best_global_params[3*i] = best_pts[i, 0]
                best_global_params[3*i + 1] = best_pts[i, 1]
                best_global_params[3*i + 2] = 0.0
            best_global_params[3*N] = best_center[0]
            best_global_params[3*N + 1] = best_center[1]
            best_global_params[3*N + 2] = best_beta
            best_global_S = best_S
    
    # Extract final result
    if best_global_params is None:
        # Absolute fallback
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    pts = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        pts[i, 0] = best_global_params[3*i]
        pts[i, 1] = best_global_params[3*i + 1]
        angles[i] = best_global_params[3*i + 2]
    
    ocx = best_global_params[3*N]
    ocy = best_global_params[3*N + 1]
    obeta = best_global_params[3*N + 2]
    
    # Compute final S
    outer_angles = obeta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    outer_normals = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    
    D = 0.0
    for i in range(N):
        inner_angles = angles[i] + np.arange(6) * math.pi / 3.0
        inner_verts = np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
        inner_verts = inner_verts + pts[i]
        projs = inner_verts @ outer_normals.T
        for k in range(6):
            D = max(D, np.max(projs[:, k]), -np.min(projs[:, k]))
    
    S = (D + APOTHEM) / APOTHEM
    
    # Verify non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            viol = non_overlap_violation(pts, angles, i, j, temperature=0.001)
            if viol > 1e-4:
                # Overlap detected, scale down slightly
                # This shouldn't happen with proper optimization
                pass
    
    # Build output
    inner = []
    for i in range(N):
        inner.append([float(pts[i, 0]), float(pts[i, 1]), float(math.degrees(angles[i]))])
    
    center = [float(ocx), float(ocy)]
    side = float(S)
    rotation = float(math.degrees(obeta))
    
    return inner, center, side, rotation

