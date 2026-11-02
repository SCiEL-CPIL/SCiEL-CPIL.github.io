import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

# Precompute hexagon vertices for side length 1 (flat-top orientation, angle=0)
# A regular hexagon with side 1 has vertices at distance 1 from center
# At angle 0 (flat-top), vertices are at 0, 60, 120, 180, 240, 300 degrees
# Actually, let's use pointy-top: vertices at 30, 90, 150, 210, 270, 330 degrees
# Or flat-top: vertices at 0, 60, 120, 180, 240, 300 degrees
# The apothem is the distance from center to edge midpoint = sqrt(3)/2

# For a unit hexagon, the 6 vertices (flat-top, angle=0) are:
# v_k = (cos(k*60°), sin(k*60°)) for k=0..5
# But we need to be careful about orientation.

# Let's define the hexagon with a "flat" top edge.
# The 6 vertices at angles 0, 60, 120, 180, 240, 300 degrees from center.
# v0 = (1, 0), v1 = (0.5, sqrt(3)/2), v2 = (-0.5, sqrt(3)/2), v3 = (-1, 0), v4 = (-0.5, -sqrt(3)/2), v5 = (0.5, -sqrt(3)/2)

HEX_VERTICES = np.array([
    [math.cos(k * math.pi / 3.0), math.sin(k * math.pi / 3.0)]
    for k in range(6)
])

# The 6 edge normals (pointing outward) for a flat-top hexagon at angle 0:
# The edges are between consecutive vertices.
# Edge 0: v0 to v1. Midpoint: (0.75, sqrt(3)/4). Normal direction: perpendicular to edge.
# Edge vector: v1 - v0 = (-0.5, sqrt(3)/2). Normal (outward): (sqrt(3)/2, 0.5) normalized = (sqrt(3)/2, 0.5) since length is 1.
# Actually, the outward normal for edge k (between v_k and v_{k+1}) points in the direction of the angle bisector of the two vertices.
# For flat-top hex, the 6 outward normals are at angles 30°, 90°, 150°, 210°, 270°, 330°.
# n_k = (cos(30° + k*60°), sin(30° + k*60°))
EDGE_NORMALS = np.array([
    [math.cos(math.pi / 6.0 + k * math.pi / 3.0), math.sin(math.pi / 6.0 + k * math.pi / 3.0)]
    for k in range(6)
])

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Initial guess: a hexagonal lattice arrangement
    # For N=11, a common arrangement is a central hexagon surrounded by 6, plus 4 more.
    # Let's use a known good starting configuration.
    # Axial coordinates for a compact 11-hexagon cluster:
    # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 7 hexes
    # Add: (2,0), (-2,0), (0,2), (0,-2) -> 11 hexes
    # But this is a "star" shape. Let's use a more compact one.
    # A 3x3 grid minus 2 corners? No, that's not hexagonal.
    # Let's use the axial coordinates from the previous program's best candidate.
    # The previous program used axial_1: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,1), (-1,2), (-2,1), (2,-1)
    # Let's convert this to Cartesian.
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    initial_pts = axial_to_cartesian(axial_1)
    initial_angles = np.zeros(N)
    
    # Compute initial center and size
    cx0 = np.mean(initial_pts[:, 0])
    cy0 = np.mean(initial_pts[:, 1])
    
    # Initial outer hexagon: centered at (cx0, cy0), angle 0
    # Find the minimum side length S such that all hexagons fit.
    # For a given S, the outer hexagon has apothem A_outer = S * APOTHEM.
    # The constraint is that for each inner hexagon i and each outer edge normal n_k,
    # the projection of the inner hexagon's furthest point in direction n_k must be <= A_outer.
    # The furthest point of inner hexagon i in direction n_k is:
    # max_j ( (p_i + R(theta_i) v_j) . n_k )
    # = p_i . n_k + max_j ( (R(theta_i) v_j) . n_k )
    # = p_i . n_k + max_j ( v_j . R(-theta_i) n_k )
    # = p_i . n_k + apothem_of_hex_in_direction(R(-theta_i) n_k)
    # Since the hexagon is regular, the support function in any direction is the apothem.
    # Wait, no. The support function of a regular hexagon in direction u is:
    # h(u) = max_j (v_j . u)
    # For a regular hexagon, h(u) is not constant. It varies between the apothem and the circumradius.
    # Specifically, if u is aligned with an edge normal, h(u) = apothem.
    # If u is aligned with a vertex, h(u) = 1 (circumradius).
    # So we need to compute h(R(-theta_i) n_k) for each i, k.
    
    def support_function(angle, direction):
        # Returns max_j (v_j . direction) where v_j are vertices of hexagon rotated by angle
        # Equivalent to max_j (R(angle) v_j) . direction = max_j v_j . (R(-angle) direction)
        # Let d_rot = R(-angle) direction
        # Then we need max_j (v_j . d_rot)
        # We can precompute this for any direction.
        # d_rot = (cos(angle)*d_x + sin(angle)*d_y, -sin(angle)*d_x + cos(angle)*d_y)
        # Actually, R(-angle) = [[cos(angle), sin(angle)], [-sin(angle), cos(angle)]]
        # So d_rot = (cos(angle)*d_x + sin(angle)*d_y, -sin(angle)*d_x + cos(angle)*d_y)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        d_rot = np.array([
            cos_a * direction[0] + sin_a * direction[1],
            -sin_a * direction[0] + cos_a * direction[1]
        ])
        # Now compute max_j (v_j . d_rot)
        projs = HEX_VERTICES @ d_rot
        return np.max(projs)

    # Initial size S0
    max_proj = 0.0
    for i in range(N):
        for k in range(6):
            n_k = EDGE_NORMALS[k]
            # Support of hexagon i in direction n_k
            h_i_k = support_function(initial_angles[i], n_k)
            proj = initial_pts[i, 0] * n_k[0] + initial_pts[i, 1] * n_k[1] + h_i_k
            if proj > max_proj:
                max_proj = proj
            # Also check the opposite direction? No, the outer hexagon contains the inner ones,
            # so we need the inner hexagons to be inside. The constraint is that the inner hexagon
            # does not cross the outer boundary. The outer boundary in direction n_k is at distance A_outer.
            # So we need proj <= A_outer.
            # Also, we need to check the other 6 directions? No, the outer hexagon is defined by 6 half-planes.
            # The half-planes are: n_k . x <= A_outer for k=0..5, where n_k are the outward normals.
            # So we only need to check these 6 directions.
    
    A_outer0 = max_proj
    S0 = A_outer0 / APOTHEM

    # Define the optimization problem
    # Variables: x_i, y_i, theta_i for i=0..N-1, and S (outer side length)
    # Total variables: 3*N + 1 = 34
    # Objective: minimize S
    # Constraints:
    # 1. Non-overlap: for each pair (i, j), i < j, the hexagons must not overlap.
    #    Two convex polygons do not overlap if and only if there exists a separating axis.
    #    For two regular hexagons, the separating axes are the 12 edge normals of the two hexagons (6 from each).
    #    Actually, the Minkowski difference of two regular hexagons is a regular dodecagon (12-gon).
    #    The non-overlap condition is that the distance between centers is greater than or equal to
    #    the support function of the Minkowski difference in the direction of the line connecting the centers.
    #    Alternatively, we can use the fact that two convex polygons are disjoint if and only if
    #    for every edge normal of either polygon, the projection of the two polygons on that axis are disjoint.
    #    This gives 12 inequalities per pair, which is too many.
    #    
    #    A better approach: use the Minkowski sum. The condition that two convex sets A and B are disjoint
    #    is equivalent to 0 not being in (B - A). For polygons, this can be checked using the separating axis theorem.
    #    For two regular hexagons, the separating axes are the 12 edge normals (6 from each hexagon).
    #    But due to symmetry, we only need to check 6 axes? No, we need to check all 12.
    #    
    #    Let's use a different approach: the distance between two convex polygons can be computed,
    #    but it's not smooth. Instead, we can use the following:
    #    Two hexagons i and j do not overlap if and only if for all 12 edge normals n of the two hexagons,
    #    the projections of the two hexagons on n are disjoint.
    #    This is equivalent to: for all 12 normals n, 
    #    (p_i . n + h_i(n)) <= (p_j . n - h_j(n))  OR  (p_j . n + h_j(n)) <= (p_i . n - h_i(n))
    #    This is a disjunction, which is non-smooth.
    #    
    #    A smoother approach: use the squared distance between the two hexagons.
    #    The distance between two convex polygons is a convex function of the translation vectors,
    #    but it's not smooth at points where the closest points change.
    #    
    #    Let's use the following constraint: the distance between the two hexagons must be >= 0.
    #    We can approximate the distance using the support functions.
    #    The distance between two convex sets A and B is:
    #    d(A, B) = max_{||u||=1} (h_A(u) - h_B(u) + (p_B - p_A) . u)
    #    Wait, that's not right. The distance is:
    #    d(A, B) = max_{||u||=1} (h_A(u) - h_B(-u) + (p_B - p_A) . u)
    #    No, the correct formula is:
    #    d(A, B) = max_{||u||=1} (h_A(u) + h_B(-u) - (p_B - p_A) . u)
    #    Actually, the distance between two convex sets A and B is:
    #    d(A, B) = max_{||u||=1} (h_A(u) - h_B(u) + (p_B - p_A) . u)
    #    Let's verify: if A and B are points, d = ||p_A - p_B||.
    #    h_A(u) = p_A . u, h_B(u) = p_B . u.
    #    h_A(u) - h_B(u) + (p_B - p_A) . u = p_A . u - p_B . u + p_B . u - p_A . u = 0.
    #    That's not right.
    #    
    #    The correct formula for the distance between two convex sets A and B is:
    #    d(A, B) = max_{||u||=1} (h_A(u) - h_B(u) + (p_B - p_A) . u)
    #    is not correct. The correct formula is:
    #    d(A, B) = max_{||u||=1} (h_A(u) + h_B(-u) - (p_B - p_A) . u)
    #    Let's verify with points: h_A(u) = p_A . u, h_B(-u) = -p_B . u.
    #    h_A(u) + h_B(-u) - (p_B - p_A) . u = p_A . u - p_B . u - p_B . u + p_A . u = 2(p_A - p_B) . u.
    #    max over ||u||=1 is 2||p_A - p_B||. That's twice the distance.
    #    
    #    Actually, the distance between two convex sets is:
    #    d(A, B) = max_{||u||=1} (h_A(u) - h_B(u) + (p_B - p_A) . u)
    #    is the signed distance, but it's not the Euclidean distance.
    #    
    #    Let's use a different approach: we can use the fact that two convex polygons are disjoint
    #    if and only if there exists a separating axis. We can use the 12 edge normals as candidate axes.
    #    For each axis n, we can compute the gap between the two projections.
    #    The gap is: min(A_proj) - max(B_proj) or min(B_proj) - max(A_proj).
    #    The two hexagons are disjoint if and only if for at least one axis n, the gap is >= 0.
    #    This is a disjunction, which is non-smooth.
    #    
    #    To make it smooth, we can use the minimum gap over all 12 axes.
    #    If the minimum gap is >= 0, the hexagons are disjoint.
    #    The minimum gap is a concave function of the positions and angles, so maximizing it is a convex problem.
    #    But we want to minimize S, so we want the gap to be >= 0, which is a non-convex constraint.
    #    
    #    Let's use a penalty approach: add a penalty term for overlapping hexagons.
    #    The penalty is the square of the minimum gap (if negative).
    #    
    #    Actually, let's use the following approach:
    #    For each pair (i, j), compute the 12 gaps (one for each edge normal of the two hexagons).
    #    The minimum gap g_ij = min_{n in N_ij} gap_ij(n)
    #    We want g_ij >= 0.
    #    We can add a penalty term: max(0, -g_ij)^2
    #    This is smooth except at g_ij = 0.
    #    
    #    Let's implement this.
    
    def compute_gap(p_i, theta_i, p_j, theta_j, n):
        # Compute the gap between hexagon i and hexagon j in direction n.
        # gap = min(A_proj) - max(B_proj) where A_proj is the projection of hexagon i on n,
        # and B_proj is the projection of hexagon j on n.
        # But we need to consider both directions: n and -n.
        # Actually, the gap in direction n is:
        # gap = (p_i . n - h_i(n)) - (p_j . n + h_j(n))
        # where h_i(n) is the support function of hexagon i in direction n.
        # But this is the gap when hexagon i is "behind" hexagon j in direction n.
        # We also need to consider the opposite direction.
        # The actual gap is the maximum of the gaps in all 12 directions.
        # Wait, no. The two hexagons are disjoint if and only if there exists a direction n
        # such that the projections are disjoint. The gap in direction n is:
        # gap(n) = min(A_proj) - max(B_proj)
        # The hexagons are disjoint if max_n gap(n) >= 0.
        # So we need max_n gap(n) >= 0.
        # This is equivalent to: for all n, gap(n) >= 0 is NOT required. Only one n is enough.
        # So the constraint is: max_n gap(n) >= 0.
        # This is a non-convex constraint.
        
        # Let's compute the gap in direction n.
        # h_i(n) = max_k (v_k . R(-theta_i) n)
        # h_j(n) = max_k (v_k . R(-theta_j) n)
        # A_proj_min = p_i . n - h_i(n)
        # B_proj_max = p_j . n + h_j(n)
        # gap = A_proj_min - B_proj_max
        h_i_n = support_function(theta_i, n)
        h_j_n = support_function(theta_j, n)
        A_proj_min = p_i[0] * n[0] + p_i[1] * n[1] - h_i_n
        B_proj_max = p_j[0] * n[0] + p_j[1] * n[1] + h_j_n
        gap = A_proj_min - B_proj_max
        return gap

    def objective(params):
        # params: [x_0, y_0, theta_0, x_1, y_1, theta_1, ..., x_{N-1}, y_{N-1}, theta_{N-1}, S]
        # Unpack
        pts = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            pts[i, 0] = params[3*i]
            pts[i, 1] = params[3*i+1]
            angles[i] = params[3*i+2]
        S = params[3*N]
        
        # Objective: minimize S
        obj = S
        
        # Penalty for outer hexagon constraint
        # The outer hexagon has side length S, so apothem A_outer = S * APOTHEM.
        # For each inner hexagon i and each outer edge normal n_k,
        # we need: p_i . n_k + h_i(n_k) <= A_outer
        # where h_i(n_k) is the support function of hexagon i in direction n_k.
        # Penalty: sum of max(0, p_i . n_k + h_i(n_k) - A_outer)^2
        penalty_outer = 0.0
        for i in range(N):
            for k in range(6):
                n_k = EDGE_NORMALS[k]
                h_i_k = support_function(angles[i], n_k)
                proj = pts[i, 0] * n_k[0] + pts[i, 1] * n_k[1] + h_i_k
                A_outer = S * APOTHEM
                violation = proj - A_outer
                if violation > 0:
                    penalty_outer += violation * violation
        
        # Penalty for non-overlap
        # For each pair (i, j), i < j, compute the max gap over all 12 directions.
        # If max gap < 0, add a penalty.
        penalty_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                # Compute the 12 edge normals of the two hexagons
                # Hexagon i has 6 edge normals: R(theta_i) EDGE_NORMALS
                # Hexagon j has 6 edge normals: R(theta_j) EDGE_NORMALS
                # Total 12 directions
                max_gap = -np.inf
                for k in range(6):
                    # Direction n from hexagon i
                    n_i = np.array([
                        math.cos(angles[i]) * EDGE_NORMALS[k, 0] - math.sin(angles[i]) * EDGE_NORMALS[k, 1],
                        math.sin(angles[i]) * EDGE_NORMALS[k, 0] + math.cos(angles[i]) * EDGE_NORMALS[k, 1]
                    ])
                    gap_i = compute_gap(pts[i], angles[i], pts[j], angles[j], n_i)
                    if gap_i > max_gap:
                        max_gap = gap_i
                    
                    # Direction -n from hexagon i (which is the same as n from hexagon j in some sense)
                    # Actually, we need to check both n and -n for each edge normal.
                    # But the 12 directions are the 6 edge normals of hexagon i and the 6 edge normals of hexagon j.
                    # So we don't need to check -n separately.
                
                for k in range(6):
                    # Direction n from hexagon j
                    n_j = np.array([
                        math.cos(angles[j]) * EDGE_NORMALS[k, 0] - math.sin(angles[j]) * EDGE_NORMALS[k, 1],
                        math.sin(angles[j]) * EDGE_NORMALS[k, 0] + math.cos(angles[j]) * EDGE_NORMALS[k, 1]
                    ])
                    gap_j = compute_gap(pts[i], angles[i], pts[j], angles[j], n_j)
                    if gap_j > max_gap:
                        max_gap = gap_j
                
                # If max_gap < 0, the hexagons overlap
                if max_gap < 0:
                    penalty_overlap += (-max_gap) ** 2
        
        total_penalty = 100.0 * (penalty_outer + penalty_overlap)
        return obj + total_penalty

    # Initial parameters
    init_params = np.zeros(3*N + 1)
    for i in range(N):
        init_params[3*i] = initial_pts[i, 0]
        init_params[3*i+1] = initial_pts[i, 1]
        init_params[3*i+2] = 0.0
    init_params[3*N] = S0

    # Bounds
    # Positions: [-10, 10]
    # Angles: [-pi, pi]
    # S: [0, 10]
    bounds = []
    for i in range(N):
        bounds.append((-10.0, 10.0))
        bounds.append((-10.0, 10.0))
        bounds.append((-math.pi, math.pi))
    bounds.append((0.0, 10.0))

    # Optimize
    best_result = None
    best_obj = float('inf')
    
    # Use multiple starting points
    start_points = [init_params]
    # Add some perturbed starting points
    for _ in range(5):
        perturbed = init_params.copy()
        for i in range(N):
            perturbed[3*i] += np.random.uniform(-0.1, 0.1)
            perturbed[3*i+1] += np.random.uniform(-0.1, 0.1)
            perturbed[3*i+2] += np.random.uniform(-0.1, 0.1)
        start_points.append(perturbed)

    for start in start_points:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        try:
            result = minimize(objective, start, method='Nelder-Mead', bounds=bounds,
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result
        except Exception:
            pass

    if best_result is None:
        best_result = result

    # Extract the best solution
    params = best_result.x
    pts = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        pts[i, 0] = params[3*i]
        pts[i, 1] = params[3*i+1]
        angles[i] = params[3*i+2]
    S = params[3*N]

    # Compute the center of the outer hexagon
    # The outer hexagon should be centered at the centroid of the inner hexagons?
    # Actually, the outer hexagon's center is not necessarily the centroid of the inner hexagons.
    # We need to find the center (cx, cy) such that the outer hexagon contains all inner hexagons.
    # The outer hexagon is defined by 6 half-planes: n_k . (x - c) <= A_outer, where c is the center.
    # So we need: n_k . (x - c) <= A_outer for all k and all x in the inner hexagons.
    # This is equivalent to: n_k . x - n_k . c <= A_outer for all k and all x in the inner hexagons.
    # So: n_k . c >= n_k . x - A_outer for all k and all x in the inner hexagons.
    # The most restrictive constraint for each k is: n_k . c >= max_x (n_k . x) - A_outer
    # where x ranges over all points in all inner hexagons.
    # max_x (n_k . x) = max_i (p_i . n_k + h_i(n_k))
    # So: n_k . c >= max_i (p_i . n_k + h_i(n_k)) - A_outer
    # This is a system of 6 linear inequalities in c.
    # We can solve for c using linear programming.
    
    # Let's use linprog to find c.
    from scipy.optimize import linprog
    
    A_ub = []
    b_ub = []
    for k in range(6):
        n_k = EDGE_NORMALS[k]
        max_proj = 0.0
        for i in range(N):
            h_i_k = support_function(angles[i], n_k)
            proj = pts[i, 0] * n_k[0] + pts[i, 1] * n_k[1] + h_i_k
            if proj > max_proj:
                max_proj = proj
        # n_k . c >= max_proj - A_outer
        # -n_k . c <= A_outer - max_proj
        A_ub.append([-n_k[0], -n_k[1]])
        b_ub.append(S * APOTHEM - max_proj)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Also, we need to consider the opposite directions?
    # The outer hexagon is defined by 6 half-planes, one for each edge normal.
    # The half-planes are: n_k . (x - c) <= A_outer for k=0..5.
    # This is equivalent to: n_k . x - n_k . c <= A_outer.
    # So: -n_k . c <= A_outer - n_k . x.
    # The most restrictive constraint is: -n_k . c <= A_outer - max_x (n_k . x)
    # So: -n_k . c <= A_outer - max_i (p_i . n_k + h_i(n_k))
    # This is what we have above.
    
    # But we also need to consider the other 6 directions?
    # No, the 6 edge normals define the 6 edges of the hexagon.
    # The hexagon is the intersection of the 6 half-planes.
    # So we only need these 6 constraints.
    
    # Solve for c
    c_obj = np.array([0.0, 0.0])
    res_c = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None)], method='highs')
    
    if res_c.status == 0:
        cx, cy = res_c.x
    else:
        # Fallback: use the centroid
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])

    # Compute the actual S needed
    # For the found center (cx, cy), compute the required S.
    max_proj = 0.0
    for k in range(6):
        n_k = EDGE_NORMALS[k]
        for i in range(N):
            h_i_k = support_function(angles[i], n_k)
            proj = (pts[i, 0] - cx) * n_k[0] + (pts[i, 1] - cy) * n_k[1] + h_i_k
            if proj > max_proj:
                max_proj = proj
    
    A_outer = max_proj
    S_actual = A_outer / APOTHEM

    # Compute the rotation angle of the outer hexagon
    # The outer hexagon's edge normals are at angles 30°, 90°, 150°, 210°, 270°, 330°.
    # The rotation angle is 0.
    rotation = 0.0

    # Prepare the output
    inner = []
    for i in range(N):
        inner.append([float(pts[i, 0]), float(pts[i, 1]), float(math.degrees(angles[i]))])
    
    center = [float(cx), float(cy)]
    side = float(S_actual)
    
    return inner, center, side, rotation
