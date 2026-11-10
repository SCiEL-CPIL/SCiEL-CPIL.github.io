# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import linprog

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + math.radians(60 * i)
        verts.append((cx + d * math.cos(theta), cy + d * math.sin(theta)))
    return np.array(verts)

def _hex_edge_normals(angle_deg):
    """
    Return 6 unit outward normals to the edges of a regular hexagon with given angle.
    For a hexagon with vertices at angles angle + 60*i, the edge between vertex i and i+1
    has its outward normal pointing at angle + 30 + 60*i degrees.
    """
    a = math.radians(angle_deg)
    normals = []
    for i in range(6):
        theta = a + math.radians(30 + 60 * i)
        normals.append((math.cos(theta), math.sin(theta)))
    return np.array(normals)

def _dodecagon_normals(angle1, angle2):
    """
    Compute the 12 outward normals of the Minkowski difference (sum) of two unit hexagons
    with angles angle1 and angle2. The Minkowski sum of two regular hexagons with side 1
    is a regular dodecagon (12-gon) if the angles are generic, or a hexagon if aligned.
    
    Actually, the Minkowski sum of two regular hexagons with side 1 and angles a1, a2
    has edge normals that are the union of the 6 normals of each hexagon, sorted by angle.
    The resulting shape has at most 12 edges.
    
    For the non-overlap constraint: two hexagons with centers c1, c2 and angles a1, a2
    do not overlap if and only if for every edge normal n of the Minkowski sum,
    dot(c2 - c1, n) >= -h(n) where h(n) is the support function... 
    
    Wait, let me think more carefully. Two convex polygons P and Q (centered at c1, c2)
    do not overlap iff there exists a direction n such that max_{p in P} dot(p, n) < min_{q in Q} dot(q, n).
    Equivalently, for the Minkowski difference P - Q = {p - q : p in P, q in Q}, 
    the origin is not in P - Q + (c2 - c1) ... 
    
    Actually, two convex sets A and B are disjoint iff there exists a separating hyperplane.
    For two convex polygons, the separating axis test checks all edge normals of both polygons.
    
    For two regular hexagons with side 1 and angles a1, a2, the relevant normals are:
    - The 6 edge normals of hexagon 1: n_i = (cos(a1 + 30 + 60*i), sin(a1 + 30 + 60*i))
    - The 6 edge normals of hexagon 2: m_j = (cos(a2 + 30 + 60*j), sin(a2 + 30 + 60*j))
    
    The non-overlap constraint is: for all 12 normals, the projection intervals don't overlap.
    For normal n, hexagon 1 (center c1, side 1) projects to [dot(c1,n) - r1(n), dot(c1,n) + r1(n)]
    where r1(n) is the support function of a unit hexagon in direction n.
    Similarly for hexagon 2.
    
    The gap in direction n is: (dot(c2,n) - r2(n)) - (dot(c1,n) + r1(n)) >= 0
    i.e., dot(c2 - c1, n) >= r1(n) + r2(n)
    
    The support function of a regular hexagon with side 1 and angle a in direction n (unit vector at angle phi) is:
    h(n) = max_i dot(vertex_i, n) where vertex_i are the 6 vertices.
    
    For a unit hexagon centered at origin with angle a, vertices are at (cos(a+60i), sin(a+60i)).
    h(phi) = max_i cos(phi - a - 60*i) = max_i cos(phi - a - 60*i)
    
    This is a piecewise linear function. For a regular hexagon, the support function is:
    h(phi) = cos(phi - a - 60*k) where k is chosen so that phi - a - 60*k is in [-30, 30] degrees.
    So h(phi) = cos(delta) where delta is the angular distance from phi to the nearest vertex direction, capped at 30 degrees.
    
    Actually, h(phi) for a regular hexagon with circumradius 1 (side 1) is:
    Let delta = ((phi - a) mod 60) - 30, then h = cos(delta) if |delta| <= 30, which is always true.
    Wait: the vertices are at angles a, a+60, ..., a+300. The support in direction phi is the max dot product
    with a vertex, which is cos(phi - a - 60*k) for the k that minimizes |phi - a - 60*k|.
    The minimum angular difference is at most 30 degrees.
    So h(phi) = cos(min_{k} |phi - a - 60*k|) where the angle is wrapped to [-30, 30].
    
    Let me just compute it numerically for each direction.
    """
    normals1 = _hex_edge_normals(angle1)
    normals2 = _hex_edge_normals(angle2)
    
    # For each normal, compute the support function of a unit hexagon
    def support_hex(angle_hex, n):
        """Support function of unit hexagon with given angle in direction n."""
        verts = _hex_vertices(0, 0, 1.0, angle_hex)
        return np.max(verts @ n)
    
    # Collect all 12 normals and their required gaps
    all_normals = []
    all_gaps = []
    
    for n in normals1:
        r1 = support_hex(angle1, n)
        r2 = support_hex(angle2, n)
        all_normals.append(n)
        all_gaps.append(r1 + r2)
    
    for n in normals2:
        r1 = support_hex(angle1, n)
        r2 = support_hex(angle2, n)
        all_normals.append(n)
        all_gaps.append(r1 + r2)
    
    return np.array(all_normals), np.array(all_gaps)

def _solve_lp_for_fixed_angles(inner_angles, outer_angle, time_budget=30.0):
    """
    Given fixed angles for all 12 inner hexagons and the outer hexagon,
    solve an LP to find the optimal centers and minimum outer side length s.
    
    Variables: 12*2 (inner centers) + 1 (s) + 66*12 (aux for SAT) = 817 variables
    x = [x1, y1, ..., x12, y12, s, t_1_2_0, t_1_2_1, ..., t_1_2_11, t_1_3_0, ...]
    
    Objective: minimize s
    Constraints:
    1. Non-overlap: for each pair (i,j), SAT minimum gap >= 0.
       Linearized with auxiliary variable t_k >= dot(cj-ci, n_k) - gap_k for each normal k,
       and t_k >= 0. This ensures the minimum gap over all axes is >= 0.
    2. Containment: for each inner hexagon i and each of the 6 outer normals,
       dot(ci, n_outer) + h_inner(n_outer) <= s * h_outer_unit(n_outer)
    """
    start_time = time.time()
    
    n_inner = 12
    n_pairs = n_inner * (n_inner - 1) // 2  # 66 pairs
    n_aux_per_pair = 12
    n_vars = n_inner * 2 + 1 + n_pairs * n_aux_per_pair
    
    # Objective: minimize s
    c_obj = np.zeros(n_vars)
    c_obj[-1] = 1.0  # minimize s
    
    A_ub = []
    b_ub = []
    
    # Precompute support functions
    outer_normals = _hex_edge_normals(outer_angle)
    
    # Support of unit outer hexagon in each outer normal direction
    outer_support = np.array([support_hex_unit(outer_angle, n) for n in outer_normals])
    
    # For each inner hexagon, precompute support in each outer normal direction
    inner_supports = []
    for i in range(n_inner):
        sup = np.array([support_hex_unit(inner_angles[i], n) for n in outer_normals])
        inner_supports.append(sup)
    
    # Containment constraints: for each inner i, each outer normal k
    for i in range(n_inner):
        for k in range(6):
            n = outer_normals[k]
            # dot(ci, n) - s * outer_support[k] <= -inner_supports[i][k]
            row = np.zeros(n_vars)
            row[2*i] = n[0]
            row[2*i+1] = n[1]
            row[-1] = -outer_support[k]
            A_ub.append(row)
            b_ub.append(-inner_supports[i][k])
    
    # Non-overlap constraints: for each pair (i, j), i < j
    pair_idx = 0
    for i in range(n_inner):
        for j in range(i+1, n_inner):
            normals, gaps = _dodecagon_normals(inner_angles[i], inner_angles[j])
            aux_start = n_inner * 2 + 1 + pair_idx * n_aux_per_pair
            
            for k in range(len(normals)):
                n = normals[k]
                gap = gaps[k]
                # t_k >= dot(cj - ci, n) - gap
                # => dot(ci, n) - dot(cj, n) - t_k <= -gap
                row = np.zeros(n_vars)
                row[2*i] = n[0]
                row[2*i+1] = n[1]
                row[2*j] = -n[0]
                row[2*j+1] = -n[1]
                row[aux_start + k] = -1.0
                A_ub.append(row)
                b_ub.append(-gap)
            
            # t_k >= 0 for all k (ensures min gap >= 0)
            for k in range(len(normals)):
                row = np.zeros(n_vars)
                row[aux_start + k] = -1.0
                A_ub.append(row)
                b_ub.append(0.0)
            
            pair_idx += 1
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: s >= sqrt(12), centers are free, aux vars >= 0
    bounds = [(None, None)] * (n_inner * 2) + [(math.sqrt(12), None)] + [(0, None)] * (n_pairs * n_aux_per_pair)
    
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        centers = result.x[:n_inner*2].reshape(n_inner, 2)
        s = result.x[-1]
        return centers, s
    else:
        return None, None

def support_hex_unit(angle_deg, n):
    """Support function of a unit regular hexagon (side 1) with given angle in unit direction n."""
    verts = _hex_vertices(0, 0, 1.0, angle_deg)
    return np.max(verts @ n)

def _refine_with_lp(inner_angles, outer_angle, time_budget=60.0):
    """
    Given fixed angles, solve LP and return the configuration.
    """
    start_time = time.time()
    centers, s = _solve_lp_for_fixed_angles(inner_angles, outer_angle, time_budget)
    if centers is not None:
        inner = np.zeros((12, 3))
        inner[:, 0] = centers[:, 0]
        inner[:, 1] = centers[:, 1]
        inner[:, 2] = inner_angles
        outer = np.array([0.0, 0.0, outer_angle])
        return inner, outer, s
    return None, None, None

def _honeycomb_seed_angles():
    """Generate a good initial set of angles for the LP approach."""
    # Start with all hexagons at angle 0
    inner_angles = np.zeros(12)
    outer_angle = 0.0
    return inner_angles, outer_angle

def _random_angles(seed=42):
    """Generate random angles for exploration."""
    rng = np.random.default_rng(seed)
    inner_angles = rng.uniform(0, 60, size=12)
    outer_angle = rng.uniform(0, 60)
    return inner_angles, outer_angle

def _check_validity(inner, outer, s, tol=1e-8):
    """Check if the packing is valid."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False, "non-finite values"
    if s < math.sqrt(12) - 1e-9:
        return False, "s too small"
    
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    
    # Check outer vertices distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False, "outer vertices not distinct"
    
    # Check inner hexagons don't overlap
    inner_verts = []
    for i in range(12):
        inner_verts.append(_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]))
    
    for i in range(12):
        for j in range(i+1, 12):
            # SAT test
            min_gap = float('inf')
            for verts in [inner_verts[i], inner_verts[j]]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm_len = np.linalg.norm(normal)
                    if norm_len < 1e-12:
                        continue
                    normal = normal / norm_len
                    proj1 = inner_verts[i] @ normal
                    proj2 = inner_verts[j] @ normal
                    gap = max(np.min(proj2) - np.max(proj1), np.min(proj1) - np.max(proj2))
                    if gap < min_gap:
                        min_gap = gap
            if min_gap < -1e-6:
                return False, f"inner {i} and {j} overlap"
    
    # Check inner vertices inside outer hexagon
    for i in range(12):
        for v in inner_verts[i]:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1) % 6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < -1e-6:
                    return False, f"inner {i} vertex outside outer"
    
    return True, "ok"

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses LP-based optimization with angle exploration.
    """
    deadline = time.time() + 620.0
    
    best_inner = None
    best_outer = None
    best_s = float('inf')
    
    # Try several angle configurations
    angle_configs = []
    
    # Config 1: All aligned
    angle_configs.append((np.zeros(12), 0.0))
    
    # Config 2: Alternating angles
    angles_alt = np.array([0.0, 30.0] * 6)
    angle_configs.append((angles_alt, 0.0))
    
    # Config 3: Random angles
    for seed in range(10):
        angle_configs.append(_random_angles(seed))
    
    # Config 4: Outer angle variations with aligned inner
    for oa in [0, 15, 30, 45]:
        angle_configs.append((np.zeros(12), float(oa)))
    
    # Try each configuration
    for inner_angles, outer_angle in angle_configs:
        if time.time() > deadline - 10:
            break
        
        remaining = min(50.0, deadline - time.time() - 10)
        if remaining < 5:
            break
        
        inner, outer, s = _refine_with_lp(inner_angles, outer_angle, time_budget=remaining)
        
        if inner is not None:
            is_valid, msg = _check_validity(inner, outer, s)
            if is_valid and s < best_s:
                best_s = s
                best_inner = inner.copy()
                best_outer = outer.copy()
    
    # If we found a good solution, do a local angle refinement
    if best_inner is not None and time.time() < deadline - 30:
        # Try perturbing angles around the best solution
        base_inner_angles = best_inner[:, 2].copy()
        base_outer_angle = best_outer[2]
        
        for pert_seed in range(20):
            if time.time() > deadline - 15:
                break
            
            rng = np.random.default_rng(pert_seed + 100)
            perturbation = rng.uniform(-5, 5, size=12)
            new_inner_angles = base_inner_angles + perturbation
            new_outer_angle = base_outer_angle + rng.uniform(-5, 5)
            
            remaining = min(30.0, deadline - time.time() - 10)
            if remaining < 5:
                break
            
            inner, outer, s = _refine_with_lp(new_inner_angles, new_outer_angle, time_budget=remaining)
            
            if inner is not None:
                is_valid, msg = _check_validity(inner, outer, s)
                if is_valid and s < best_s:
                    best_s = s
                    best_inner = inner.copy()
                    best_outer = outer.copy()
    
    # Fine outer angle search around the best solution
    if best_inner is not None and time.time() < deadline - 20:
        base_inner_angles = best_inner[:, 2].copy()
        base_outer_angle = best_outer[2]
        
        # Sweep outer angle in fine steps
        for oa_deg in np.linspace(base_outer_angle - 10, base_outer_angle + 10, 41):
            if time.time() > deadline - 10:
                break
            
            remaining = min(20.0, deadline - time.time() - 10)
            if remaining < 5:
                break
            
            inner, outer, s = _refine_with_lp(base_inner_angles, float(oa_deg), time_budget=remaining)
            
            if inner is not None:
                is_valid, msg = _check_validity(inner, outer, s)
                if is_valid and s < best_s:
                    best_s = s
                    best_inner = inner.copy()
                    best_outer = outer.copy()
    
    # Fallback if no valid solution found
    if best_inner is None:
        # Use a simple honeycomb arrangement
        positions = []
        dx = 3.0
        dy = math.sqrt(3)
        for row in range(4):
            for col in range(3):
                x = col * dx + (row % 2) * dx/2
                y = row * dy
                positions.append((x, y, 0.0))
        
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        
        best_inner = np.array([[p[0]-cx, p[1]-cy, p[2]] for p in positions])
        best_outer = np.array([0.0, 0.0, 0.0])
        
        # Compute s
        outer_normals = _hex_edge_normals(0.0)
        max_s = 0.0
        for i in range(12):
            verts = _hex_vertices(best_inner[i,0], best_inner[i,1], 1.0, best_inner[i,2])
            for v in verts:
                p = np.array([v[0] - best_outer[0], v[1] - best_outer[1]])
                dots = outer_normals @ p
                s_needed = np.max(dots) * 2.0 / math.sqrt(3)
                if s_needed > max_s:
                    max_s = s_needed
        best_s = max(max_s, math.sqrt(12))
    
    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
