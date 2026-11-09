# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize, differential_evolution

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _is_valid(inner, outer, s, tol=1e-6, edge_tol=1e-8):
    """Check validity of a packing configuration."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < math.sqrt(12) - 1e-9:
        return False
    
    ocx, ocy, oang = outer[0], outer[1], outer[2]
    outer_verts = _hex_vertices(ocx, ocy, s, oang)
    
    # Check outer vertices are distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    # Check inner hexagons have unit edges
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1.0) > edge_tol:
                return False
    
    # Check separation between inner hexagons using SAT
    for i in range(12):
        for j in range(i+1, 12):
            v1 = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            
            normals = []
            for verts in [v1, v2]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k+1)%6]
                    edge = p2 - p1
                    n = np.array([-edge[1], edge[0]])
                    n_norm = np.linalg.norm(n)
                    if n_norm > 1e-12:
                        n = n / n_norm
                        normals.append(n)
            
            separated = False
            for n in normals:
                proj1 = np.dot(v1, n)
                proj2 = np.dot(v2, n)
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                gap = min(max1, max2) - max(min1, min2)
                if gap > -tol:
                    separated = True
                    break
            
            if not separated:
                return False
    
    # Check inner vertices inside outer hexagon
    for i in range(12):
        verts = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for v in verts:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                edge = p2 - p1
                rel = v - p1
                cross = edge[0] * rel[1] - edge[1] * rel[0]
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True

def _compute_min_s(inner, outer_angle_deg):
    """Compute the minimum outer side length needed to contain all inner hexagons.
    Returns (s, outer_center) or (inf, None) if invalid."""
    oang = np.deg2rad(outer_angle_deg)
    
    # For a hexagon with center (cx, cy) and angle a, the support function in direction u is:
    # h(u) = max over vertices of dot(vertex, u)
    # For a regular hexagon with side d, h(u) = d * max(cos(a + 60k - theta)) where theta = atan2(u_y, u_x)
    # Actually, for a regular hexagon, the support in any direction is d * cos(min angle to nearest vertex normal)
    # More precisely: h(u) = d * max_{k=0..5} cos(angle(u) - (a + 60k))
    
    # The outer hexagon with center C, side s, angle A contains a point P iff
    # for all 6 edge directions, the point is on the correct side.
    # Equivalent: max over 6 support directions of the outer hexagon >= max over all inner vertices
    
    # The support function of the outer hexagon in direction u (from center) is:
    # h_outer(u) = s * max_{k=0..5} cos(angle(u) - (A + 60k))
    # Wait, that's the support of the hexagon centered at origin.
    # If outer center is C, then a point P is inside iff |P - C| in support sense <= s
    
    # Actually, for a convex polygon, P is inside iff for all edge normals n_i, dot(P - C, n_i) <= h(n_i)
    # where h(n_i) is the support in direction n_i.
    
    # For a regular hexagon with side s and angle A:
    # The 6 edge normals are at angles A + 60k + 90 (or A + 60k depending on convention)
    # Let's use the vertex-based check which is more direct.
    
    # Alternative: For each inner vertex v, we need v to be inside the outer hexagon.
    # The outer hexagon with center C and side s and angle A contains v iff:
    # For all 6 edges, cross(edge, v - p1) >= 0 (for CCW)
    
    # This is complex. Let me use a different approach:
    # The minimum s such that all points fit in a hexagon of side s centered at C with angle A.
    # For a point P, the minimum s needed (with center at C and angle A) is:
    # s_min(P) = max over the 6 "half-plane constraints"
    
    # Actually, let me just do a direct geometric computation.
    # For a regular hexagon with center C, side s, angle A:
    # Vertex i is at C + s * (cos(A + 60i), sin(A + 60i))
    # The hexagon is the intersection of 6 half-planes.
    # Edge i goes from vertex i to vertex i+1.
    # The inward normal of edge i points toward the center.
    
    # Let me compute the required s for a given center and angle by checking all inner vertices.
    
    # For a point P to be inside a hexagon with center C, side s, angle A:
    # Let Q = P - C
    # The hexagon (centered at origin, side s, angle A) contains Q iff
    # for all k=0..5, the projection of Q onto the outward normal of edge k is <= s * cos(30°) = s * sqrt(3)/2
    
    # Wait, for a regular hexagon with side s, the apothem (distance from center to edge) is s * sqrt(3)/2.
    # The 6 edge outward normals are at angles A + 60k + 90° (if vertices are at A + 60k)
    # Actually, if vertices are at angles A + 60k from center, then edges are between consecutive vertices.
    # The outward normal of edge between vertex k and k+1 is at angle A + 60k + 30° + 90° = A + 60k + 120°? 
    # Let me just compute it directly.
    
    # Simpler approach: binary search on s for a given center and angle.
    # But that's slow. Let me compute directly.
    
    # For a regular hexagon centered at origin with side s and angle A (first vertex at angle A):
    # Vertices: V_k = s * (cos(A + 60k), sin(A + 60k)) for k=0..5
    # Edge k: from V_k to V_{k+1}
    # The line containing edge k: we can compute the distance from origin to this line.
    # For a regular hexagon, this distance is the apothem = s * sqrt(3)/2.
    # The outward normal direction for edge k is the direction from center to the midpoint of edge k.
    # Midpoint of edge k: (V_k + V_{k+1})/2
    # Direction: normalize(V_k + V_{k+1})
    
    # So for a point Q to be inside: dot(Q, n_k) <= s * sqrt(3)/2 for all k,
    # where n_k is the unit outward normal of edge k.
    
    # But n_k depends on A. Let me compute n_k for given A.
    
    # V_k = s*(cos(A+60k), sin(A+60k))
    # V_k + V_{k+1} = s*(cos(A+60k)+cos(A+60(k+1)), sin(A+60k)+sin(A+60(k+1)))
    # = s*2*cos(30°)*(cos(A+60k+30°), sin(A+60k+30°))
    # = s*sqrt(3)*(cos(A+60k+30°), sin(A+60k+30°))
    # So n_k = (cos(A+60k+30°), sin(A+60k+30°))
    
    # Therefore, Q is inside iff for all k: dot(Q, n_k) <= s*sqrt(3)/2
    # i.e., s >= (2/sqrt(3)) * max_k dot(Q, n_k)
    
    # So for a given center C and angle A, the minimum s is:
    # s_min = (2/sqrt(3)) * max over all inner vertices v of max_k dot(v - C, n_k(A))
    
    # where n_k(A) = (cos(A + 60k + 30°), sin(A + 60k + 30°))
    
    all_verts = []
    for i in range(12):
        verts = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # n_k angles in radians
    n_angles = oang + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
    normals = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
    
    # For each vertex v, compute max_k dot(v - C, n_k)
    # We want to minimize over C: max_v max_k dot(v - C, n_k)
    # = max_v max_k (dot(v, n_k) - dot(C, n_k))
    # = max_k (max_v dot(v, n_k) - dot(C, n_k))
    
    # So s_min(C) = (2/sqrt(3)) * max_k (max_v dot(v, n_k) - dot(C, n_k))
    
    # To minimize this over C, we can use the fact that it's a convex function of C.
    # The optimal C equalizes the max over k.
    
    # For now, let's just compute for C = (0,0) and optimize later.
    # Actually, let me just return the s for C=(0,0) and let the optimizer handle C.
    
    # But the function signature returns (s, outer_center). Let me just compute s for given C=(0,0).
    # Actually, I'll restructure: the optimizer will optimize C and angles, and compute s.
    
    # For C = (0,0):
    # s_min = (2/sqrt(3)) * max_k max_v dot(v, n_k)
    
    dots = all_verts @ normals.T  # shape (72, 6)
    max_dots = dots.max(axis=0)  # shape (6,)
    s_min = (2.0 / math.sqrt(3.0)) * max_dots.max()
    
    return s_min, np.array([0.0, 0.0])

def _objective(params):
    """Objective: minimize s. Params: 24 center coords + 12 angles + 1 outer angle = 37 params."""
    cx_cy = params[:24].reshape(12, 2)
    inner_angles = params[24:36]
    outer_angle = params[36]
    
    inner = np.zeros((12, 3))
    inner[:, 0] = cx_cy[:, 0]
    inner[:, 1] = cx_cy[:, 1]
    inner[:, 2] = inner_angles
    
    s, outer_center = _compute_min_s(inner, outer_angle)
    
    if not np.isfinite(s) or s < math.sqrt(12) - 1e-9:
        return s + 100.0
    
    outer = np.array([outer_center[0], outer_center[1], outer_angle])
    
    if not _is_valid(inner, outer, s):
        return s + 100.0
    
    return s

def hexagon_packing_12():
    start_time = time.time()
    deadline = 620.0
    
    best_s = float('inf')
    best_inner = None
    best_outer = None
    
    # Initial configuration: hexagonal packing
    # 12 hexagons: center + 6 ring + 5 more
    # Let's use a symmetric initial guess
    # Hexagonal lattice spacing for touching hexagons: 2*sqrt(3) ≈ 3.464
    
    # Initial guess: 4-4-4 rows or similar
    # Let me use a simple grid and then optimize
    
    # Actually, let me start with a known good configuration
    # 12 hexagons in a hexagon: 
    # Row 0 (center): 1
    # Ring 1: 6
    # Ring 2 partial: 5
    # Total: 12
    
    # Positions in hex lattice (axial coords) -> Cartesian
    # Hex lattice: each hex at distance 2*sqrt(3) from neighbors
    # Axial to Cartesian: x = 2*sqrt(3)*q + sqrt(3)*r, y = 3*r/2 * ... 
    # Actually for hexagons with side 1, center-to-center distance for touching is 2*sqrt(3)
    
    # Let me just use a simple initial guess and let DE optimize
    # Symmetric initial guess:
    # Place hexagons in a roughly circular pattern
    
    r = 2 * math.sqrt(3)  # center-to-center distance for touching
    
    # 12 positions: center + 11 around
    # Or use a 3x4 rectangular-ish arrangement
    
    # Let me use a hexagonal number arrangement:
    # Positions (x, y) for a hexagonal packing:
    # Level 0: (0, 0)
    # Level 1: 6 positions at distance r
    # Level 2: 12 positions at distance 2r, but we only need 5 more
    
    initial_positions = []
    # Center
    initial_positions.append((0, 0))
    # Ring 1: 6 hexagons
    for k in range(6):
        angle = math.pi / 3 * k
        initial_positions.append((r * math.cos(angle), r * math.sin(angle)))
    # Ring 2: 5 more hexagons (out of 12 possible)
    # Place them at alternating positions in ring 2
    for k in range(0, 12, 2):
        if len(initial_positions) >= 12:
            break
        angle = math.pi / 3 * k + math.pi / 6
        initial_positions.append((2 * r * math.cos(angle), 2 * r * math.sin(angle)))
    
    # Ensure we have exactly 12
    while len(initial_positions) < 12:
        initial_positions.append((0, 0))
    initial_positions = initial_positions[:12]
    
    # Initial angles: 0 for all inner, 0 for outer
    initial_params = np.zeros(37)
    for i in range(12):
        initial_params[2*i] = initial_positions[i][0]
        initial_params[2*i+1] = initial_positions[i][1]
    # Angles default to 0
    
    # Bounds
    bounds = [(-8, 8)] * 24  # centers
    bounds += [(-180, 180)] * 12  # inner angles
    bounds += [(-30, 30)]  # outer angle (symmetry)
    
    # First, try a quick DE
    def callback(xk, convergence):
        nonlocal best_s, best_inner, best_outer
        val = _objective(xk)
        if val < best_s:
            cx_cy = xk[:24].reshape(12, 2)
            inner_angles = xk[24:36]
            outer_angle = xk[36]
            inner = np.zeros((12, 3))
            inner[:, 0] = cx_cy[:, 0]
            inner[:, 1] = cx_cy[:, 1]
            inner[:, 2] = inner_angles
            s, oc = _compute_min_s(inner, outer_angle)
            outer = np.array([oc[0], oc[1], outer_angle])
            if _is_valid(inner, outer, s):
                best_s = s
                best_inner = inner.copy()
                best_outer = outer.copy()
        return time.time() > deadline
    
    try:
        res = differential_evolution(
            _objective,
            bounds,
            maxiter=150,
            tol=1e-7,
            mut=(0.5, 1.5),
            recomb=0.7,
            seed=42,
            polish=False,
            callback=callback
        )
        if res.fun < best_s:
            val = _objective(res.x)
            if val < best_s:
                cx_cy = res.x[:24].reshape(12, 2)
                inner_angles = res.x[24:36]
                outer_angle = res.x[36]
                inner = np.zeros((12, 3))
                inner[:, 0] = cx_cy[:, 0]
                inner[:, 1] = cx_cy[:, 1]
                inner[:, 2] = inner_angles
                s, oc = _compute_min_s(inner, outer_angle)
                outer = np.array([oc[0], oc[1], outer_angle])
                if _is_valid(inner, outer, s):
                    best_s = s
                    best_inner = inner.copy()
                    best_outer = outer.copy()
    except Exception as e:
        print(f"DE failed: {e}")
    
    # Local refinement with Nelder-Mead
    if best_inner is not None and time.time() < deadline - 60:
        best_x = np.concatenate([best_inner[:, :2].flatten(), best_inner[:, 2], [best_outer[2]]])
        
        def local_obj(x):
            return _objective(x)
        
        try:
            res2 = minimize(
                local_obj,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res2.fun < best_s:
                cx_cy = res2.x[:24].reshape(12, 2)
                inner_angles = res2.x[24:36]
                outer_angle = res2.x[36]
                inner = np.zeros((12, 3))
                inner[:, 0] = cx_cy[:, 0]
                inner[:, 1] = cx_cy[:, 1]
                inner[:, 2] = inner_angles
                s, oc = _compute_min_s(inner, outer_angle)
                outer = np.array([oc[0], oc[1], outer_angle])
                if _is_valid(inner, outer, s):
                    best_s = s
                    best_inner = inner.copy()
                    best_outer = outer.copy()
        except Exception as e:
            print(f"Nelder-Mead failed: {e}")
    
    # Second DE pass with tighter bounds
    if best_inner is not None and time.time() < deadline - 30:
        best_x = np.concatenate([best_inner[:, :2].flatten(), best_inner[:, 2], [best_outer[2]]])
        
        tight_bounds = []
        for i in range(24):
            tight_bounds.append((best_x[i] - 0.5, best_x[i] + 0.5))
        for i in range(12):
            tight_bounds.append((best_x[24+i] - 30, best_x[24+i] + 30))
        tight_bounds.append((best_x[36] - 15, best_x[36] + 15))
        
        try:
            res3 = differential_evolution(
                _objective,
                tight_bounds,
                maxiter=50,
                tol=1e-9,
                mut=(0.3, 1.0),
                recomb=0.8,
                seed=123,
                polish=False,
                callback=callback
            )
            if res3.fun < best_s:
                val = _objective(res3.x)
                if val < best_s:
                    cx_cy = res3.x[:24].reshape(12, 2)
                    inner_angles = res3.x[24:36]
                    outer_angle = res3.x[36]
                    inner = np.zeros((12, 3))
                    inner[:, 0] = cx_cy[:, 0]
                    inner[:, 1] = cx_cy[:, 1]
                    inner[:, 2] = inner_angles
                    s, oc = _compute_min_s(inner, outer_angle)
                    outer = np.array([oc[0], oc[1], outer_angle])
                    if _is_valid(inner, outer, s):
                        best_s = s
                        best_inner = inner.copy()
                        best_outer = outer.copy()
        except Exception as e:
            print(f"Refined DE failed: {e}")
    
    # Final local refinement
    if best_inner is not None and time.time() < deadline - 10:
        best_x = np.concatenate([best_inner[:, :2].flatten(), best_inner[:, 2], [best_outer[2]]])
        
        def local_obj(x):
            return _objective(x)
        
        try:
            res4 = minimize(
                local_obj,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if res4.fun < best_s:
                cx_cy = res4.x[:24].reshape(12, 2)
                inner_angles = res4.x[24:36]
                outer_angle = res4.x[36]
                inner = np.zeros((12, 3))
                inner[:, 0] = cx_cy[:, 0]
                inner[:, 1] = cx_cy[:, 1]
                inner[:, 2] = inner_angles
                s, oc = _compute_min_s(inner, outer_angle)
                outer = np.array([oc[0], oc[1], outer_angle])
                if _is_valid(inner, outer, s):
                    best_s = s
                    best_inner = inner.copy()
                    best_outer = outer.copy()
        except Exception as e:
            print(f"Final Nelder-Mead failed: {e}")
    
    # Fallback
    if best_inner is None:
        # Use the seed-like solution
        inner = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.165, 0],
                [1.25, 2.165, 0],
                [-1.25, -2.165, 0],
                [1.25, -2.165, 0],
                [-3.75, 2.165, 0],
                [3.75, 2.165, 0],
                [-3.75, -2.165, 0],
                [3.75, -2.165, 0],
                [0, -4, 0],
            ]
        )
        outer = np.array([0, 0, 0])
        s = 8.0
        return inner, outer, s
    
    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
