import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
D = math.sqrt(3) # Min distance between centers of aligned unit hexagons

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0 # Be safe

    # Vertices of a unit hexagon (pointy-top or flat-top doesn't matter for support function magnitude, 
    # but let's define standard flat-top: vertices at 0, 60, ... degrees)
    # Actually, let's use the standard "flat top" hexagon where sides are horizontal/vertical-ish?
    # Standard math: vertices at k * 60 degrees.
    hex_verts = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] 
        for i in range(6)
    ])

    # Support function of the unit hexagon in direction n
    # h(n) = max(v . n for v in hex_verts)
    # Since the hexagon is centered at origin and symmetric, h(n) = max(abs(v . n))
    # But for the bounding box calculation:
    # The condition for all vertices of hex i to be inside outer hex (centered 0, side S, angle theta) is:
    # max_k | (c_i + v_k) . n_j | <= S * sqrt(3)/2 for all j
    # This is equivalent to: c_i . n_j + max_k (v_k . n_j) <= S * sqrt(3)/2 
    # AND -c_i . n_j - max_k (v_k . n_j) <= S * sqrt(3)/2  (using min)
    # Actually, max_k | (c_i + v_k) . n_j | = | c_i . n_j | + max_k (v_k . n_j) is NOT correct in general.
    # It is | c_i . n_j + v_k . n_j |.
    # The maximum over k of | c_i . n_j + v_k . n_j | is NOT simply | c_i . n_j | + h(n_j).
    # However, since the hexagon is convex and contains the origin, the max projection of the shifted hexagon
    # in direction n is c_i . n + h_hex(n).
    # The min projection is c_i . n - h_hex(n) (by symmetry h(-n)=h(n)).
    # So the range of projections for hex i in direction n_j is [c_i . n_j - h(n_j), c_i . n_j + h(n_j)].
    # For the outer hexagon to contain it, we need the outer hexagon's "width" in direction n_j to cover this range.
    # The outer hexagon (centered at 0) has support function S * sqrt(3)/2 in all 6 normal directions?
    # No, the outer hexagon is a regular hexagon. Its support function in direction n is:
    # If n is a normal to a side, h_outer(n) = S * sqrt(3)/2.
    # If n is a vertex direction, h_outer(n) = S.
    # The condition is that the inner hexagon is inside the outer.
    # This is equivalent to: for all directions n, h_inner(n) <= h_outer(n) - c_i . n ... no.
    # The condition for a set A to be inside a set B (centered at 0) is:
    # h_A(n) - h_B(n) <= 0 for all n? No.
    # A is inside B iff h_A(n) <= h_B(n) for all n? No, that's for convex sets containing origin.
    # Here B is centered at 0. A is centered at c_i.
    # A is inside B iff h_A(n) - c_i . n <= h_B(n) for all n.
    # h_A(n) = c_i . n + h_unit(n).
    # So c_i . n + h_unit(n) - c_i . n = h_unit(n) <= h_B(n) for all n?
    # This implies the unit hexagon must fit inside the outer hexagon regardless of position?
    # That's not right. The position c_i matters.
    
    # Let's re-evaluate.
    # Point p is in outer hexagon if |p . n_j| <= S * sqrt(3)/2 for the 6 normals n_j?
    # This is true if the outer hexagon is defined by the intersection of half-planes defined by its 6 sides.
    # The normals to the sides are at angles 0, 60, ... 300 (if flat-top).
    # Let's fix the outer hexagon orientation to be 0 (flat-top).
    # Normals n_j = [cos(j*pi/3), sin(j*pi/3)].
    # Condition: max_{v in inner_i} | v . n_j | <= S * sqrt(3)/2 for all j.
    # max_{v in inner_i} | v . n_j | = max_{k} | (c_i + hex_verts[k]) . n_j |
    
    def calc_S(centers):
        # centers: (11, 2)
        # Compute max projection for all 6 directions
        # n_j for j=0..5
        angles = np.arange(6) * (math.pi / 3)
        normals = np.column_stack([np.cos(angles), np.sin(angles)]) # (6, 2)
        
        # all_verts: (66, 2)
        # centers: (11, 2) -> (11, 1, 2)
        # hex_verts: (6, 2) -> (1, 6, 2)
        all_verts = centers[:, np.newaxis, :] + hex_verts[np.newaxis, :, :]
        all_verts = all_verts.reshape(-1, 2) # (66, 2)
        
        # Projections: (66, 6)
        projs = np.abs(all_verts @ normals.T)
        
        # Max projection for each normal direction
        max_projs = np.max(projs, axis=0) # (6,)
        
        # S must satisfy S * sqrt(3)/2 >= max_projs[j] for all j
        # S >= max_j (max_projs[j]) / (sqrt(3)/2)
        max_proj = np.max(max_projs)
        return (2.0 / math.sqrt(3)) * max_proj

    def overlap_penalty(centers):
        # centers: (11, 2)
        # dist[i, j]
        # Use broadcasting
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Penalty for dist < D
        # We want penalty to be 0 if dist >= D
        # violation = max(0, D - dist)
        # Sum of squares
        violations = np.maximum(0, D - dists)
        # Exclude self (diagonal)
        np.fill_diagonal(violations, 0)
        return np.sum(violations ** 2)

    def objective(centers):
        S = calc_S(centers)
        pen = overlap_penalty(centers)
        return S + 100.0 * pen

    # Initial configurations
    def make_hex_cluster():
        pts = [(0.0, 0.0)]
        for i in range(6):
            a = i * math.pi / 3
            pts.append((D * math.cos(a), D * math.sin(a)))
        # 4 from ring 2
        # Ring 2 axial (q, r) with max(abs(q), abs(r), abs(-q-r)) == 2
        # Let's pick (2,0), (0,2), (-2,0), (0,-2) in axial?
        # Axial to Cartesian: x = D * (q + r/2), y = D * sqrt(3)/2 * r
        # (2,0) -> x=2D, y=0
        # (0,2) -> x=D, y=D*sqrt(3)
        # (-2,0) -> x=-2D, y=0
        # (0,-2) -> x=-D, y=-D*sqrt(3)
        ring2 = [
            (2*D, 0.0),
            (D, D*math.sqrt(3)),
            (-2*D, 0.0),
            (-D, -D*math.sqrt(3))
        ]
        pts.extend(ring2)
        return np.array(pts)

    def make_staggered_rows():
        # 2 rows of 5, 1 in middle
        # Row 1: y = 0, x = -2D, -D, 0, D, 2D
        # Row 2: y = D*sqrt(3), x = -1.5D, -0.5D, 0.5D, 1.5D, 2.5D? 
        # This might be too wide.
        # Let's try a tighter staggered pattern.
        # Just use random perturbation of hex cluster as other starts.
        return None

    best_S = float('inf')
    best_centers = None

    initial_configs = [make_hex_cluster()]
    
    # Add a few random perturbations
    rng = np.random.default_rng(42)
    for _ in range(5):
        base = make_hex_cluster()
        perturbed = base + rng.normal(0, 0.1, base.shape)
        initial_configs.append(perturbed)

    for i, init_c in enumerate(initial_configs):
        if time.time() > deadline - safety_margin:
            break
            
        try:
            # Nelder-Mead is good for this
            res = minimize(objective, init_c, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            # Check if the result is valid (low penalty)
            final_pen = overlap_penalty(res.x)
            if final_pen < 1e-4: # Very small overlap
                S_val = calc_S(res.x)
                if S_val < best_S:
                    best_S = S_val
                    best_centers = res.x.copy()
        except Exception as e:
            pass

    # If no valid solution found, fallback to the best one even with penalty? 
    # Or just return the initial cluster if it's valid.
    if best_centers is None:
        # Check initial configs for validity
        for init_c in initial_configs:
            if overlap_penalty(init_c) < 1e-4:
                S_val = calc_S(init_c)
                if S_val < best_S:
                    best_S = S_val
                    best_centers = init_c.copy()
        
        if best_centers is None:
            # Emergency fallback: line
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0

    # Format output
    inner_rows = []
    for cx, cy in best_centers:
        inner_rows.append([float(cx), float(cy), 0.0])
        
    center = (0.0, 0.0)
    side = float(best_S)
    angle_deg = 0.0 # Fixed outer hexagon orientation
    
    return inner_rows, center, side, angle_deg
