"""Direct construction of 12 unit hexagons inside a regular hexagon, minimizing outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math
import random

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _min_required_side(params, outer_angle_deg):
    """
    Calculate the minimum side length of a regular hexagon (centered at origin, rotated by outer_angle_deg)
    that contains all N unit hexagons defined by params [x0, y0, a0, ...].
    
    This uses the support function approach for exact calculation.
    """
    outer_angle = np.radians(outer_angle_deg)
    # 6 outward normals of the outer hexagon
    # The edges of a regular hexagon are perpendicular to these normals
    # Normal angles are outer_angle + 60 * i
    n_out = np.array([
        [np.cos(outer_angle + np.radians(60 * i)),
         np.sin(outer_angle + np.radians(60 * i))]
        for i in range(6)
    ])
    
    max_ext = 0.0
    
    for k in range(N):
        cx = params[3*k]
        cy = params[3*k+1]
        ang = np.radians(params[3*k+2])
        
        # Support function of the inner unit hexagon in direction n
        # h(n) = max over vertices v of dot(v, n)
        # Vertices of unit hex at angle ang: v_i = (cos(ang + 60i), sin(ang + 60i))
        # dot(v_i, n) = cos(ang + 60i - theta_n)
        
        for ni in range(6):
            dx = n_out[ni, 0]
            dy = n_out[ni, 1]
            theta_d = np.arctan2(dy, dx)
            
            # Compute support of inner hex in this direction
            # angles_v = ang + 60 * arange(6)
            # dots = cos(angles_v - theta_d)
            # h = max(dots)
            
            # Optimization: The max of cos(x + 60k - theta) occurs when x+60k-theta is closest to 0 mod 360.
            # Or just compute all 6. It's cheap.
            angles_v = ang + np.radians(60 * np.arange(6))
            dots = np.cos(angles_v - theta_d)
            h = np.max(dots)
            
            # The center of the inner hex is at (cx, cy).
            # The projection of the center onto the normal is dot((cx, cy), n)
            dot_center = cx * dx + cy * dy
            
            # The furthest extent in direction n is dot_center + h
            needed = dot_center + h
            
            if needed > max_ext:
                max_ext = needed
                
    # The apothem of the outer hexagon must be at least max_ext.
    # Apothem = S * sqrt(3) / 2
    # So S = 2 * max_ext / sqrt(3)
    S = 2.0 * max_ext / np.sqrt(3.0)
    return S

def _check_valid(params, outer_side, outer_center, outer_angle, tol=1e-8):
    """
    Validate using SAT (Separating Axis Theorem) for robustness.
    Returns True if valid.
    """
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = params[3*i], params[3*i+1], params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # 1. Containment Check
    # For each inner vertex, check if it's inside outer hex
    # Outer hex is convex. Check cross products.
    n_out = len(outer_verts)
    for iv in inner_verts_list:
        for v in iv:
            inside = True
            for i in range(n_out):
                v1 = outer_verts[i]
                v2 = outer_verts[(i + 1) % n_out]
                edge = v2 - v1
                rel = v - v1
                cross = edge[0] * rel[1] - edge[1] * rel[0]
                # Assuming CCW winding for outer hex
                if cross < -tol:
                    inside = False
                    break
            if not inside:
                return False

    # 2. Disjointness Check
    # For each pair, check SAT overlap
    for i in range(N):
        for j in range(i + 1, N):
            h1 = inner_verts_list[i]
            h2 = inner_verts_list[j]
            
            # Axes are normals to edges of h1 and h2
            axes = []
            for poly in [h1, h2]:
                n = len(poly)
                for k in range(n):
                    p1 = poly[k]
                    p2 = poly[(k + 1) % n]
                    edge = p2 - p1
                    norm = np.linalg.norm(edge)
                    if norm < 1e-10:
                        continue
                    normal = np.array([-edge[1], edge[0]]) / norm
                    axes.append(normal)
            
            overlap_found = True
            for axis in axes:
                proj1 = np.dot(h1, axis)
                proj2 = np.dot(h2, axis)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 <= min2 + tol or max2 <= min1 + tol:
                    overlap_found = False
                    break
            
            if overlap_found:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    def build_hexagonal_lattice(nrows, ncols):
        positions = []
        for r in range(nrows):
            for c in range(ncols):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3)
                positions.append((x, y))
        return positions

    def build_ring_config():
        positions = [(0.0, 0.0)]
        for i in range(6):
            ang = np.radians(60 * i)
            positions.append((2.0 * np.cos(ang), 2.0 * np.sin(ang)))
        # Add 5 more in second ring to make 12
        # Second ring positions are at distance sqrt(12) ~ 3.46
        # Or just use lattice 3x4 which is more compact?
        # Let's stick to simple lattices for initial seeds.
        return positions[:12]

    def make_params_from_positions(positions, angles=None):
        if angles is None:
            angles = [0.0] * len(positions)
        params = []
        for i, (x, y) in enumerate(positions):
            params.extend([float(x), float(y), float(angles[i])])
        return np.array(params)

    # Generate candidate initial configurations
    configs = []
    
    # 3 rows x 4 cols hex grid
    pos_3x4 = build_hexagonal_lattice(3, 4)
    if len(pos_3x4) == N:
        arr = np.array(pos_3x4)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    # 4 rows x 3 cols hex grid
    pos_4x3 = build_hexagonal_lattice(4, 3)
    if len(pos_4x3) == N:
        arr = np.array(pos_4x3)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
        
    # 2 rows x 6 cols
    pos_2x6 = build_hexagonal_lattice(2, 6)
    if len(pos_2x6) == N:
        arr = np.array(pos_2x6)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())

    # 6 rows x 2 cols
    pos_6x2 = build_hexagonal_lattice(6, 2)
    if len(pos_6x2) == N:
        arr = np.array(pos_6x2)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())

    # 12 in a line
    line_pos = [(float(i - 5.5), 0.0) for i in range(12)]
    configs.append(line_pos)
    
    # Compact 2x6 with offset (same as above but ensuring centroid)
    # Add some random perturbations to break symmetry
    random.seed(42)
    for _ in range(5):
        base = configs[0].copy()
        perturbed = []
        for (x, y) in base:
            perturbed.append((x + random.uniform(-0.2, 0.2), y + random.uniform(-0.2, 0.2)))
        configs.append(perturbed)
    
    best_score = float('inf')
    best_result = None
    
    outer_angles = [0.0, 30.0]
    
    for outer_angle in outer_angles:
        if time.time() > deadline - 20.0:
            break
            
        outer_center = (0.0, 0.0)
        
        for config in configs:
            if time.time() > deadline - 15.0:
                break
                
            if len(config[0]) == 2:
                init_params = make_params_from_positions(config)
            else:
                init_params = np.array([item for tri in config for item in tri])
            
            # Objective: Minimize the required side length
            def objective(p):
                return _min_required_side(p, outer_angle)
            
            try:
                result = minimize(
                    objective,
                    init_params,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                opt_params = result.x
                opt_val = result.fun
            except Exception:
                continue
            
            # Check if this is a valid packing with a slightly larger side to be safe
            # The analytical side is the exact minimum. Adding a tiny epsilon ensures validity in floating point.
            test_side = opt_val + 1e-7
            
            if test_side < best_score:
                # Validate with SAT to be absolutely sure
                if _check_valid(opt_params, test_side, outer_center, outer_angle):
                    best_score = test_side
                    best_result = (opt_params.copy(), outer_center, test_side, outer_angle)
                    # If we get a very good score, we might want to break early or refine
                    if best_score < 3.2: # Heuristic: 3.2 is quite good
                         # Don't break immediately, try to polish
                         pass
    
    # Final Polish: If we have a result, try to squeeze it further
    if best_result is not None and time.time() < deadline - 10.0:
        opt_params, outer_center, outer_side, outer_angle = best_result
        
        # Try to reduce the side length by optimizing with a fixed, slightly smaller target
        # Or just re-optimize with tighter tolerances
        def final_objective(p):
            return _min_required_side(p, outer_angle)
            
        try:
            result = minimize(
                final_objective,
                opt_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 20000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result.fun + 1e-7 < best_score:
                new_params = result.x
                new_side = result.fun + 1e-7
                if _check_valid(new_params, new_side, outer_center, outer_angle):
                    best_result = (new_params, outer_center, new_side, outer_angle)
                    best_score = new_side
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        inner = []
        pos = build_hexagonal_lattice(3, 4)
        arr = np.array(pos)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0
    
    opt_params, outer_center, outer_side, outer_angle = best_result
    
    inner = []
    for i in range(N):
        x = float(opt_params[3*i])
        y = float(opt_params[3*i+1])
        ang = float(opt_params[3*i+2])
        inner.append([x, y, ang])
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)

# EVOLVE-BLOCK-END
