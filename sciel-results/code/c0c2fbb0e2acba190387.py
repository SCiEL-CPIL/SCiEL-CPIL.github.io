import math
import time
import numpy as np
from scipy.optimize import minimize_scalar, minimize, differential_evolution

N = 11

def construct():
    start_time = time.time()
    deadline = 145.0
    safety_margin = 10.0

    d = math.sqrt(3)

    # Generate base centers: center + ring 1 (6) + ring 2 (12)
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers.append((x, y))

    ring2_centers = []
    ring2_indices = {}
    idx = 0
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_centers.append((x, y))
                ring2_indices[(q, r)] = idx
                idx += 1

    base_centers = centers[:7]
    n_base = len(base_centers)

    # Hexagon vertices relative to center (unit hexagon)
    verts_rel = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])

    def get_hex_verts(cx, cy, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        rx = verts_rel[:, 0] * c - verts_rel[:, 1] * s
        ry = verts_rel[:, 0] * s + verts_rel[:, 1] * c
        return np.column_stack([rx + cx, ry + cy])

    def sat_distance(v1, v2):
        """Return minimum separation distance between two hexagons (negative if overlapping)."""
        min_gap = float('inf')
        for v in (v1, v2):
            for i in range(6):
                p1 = v[i]
                p2 = v[(i + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                normal = normal / np.linalg.norm(normal)
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                gap = min(proj1.max() - proj2.min(), proj2.max() - proj1.min())
                min_gap = min(min_gap, gap)
        return min_gap

    def compute_S(params):
        """Compute the enclosing hexagon side length for a set of hexagons."""
        n_hex = len(params) // 3
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        
        # Collect all vertices
        all_pts = []
        for i in range(n_hex):
            verts = get_hex_verts(xs[i], ys[i], thetas[i])
            all_pts.append(verts)
        all_pts = np.vstack(all_pts)
        
        # The enclosing regular hexagon has sides at angles theta + k*pi/3
        # Side length S satisfies: max projection onto each normal direction = S * sqrt(3)/2
        # Normal directions are at angles theta + k*pi/3 for k=0..5
        # Actually, the normals to the sides are at angles theta + k*pi/3 + pi/6? 
        # Let's think: A regular hexagon with "pointy" vertices at angles theta + k*pi/3
        # has flat sides. The normal to the side between vertices at angle a and a+pi/3
        # is at angle a + pi/6.
        # So normals are at theta + pi/6 + k*pi/3 for k=0..5.
        
        # Alternative: A regular hexagon with "flat" top has vertices at angles pi/6 + k*pi/3.
        # Its sides have normals at angles 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3.
        # So if we want the enclosing hexagon to have rotation angle alpha,
        # its side normals are at alpha + k*pi/3 for k=0..5.
        # The side length S is related to the maximum projection P by: P = S * sqrt(3)/2
        # So S = 2*P/sqrt(3)
        
        # We can choose the rotation of the outer hexagon freely.
        # So we need to find alpha that minimizes max_k max_i |dot(p_i, n_k)|
        # where n_k = (cos(alpha + k*pi/3), sin(alpha + k*pi/3))
        
        # This is a 1D optimization over alpha in [0, pi/3)
        # For a given set of points, the function is piecewise linear in alpha.
        
        # For now, we'll sample alpha and refine.
        # But for the objective function in optimization, we need to be efficient.
        
        # Actually, the optimal alpha for a given set of points can be found by checking
        # critical angles where some vertex aligns with a normal direction.
        # But that's complex. Let's use a simpler approach:
        # The enclosing hexagon side length is the minimum over alpha of
        # (2/sqrt(3)) * max_k max_i |dot(p_i - center, n_k(alpha))|
        
        # For simplicity in the optimization loop, let's just use alpha=0
        # and optimize positions. The outer rotation can be optimized separately.
        
        k_angles = np.arange(6) * math.pi / 3
        normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
        
        dots = np.abs(np.sum(normals[:, np.newaxis, :] * all_pts[np.newaxis, :, :], axis=-1))
        max_per_normal = np.max(dots, axis=-1)
        max_proj = np.max(max_per_normal)
        return (2.0 / math.sqrt(3)) * max_proj

    def compute_S_with_rotation(params, alpha):
        n_hex = len(params) // 3
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        
        all_pts = []
        for i in range(n_hex):
            verts = get_hex_verts(xs[i], ys[i], thetas[i])
            all_pts.append(verts)
        all_pts = np.vstack(all_pts)
        
        k_angles = alpha + np.arange(6) * math.pi / 3
        normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
        
        dots = np.abs(np.sum(normals[:, np.newaxis, :] * all_pts[np.newaxis, :, :], axis=-1))
        max_per_normal = np.max(dots, axis=-1)
        max_proj = np.max(max_per_normal)
        return (2.0 / math.sqrt(3)) * max_proj

    def objective(params):
        if time.time() > deadline - safety_margin:
            return 1e10
            
        n_hex = len(params) // 3
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        
        # SAT overlap penalty
        verts_list = [get_hex_verts(xs[i], ys[i], thetas[i]) for i in range(n_hex)]
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                gap = sat_distance(verts_list[i], verts_list[j])
                if gap < 0:
                    penalty += gap * gap * 1000.0
        
        S = compute_S(params)
        return S + penalty

    # Try different combinations of ring 2 hexagons
    # The 12 ring 2 positions form a hexagonal ring. We need to choose 4 of them.
    # Symmetry: we can assume we choose positions that are "balanced"
    
    # Generate all combinations of 4 from 12
    import itertools
    all_combos = list(itertools.combinations(range(12), 4))
    
    best_S = float('inf')
    best_params = None
    best_outer_alpha = 0.0

    # Phase 1: Coarse search over combinations
    # For each combo, try to optimize with a simple initial guess
    
    # To speed up, we'll first evaluate with fixed theta=0 and just vary positions slightly
    # Actually, let's just do a coarse Nelder-Mead for the top few combos
    
    # Evaluate all combos with the initial lattice positions (theta=0)
    combo_scores = []
    for combo in all_combos:
        final_centers = base_centers + [ring2_centers[i] for i in combo]
        params = np.zeros(N * 3)
        for i, (cx, cy) in enumerate(final_centers):
            params[3*i] = cx
            params[3*i+1] = cy
            params[3*i+2] = 0.0
        
        S_val = compute_S(params)
        
        # Check overlaps
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        verts_list = [get_hex_verts(xs[i], ys[i], thetas[i]) for i in range(N)]
        overlap = False
        for i in range(N):
            for j in range(i + 1, N):
                if sat_distance(verts_list[i], verts_list[j]) < 0:
                    overlap = True
                    break
            if overlap:
                break
        
        if not overlap:
            combo_scores.append((S_val, combo))
    
    combo_scores.sort(key=lambda x: x[0])
    
    # Phase 2: Refine the top combos
    for S_coarse, combo in combo_scores[:20]:
        if time.time() > deadline - safety_margin:
            break
        
        final_centers = base_centers + [ring2_centers[i] for i in combo]
        init_params = np.zeros(N * 3)
        for i, (cx, cy) in enumerate(final_centers):
            init_params[3*i] = cx
            init_params[3*i+1] = cy
            init_params[3*i+2] = 0.0
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        
        except Exception:
            continue
    
    # If we have a valid best_params, optimize the outer hexagon rotation
    if best_params is not None:
        if time.time() < deadline - safety_margin:
            def outer_obj(alpha):
                return compute_S_with_rotation(best_params, alpha)
            
            try:
                res_outer = minimize_scalar(outer_obj, bounds=(0, math.pi/3), method='bounded',
                                           options={'xatol': 1e-8})
                if res_outer.success:
                    best_outer_alpha = res_outer.x
                    best_S = res_outer.fun
            except Exception:
                best_outer_alpha = 0.0
    else:
        # Fallback: use the best coarse combo without optimization
        if combo_scores:
            S_coarse, combo = combo_scores[0]
            final_centers = base_centers + [ring2_centers[i] for i in combo]
            best_params = np.zeros(N * 3)
            for i, (cx, cy) in enumerate(final_centers):
                best_params[3*i] = cx
                best_params[3*i+1] = cy
                best_params[3*i+2] = 0.0
            best_S = S_coarse
            best_outer_alpha = 0.0
        else:
            # Should not happen
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0

    # Construct the inner hexagons from best_params
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        angle_deg = math.degrees(best_params[3*i+2])
        inner.append([x, y, angle_deg])

    return inner, (0.0, 0.0), best_S, math.degrees(best_outer_alpha)
