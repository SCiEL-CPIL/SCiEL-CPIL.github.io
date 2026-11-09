import numpy as np
import time
import random

def _hex_vertices(cx, cy, d, a):
    a_rad = np.deg2rad(a)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_outer_side(positions, angle_deg):
    """
    Compute the minimum side length of a regular hexagon with the given angle
    that encloses all unit hexagons at the given positions.
    """
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    # All vertices of all inner hexagons
    all_verts = positions[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    
    a_rad = np.deg2rad(angle_deg)
    # Normals to the edges of the outer hexagon
    normals = []
    for i in range(6):
        n_angle = a_rad + i * np.pi/3 + np.pi/6
        n_unit = np.array([np.cos(n_angle), np.sin(n_angle)])
        normals.append(n_unit)
    normals = np.array(normals)
    
    projections = all_verts_flat @ normals.T
    max_projections = np.max(projections, axis=0)
    # The distance from center to edge is s * sqrt(3) / 2
    # So max_projection = s * sqrt(3) / 2  => s = max_projection * 2 / sqrt(3)
    min_s = np.max(max_projections) * 2 / np.sqrt(3)
    return min_s

def _check_validity(inner, outer, s):
    """
    Validates the solution according to the task contract.
    """
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
        
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False
                
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False

    inner_verts_list = []
    for i in range(12):
        cx, cy, ang = inner[i]
        inner_verts_list.append(_hex_vertices(cx, cy, 1, ang))
        
    for i in range(12):
        for j in range(i+1, 12):
            va = inner_verts_list[i]
            vb = inner_verts_list[j]
            separated = False
            for k in range(6):
                v1 = va[k]
                v2 = va[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if separated: continue
            
            for k in range(6):
                v1 = vb[k]
                v2 = vb[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if not separated:
                return False
                
    for i in range(12):
        for v in inner_verts_list[i]:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                e = p2 - p1
                cross_val = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
                if cross_val < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
                
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580
    
    best_score = 0.0
    best_inner = None
    best_outer = None
    best_s = None
    
    # Initial guess: Hexagonal lattice arrangement
    # This is a known good packing for 7 hexagons (center + 6 neighbors).
    # For 12, we extend the lattice.
    h = np.sqrt(3)/2
    # Center at (0,0)
    # 1st ring: 6 hexagons
    # 2nd ring: 12 hexagons (we only need 5 more to reach 12 total? No, 1+6=7. Need 5 more from ring 2)
    # Let's just use a standard lattice seed.
    init_pos = np.array([
        [0, 0],
        [2*h, 0],
        [h, 1.5],
        [-h, 1.5],
        [-2*h, 0],
        [-h, -1.5],
        [h, -1.5],
        # Ring 2 candidates
        [3*h, 1.5],
        [0, 3],
        [-3*h, 1.5],
        [-3*h, -1.5],
        [3*h, -1.5]
    ])
    
    # SA Parameters
    T0 = 1.0
    T_end = 1e-8
    alpha = 0.995
    N = 12
    steps_per_temp = 50
    
    # Precompute unit hex vertices for speed
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    
    def evaluate(positions, angle_deg):
        s = _min_outer_side(positions, angle_deg)
        
        # Calculate overlap penalty
        penalty = 0.0
        inner_verts_list = []
        for i in range(N):
            verts = positions[i, :2] + unit_hex_verts
            inner_verts_list.append(verts)
            
        for i in range(N):
            va = inner_verts_list[i]
            for j in range(i+1, N):
                vb = inner_verts_list[j]
                # SAT test
                separated = False
                min_gap = 0.0
                for k in range(6):
                    v1 = va[k]
                    v2 = va[(k+1)%6]
                    e = v2 - v1
                    n = np.array([-e[1], e[0]])
                    norm = np.linalg.norm(n)
                    if norm < 1e-12: continue
                    n = n / norm
                    proj_a = np.dot(va, n)
                    proj_b = np.dot(vb, n)
                    gap = max(np.min(proj_a) - np.max(proj_b), np.min(proj_b) - np.max(proj_a))
                    if gap > 1e-6:
                        separated = True
                        break
                    # If not separated, accumulate the "depth" of overlap for penalty
                    # We want to push them apart. The most negative gap is the deepest overlap.
                    if gap < min_gap:
                        min_gap = gap
                if not separated:
                    # Penalty is proportional to square of overlap depth
                    # This ensures we strongly penalize overlaps
                    penalty += (min_gap * min_gap) * 100.0
                    
                # Check axes from vb as well for completeness of depth calculation
                # (Though for convex polygons, if they overlap, the minimum gap on any axis is negative)
                # The above loop checks axes from va. We should also check axes from vb to get the true minimum gap.
                # However, for penalty purposes, a rough estimate is often enough to guide the search.
                # Let's do a quick check on vb axes if not separated
                if not separated:
                    for k in range(6):
                        v1 = vb[k]
                        v2 = vb[(k+1)%6]
                        e = v2 - v1
                        n = np.array([-e[1], e[0]])
                        norm = np.linalg.norm(n)
                        if norm < 1e-12: continue
                        n = n / norm
                        proj_a = np.dot(va, n)
                        proj_b = np.dot(vb, n)
                        gap = max(np.min(proj_a) - np.max(proj_b), np.min(proj_b) - np.max(proj_a))
                        if gap > 1e-6:
                            separated = True
                            break
                        if gap < min_gap:
                            min_gap = gap
                    if not separated:
                         # We already added penalty for va axes. 
                         # To avoid double counting or confusion, we just take the min_gap found so far.
                         # The penalty added above was based on va axes. 
                         # It's better to re-calculate min_gap properly if we want accurate penalty.
                         # But for speed, the va-axis penalty is a good heuristic.
                         pass
            
        # Add a large penalty if any vertex is outside the outer hexagon
        # This is handled implicitly by s increasing, but if overlaps are huge, s might be small?
        # No, s is determined by the bounding box of the vertices.
        # If they overlap, s is smaller.
        # The objective is to minimize s + penalty.
        
        return s + penalty
    
    def run_sa(initial_pos, initial_angle):
        current_pos = initial_pos.copy()
        current_angle = initial_angle
        current_cost = evaluate(current_pos, current_angle)
        
        best_pos = current_pos.copy()
        best_angle = current_angle
        best_cost = current_cost
        
        T = T0
        step_size = 0.5
        
        while T > T_end:
            for _ in range(steps_per_temp):
                # Perturb
                new_pos = current_pos.copy()
                # Move a random hexagon
                idx = np.random.randint(N)
                new_pos[idx] += np.random.randn(2) * step_size
                # Optionally perturb angle slightly
                new_angle = current_angle + np.random.randn() * 0.1
                
                new_cost = evaluate(new_pos, new_angle)
                
                # Accept if better, or with probability exp(-(new_cost - current_cost)/T)
                if new_cost < current_cost:
                    current_pos = new_pos
                    current_angle = new_angle
                    current_cost = new_cost
                    
                    if new_cost < best_cost:
                        best_pos = new_pos.copy()
                        best_angle = new_angle
                        best_cost = new_cost
                else:
                    p = np.exp(-(new_cost - current_cost) / T)
                    if np.random.random() < p:
                        current_pos = new_pos
                        current_angle = new_angle
                        current_cost = new_cost
                        
            T *= alpha
            # Reduce step size slightly to refine
            step_size *= 0.98
            
        return best_pos, best_angle, best_cost

    # Multi-start
    np.random.seed(42)
    random.seed(42)
    
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            init_pos_curr = init_pos.copy()
            init_angle_curr = 0.0
        else:
            noise_pos = np.random.randn(N, 2) * 0.5
            init_pos_curr = init_pos + noise_pos
            init_angle_curr = np.random.uniform(-10, 10)
            
        best_pos, best_angle, best_cost = run_sa(init_pos_curr, init_angle_curr)
        
        # Check validity of the best solution found
        s_candidate = _min_outer_side(best_pos, best_angle)
        
        inner_candidate = np.zeros((N, 3))
        inner_candidate[:, :2] = best_pos
        inner_candidate[:, 2] = 0.0 # Inner hexagons are axis-aligned in this formulation?
        # Wait, the task allows angles for inner hexagons.
        # My SA only optimizes positions and the OUTER angle.
        # The inner hexagons are fixed at angle 0.
        # Is this suboptimal?
        # For a regular hexagon packing, usually all hexagons are aligned.
        # Let's assume inner angles are 0 for now. The outer angle is optimized.
        # If inner angles could vary, it's a much larger search space.
        # Given the symmetry, 0 is a strong candidate.
        
        outer_candidate = np.array([0, 0, best_angle])
        
        if _check_validity(inner_candidate, outer_candidate, s_candidate):
            score_candidate = 1.0 / s_candidate
            if score_candidate > best_score:
                best_score = score_candidate
                best_inner = inner_candidate
                best_outer = outer_candidate
                best_s = s_candidate
        else:
            # If invalid, try to "repair" by projecting out overlaps?
            # Or just accept if it's close?
            # The contract says invalid scores 0. So we must be strict.
            # However, if the penalty was high, it might still be invalid.
            # Let's try a few more SA steps with lower T if invalid?
            # Or just skip.
            pass

    if best_inner is not None:
        return best_inner, best_outer, best_s
    else:
        # Fallback: Return the lattice solution even if not perfectly optimized, 
        # ensuring it's valid.
        inner_fallback = np.array([
            [0, 0, 0],
            [2.5, 0, 0],
            [-2.5, 0, 0],
            [1.25, 2.165, 0],
            [-1.25, 2.165, 0],
            [1.25, -2.165, 0],
            [-1.25, -2.165, 0],
            [3.75, 2.165, 0],
            [-3.75, 2.165, 0],
            [3.75, -2.165, 0],
            [-3.75, -2.165, 0],
            [0, 4.33, 0]
        ])
        outer_fallback = np.array([0, 0, 0])
        s_fallback = _min_outer_side(inner_fallback[:, :2], 0)
        if s_fallback < np.sqrt(12):
            s_fallback = np.sqrt(12)
        return inner_fallback, outer_fallback, s_fallback

# EVOLVE-BLOCK-END
