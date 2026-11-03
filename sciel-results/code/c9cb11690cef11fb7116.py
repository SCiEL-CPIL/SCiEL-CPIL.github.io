"""Rewritten program: SLSQP optimization with hard non-overlap constraints."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s limit

    def hex_vertices(cx, cy, angle_deg, scale=1.0):
        """Get vertices of a regular hexagon with side length=scale, centered at (cx,cy), rotated by angle_deg."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + scale * math.cos(a)
            y = cy + scale * math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def min_distance_hex(v1, v2):
        """
        Calculate the minimum distance between two convex hexagons.
        Returns positive if disjoint, negative if overlapping (penetration depth).
        Uses SAT to find the minimum projection overlap or gap.
        """
        min_dist = float('inf')
        # Axes from v1
        for i in range(6):
            edge1 = v1[(i + 1) % 6] - v1[i]
            normal = np.array([-edge1[1], edge1[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-9)
            
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            if max1 < min2:
                # Disjoint, gap is min2 - max1
                min_dist = min(min_dist, min2 - max1)
            elif max2 < min1:
                # Disjoint, gap is min1 - max2
                min_dist = min(min_dist, min1 - max2)
            else:
                # Overlapping or touching
                # The "distance" is 0 if we consider them touching, 
                # but for penetration depth, we look at the overlap amount.
                # However, SAT min_dist logic: if they overlap on all axes, min_dist remains inf?
                # No, if they overlap, we want a negative value representing penetration.
                # Standard SAT for collision doesn't give depth directly.
                # We can approximate penetration depth as the minimum overlap on any axis?
                # Actually, the minimum distance is 0 if they touch. If they overlap, it's negative.
                # A robust way: if they overlap, find the minimum overlap depth on any axis.
                # But that's not the true geometric distance.
                # For SLSQP, we just need a continuous function that is < 0 when overlapping.
                # Let's use the minimum overlap on any axis as a proxy for -distance.
                overlap = min(max1 - min2, max2 - min1)
                if overlap < 0:
                    min_dist = min(min_dist, 0.0) # They are disjoint on this axis? No, this is overlap.
                # If they overlap on this axis, the distance is <= 0.
                # To make it smooth and differentiable-ish, we can use the overlap amount.
                # If overlap > 0, they are intersecting.
                # Let's return -overlap as a measure of "how much they overlap" on this axis.
                # The true minimum distance is the minimum over all axes of (gap if disjoint, -overlap if intersecting)?
                # No, if they intersect, the distance is negative. The magnitude is the penetration depth.
                # The minimum penetration depth is the minimum overlap on any axis? No, that's the maximum gap.
                # Actually, for convex polygons, if they intersect, the minimum distance is negative.
                # We can define the constraint function as:
                # d = min over axes of (separation on that axis)
                # If separated, separation is positive.
                # If intersecting, separation is negative (overlap).
                # So we can just compute the separation on each axis and take the minimum.
                sep = min(max1 - min2, min1 - max2) # This is wrong for overlap.
                # Correct separation:
                # If max1 < min2: sep = min2 - max1
                # If max2 < min1: sep = min1 - max2
                # Else: sep = -min(max1 - min2, max2 - min1) # Negative overlap
                # Let's implement this properly.
                if max1 < min2:
                    sep = min2 - max1
                elif max2 < min1:
                    sep = min1 - max2
                else:
                    # Overlapping
                    overlap1 = max1 - min2
                    overlap2 = max2 - min1
                    # The penetration depth is roughly the minimum of these overlaps?
                    # No, the true penetration depth is the minimum translation to separate.
                    # For a constraint, we just need sep < 0 when overlapping.
                    # Using -min(overlap1, overlap2) is a valid negative measure.
                    sep = -min(overlap1, overlap2)
                
                min_dist = min(min_dist, sep)
        
        # Repeat for axes from v2
        for i in range(6):
            edge2 = v2[(i + 1) % 6] - v2[i]
            normal = np.array([-edge2[1], edge2[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-9)
            
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            if max1 < min2:
                sep = min2 - max1
            elif max2 < min1:
                sep = min1 - max2
            else:
                overlap1 = max1 - min2
                overlap2 = max2 - min1
                sep = -min(overlap1, overlap2)
            
            min_dist = min(min_dist, sep)
            
        if min_dist == float('inf'):
            return 0.0 # Should not happen if axes cover all directions
        return min_dist

    def compute_min_side(all_verts, outer_angle_deg):
        """
        Compute minimum side length of outer hexagon (centered at origin, rotated by outer_angle_deg)
        to contain all points in all_verts.
        """
        if len(all_verts) == 0:
            return 0.0
        theta = math.radians(outer_angle_deg)
        max_proj = 0.0
        for k in range(6):
            a = theta + k * math.pi / 3
            normal = np.array([math.cos(a), math.sin(a)])
            proj = np.dot(all_verts, normal)
            max_proj = max(max_proj, proj.max())
        return 2.0 * max_proj / math.sqrt(3.0)

    # Define honeycomb cluster
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    ring2_all = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            dist = (abs(q) + abs(r) + abs(q + r)) / 2
            if dist == 2:
                ring2_all.append((q, r))
    def hex_angle(q, r):
        x, y = axial_to_cartesian(q, r)
        return math.atan2(y, x)
    ring2_all.sort(key=lambda p: hex_angle(p[0], p[1]))
    ring2_selected = [ring2_all[i] for i in range(0, 12, 2)][:5]
    while len(ring2_selected) < 5:
        ring2_selected.append(ring2_all[len(ring2_selected)])

    cluster_axial = [(0, 0)] + ring1 + ring2_selected
    cluster_cart = []
    for q, r in cluster_axial:
        x, y = axial_to_cartesian(q, r)
        cluster_cart.append((x, y))
    cx = sum(p[0] for p in cluster_cart) / len(cluster_cart)
    cy = sum(p[1] for p in cluster_cart) / len(cluster_cart)
    cluster_cart = [(x - cx, y - cy) for x, y in cluster_cart]
    cluster_cart = np.array(cluster_cart)

    def build_config(params):
        """
        params: [scale, rotation_deg, tx, ty, outer_angle_deg]
        Returns positions (Nx2), angles (N), outer_angle
        """
        scale, rot_deg, tx, ty, oa_deg = params
        theta = math.radians(rot_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        positions = []
        angles = [rot_deg] * N
        for i in range(N):
            x, y = cluster_cart[i] * scale
            rx = x * cos_t - y * sin_t
            ry = x * sin_t + y * cos_t
            positions.append((rx + tx, ry + ty))
        return np.array(positions), angles, oa_deg

    def objective(x):
        if time.time() > deadline - 2:
            return 1e10
        positions, angles, oa_deg = build_config(x)
        verts_list = [hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
        all_verts = np.vstack(verts_list)
        side = compute_min_side(all_verts, oa_deg)
        return side

    def constraints_func(x):
        if time.time() > deadline - 2:
            return [0.0] * (N * (N - 1) // 2)
        
        positions, angles, oa_deg = build_config(x)
        verts_list = [hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
        
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                dist = min_distance_hex(verts_list[i], verts_list[j])
                # We want dist >= 0
                cons.append(dist)
        return cons

    # Initial guess: scale=1.0, rot=0, tx=0, ty=0, oa=0
    x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    
    # Bounds
    bounds = [(0.8, 1.5), (0, 360), (-2, 2), (-2, 2), (0, 360)]
    
    # Constraints definition for SLSQP
    # SLSQP expects a list of dicts: {'type': 'ineq', 'fun': lambda x: ...}
    # But defining 66 lambdas is inefficient. We can use a single constraint function returning an array?
    # No, SLSQP 'ineq' fun must return a scalar or array?
    # Documentation says: "If fun returns an array, each element is treated as a separate constraint."
    # So we can pass a single constraint dict with fun=constraints_func.
    
    cons = [{'type': 'ineq', 'fun': constraints_func}]
    
    best_score = float('inf')
    best_params = x0
    
    # Try SLSQP
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                          options={'maxiter': 200, 'ftol': 1e-6, 'disp': False})
        if result.fun < best_score:
            # Verify validity
            positions, angles, oa_deg = build_config(result.x)
            verts_list = [hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if min_distance_hex(verts_list[i], verts_list[j]) < -1e-4:
                        valid = False
                        break
                if not valid: break
            if valid:
                best_score = result.fun
                best_params = result.x
            else:
                # If SLSQP found a solution with slight overlap, try to fix or discard
                # We can try to push them apart slightly by increasing scale
                pass
    except Exception as e:
        print(f"SLSQP failed: {e}")

    # Fallback: Grid search + Nelder-Mead (from previous version, simplified)
    if best_score == float('inf') or time.time() > deadline - 60:
        # Use a simpler heuristic if SLSQP didn't work well or is too slow
        def evaluate_config(scale, rotation_deg, tx, ty, outer_angle_deg):
            positions, angles, oa = build_config([scale, rotation_deg, tx, ty, outer_angle_deg])
            verts_list = [hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
            for i in range(N):
                for j in range(i + 1, N):
                    if min_distance_hex(verts_list[i], verts_list[j]) < 0:
                        return float('inf'), False
            all_verts = np.vstack(verts_list)
            side = compute_min_side(all_verts, oa)
            return side, True

        best_score = float('inf')
        best_params = None
        rot_candidates = list(range(0, 180, 30))
        oa_candidates = list(range(0, 180, 30))
        scale_candidates = [0.95, 1.0, 1.05]

        for rot in rot_candidates:
            if time.time() > deadline - 10: break
            for oa in oa_candidates:
                if time.time() > deadline - 5: break
                for sc in scale_candidates:
                    if time.time() > deadline - 3: break
                    for tx in [-0.5, 0.0, 0.5]:
                        for ty in [-0.5, 0.0, 0.5]:
                            score, valid = evaluate_config(sc, float(rot), tx, ty, float(oa))
                            if valid and score < best_score:
                                best_score = score
                                best_params = np.array([sc, float(rot), tx, ty, float(oa)])

        if best_params is not None:
            try:
                result = minimize(objective, best_params, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
                # Verify
                positions, angles, oa_deg = build_config(result.x)
                verts_list = [hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        if min_distance_hex(verts_list[i], verts_list[j]) < -1e-4:
                            valid = False
                            break
                    if not valid: break
                if valid and result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x
            except Exception:
                pass

    # Finalize
    if best_params is None:
        best_params = x0
        best_score, _ = evaluate_config(1.0, 0.0, 0.0, 0.0, 0.0)
        
    scale, rot_deg, tx, ty, oa_deg = best_params
    positions, angles, outer_angle = build_config(best_params)
    
    # Center the configuration
    cx_final = np.mean(positions[:, 0])
    cy_final = np.mean(positions[:, 1])
    positions = positions - np.array([cx_final, cy_final])
    
    # Recompute final side
    verts_list = [hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
    all_verts = np.vstack(verts_list)
    final_side = compute_min_side(all_verts, outer_angle)
    
    # Verify
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            if min_distance_hex(verts_list[i], verts_list[j]) < -1e-4:
                valid = False
                break
        if not valid: break
        
    if not valid:
        # Fallback: spread out
        positions = np.array([[i % 4 * 2.5 + (i // 4 % 2) * 1.25, (i // 4) * 2.2] for i in range(N)])
        cx_f = np.mean(positions[:, 0])
        cy_f = np.mean(positions[:, 1])
        positions = positions - np.array([cx_f, cy_f])
        angles = [0.0] * N
        outer_angle = 0.0
        verts_list = [hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
        all_verts = np.vstack(verts_list)
        final_side = compute_min_side(all_verts, outer_angle)

    inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), final_side, outer_angle
# EVOLVE-BLOCK-END
