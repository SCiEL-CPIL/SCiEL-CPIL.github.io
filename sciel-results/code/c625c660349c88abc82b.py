"""Smooth penalty optimization for hexagon packing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 120.0  # Safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    verts_local = []
    s = side
    for i in range(6):
        theta = i * np.pi / 3
        verts_local.append([s * np.cos(theta), s * np.sin(theta)])
    verts_local = np.array(verts_local)
    
    for i in range(6):
        v1 = verts_local[i]
        v2 = verts_local[(i + 1) % 6]
        edge = v2 - v1
        nx, ny = edge[1], -edge[0]
        if nx * (lx - v1[0]) + ny * (ly - v1[1]) > 1e-10:
            return False
    return True

def _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle, tol=1e-6):
    """Find minimum outer side length such that all inner hexagons fit inside the outer hexagon."""
    cx, cy = outer_center
    all_verts = []
    for i in range(len(inner_positions)):
        x, y = inner_positions[i]
        ang = inner_angles[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    lo, hi = 1.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        ok = True
        for v in all_verts:
            if not _point_in_hexagon(v[0], v[1], cx, cy, mid, outer_angle):
                ok = False
                break
        if ok:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if packing is valid: no overlaps, all inside outer hexagon."""
    cx, cy = outer_center
    verts_list = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        verts_list.append(verts)
    
    # Check no overlaps using SAT
    for i in range(len(verts_list)):
        for j in range(i + 1, len(verts_list)):
            if _hexagon_overlap(verts_list[i], verts_list[j]):
                return False
    
    # Check all inside outer hexagon
    for verts in verts_list:
        for v in verts:
            if not _point_in_hexagon(v[0], v[1], cx, cy, outer_side, outer_angle):
                return False
    return True

def _hexagon_overlap(verts1, verts2, tol=1e-9):
    """Check if two convex polygons overlap using SAT (Separating Axis Theorem)."""
    axes = []
    for verts in [verts1, verts2]:
        for i in range(len(verts)):
            v1 = verts[i]
            v2 = verts[(i + 1) % len(verts)]
            edge = v2 - v1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                axes.append([nx/norm, ny/norm])
    
    for axis in axes:
        proj1 = np.dot(verts1, axis)
        proj2 = np.dot(verts2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    return True

def _initial_packing():
    """Create initial hexagonal grid packing."""
    positions = []
    angles = []
    sqrt3 = np.sqrt(3)
    # Use a tighter initial grid to start closer to the optimum
    # Hexagonal packing of 11: 
    # Row 0 (y=0): 3 hexes at x = -2, 0, 2
    # Row 1 (y=sqrt(3)): 4 hexes at x = -3, -1, 1, 3
    # Row 2 (y=2*sqrt(3)): 3 hexes at x = -2, 0, 2
    # Wait, standard hex packing:
    # Center at (0,0).
    # Neighbors at distance 2.
    # Let's just use a simple grid and let the optimizer fix it.
    # Or use the known optimal-ish configuration for 11 hexes.
    # 11 hexes in a hexagon:
    # Side length of outer hexagon for 11 hexes?
    # If we pack them in a triangular shape:
    # 1
    # 3
    # 5
    # 2? No, 1+3+5 = 9. 1+3+5+2 = 11?
    # Let's just generate a grid and trim.
    rows = 3
    for r in range(rows):
        y = (r - rows // 2) * sqrt3
        if r % 2 == 0:
            xs = range(-(rows // 2), rows - rows // 2 + 1)
        else:
            xs = range(-(rows // 2), rows - rows // 2)
        for x_idx in xs:
            x = (x_idx - (rows // 2)) * 2.0
            if r % 2 == 1:
                x += 1.0
            positions.append([x, y])
            angles.append(0.0)
    positions = positions[:N]
    angles = angles[:N]
    return np.array(positions), np.array(angles)

def construct():
    global START_TIME, DEADLINE
    
    # Initial packing
    pos, ang = _initial_packing()
    n = len(pos)
    
    # Flatten parameters: x1, y1, a1, x2, y2, a2, ..., outer_cx, outer_cy, outer_angle, outer_side
    # Actually, we want to minimize outer_side. So outer_side is the objective.
    # But outer_side is determined by the positions.
    # Better approach: Minimize a penalty function that includes the required outer side.
    # Or, fix outer_center and outer_angle, and optimize positions to minimize the required outer side.
    # Then, maybe optimize outer_center and outer_angle too.
    
    # Let's fix outer_center = (0,0) and outer_angle = 0 for simplicity in the first pass,
    # or allow them to vary.
    # Let's allow outer_center and outer_angle to vary.
    
    # Parameters:
    # 0..2*n-1: inner positions and angles
    # 2*n: outer_cx
    # 2*n+1: outer_cy
    # 2*n+2: outer_angle
    
    # Objective: Minimize the required outer side length + penalty for overlaps.
    # Required outer side is computed by _min_outer_side, which is non-smooth (binary search).
    # We can approximate the required side by taking the max over all inner vertices of the
    # "distance" to the outer hexagon boundary, but that's also non-smooth.
    
    # Alternative: Use a soft penalty for containment.
    # For each inner vertex, check if it's inside the outer hexagon.
    # If not, add a penalty proportional to the violation.
    # The objective is then: C * (max violation) + sum of overlap penalties.
    # But we want to MINIMIZE the outer side.
    
    # Let's try a different formulation:
    # Minimize s (outer side)
    # Subject to:
    # 1. All inner hexagons are inside the outer hexagon with side s.
    # 2. No inner hexagons overlap.
    
    # This is a non-linear programming problem.
    # We can use scipy.optimize.minimize with a penalty method.
    
    # Let's define the objective as:
    # f(params) = s + lambda * (sum of containment violations + sum of overlap violations)
    # Where s is the current outer side estimate.
    # But s is not in params.
    
    # Better:
    # Fix s, and check feasibility. Binary search on s.
    # For a fixed s, the problem is to find positions such that all hexagons fit and don't overlap.
    # This is still hard.
    
    # Let's go back to the smooth penalty idea.
    # Objective:
    # 1. Containment: For each inner vertex v, compute the minimum distance to the outer hexagon boundary (if outside, it's negative).
    #    We want all distances >= 0.
    #    Penalty = sum(max(0, -distance)^2)
    # 2. Overlap: For each pair of inner hexagons, compute the minimum distance between them.
    #    We want all distances >= 0.
    #    Penalty = sum(max(0, -distance)^2)
    # 3. We want to minimize the outer side s.
    #    So, add s to the objective.
    
    # But s is determined by the positions.
    # If we fix the outer hexagon (center, angle, side), and optimize the inner positions,
    # then s is fixed.
    # So, we can do:
    # 1. Start with a large s.
    # 2. Optimize inner positions to fit inside the outer hexagon (side s) without overlapping.
    # 3. If successful, decrease s and repeat.
    
    # This is a sequential approach.
    # Let's try to implement the smooth penalty for a FIXED outer hexagon.
    
    def _distance_to_hexagon_boundary(px, py, cx, cy, side, angle_deg):
        """Return the signed distance from point (px,py) to the boundary of the hexagon.
        Positive if inside, negative if outside."""
        dx = px - cx
        dy = py - cy
        a = np.radians(angle_deg)
        ca, sa = np.cos(-a), np.sin(-a)
        lx = ca * dx - sa * dy
        ly = sa * dx + ca * dy
        
        # The hexagon is defined by 6 half-planes.
        # The distance to the boundary is the minimum of the distances to the 6 edges.
        # For a regular hexagon with side s, the distance from the center to each edge is s * sqrt(3) / 2.
        # The distance from a point (lx, ly) to the edge i is:
        # d_i = (n_i · p) - c_i
        # where n_i is the unit normal to edge i, and c_i is the offset.
        
        # Let's compute the 6 normals and offsets.
        # Vertices in local frame:
        verts_local = []
        for i in range(6):
            theta = i * np.pi / 3
            verts_local.append([side * np.cos(theta), side * np.sin(theta)])
        verts_local = np.array(verts_local)
        
        min_dist = float('inf')
        for i in range(6):
            v1 = verts_local[i]
            v2 = verts_local[(i + 1) % 6]
            edge = v2 - v1
            # Inward normal
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            nx /= norm
            ny /= norm
            
            # The line equation is n · (p - v1) = 0 => n · p = n · v1
            c = nx * v1[0] + ny * v1[1]
            dist = nx * lx + ny * ly - c
            # If the point is inside, dist should be <= 0 (since n is inward, n·p <= c).
            # Wait, if n is inward, then for a point inside, n·(p - v1) <= 0 => n·p <= n·v1 = c.
            # So dist = n·p - c <= 0 for inside.
            # The distance to the edge is -dist (if inside).
            # If outside, dist > 0, and the distance is dist.
            # We want the signed distance: positive if inside, negative if outside.
            # So, signed_dist = -dist.
            min_dist = min(min_dist, -dist)
        
        return min_dist

    # This function is still non-smooth at the boundary.
    # Let's use a smooth approximation:
    # soft_min(a, b) = -log(exp(-a/t) + exp(-b/t)) * t
    # But we have 6 distances.
    # Let's just use the min, and rely on the optimizer to handle it.
    # Or, use a penalty that is smooth.
    # Penalty for containment: sum(max(0, -dist)^2)
    # This is smooth (C1).
    
    # Let's implement the smooth penalty objective.
    
    def _smooth_penalty(params, outer_cx, outer_cy, outer_angle, outer_side):
        # Unpack params
        inner_pos = params[:2*n].reshape(n, 2)
        inner_ang = params[2*n:3*n]
        
        penalty = 0.0
        
        # 1. Containment penalty
        for i in range(n):
            x, y = inner_pos[i]
            ang = inner_ang[i]
            verts = _hex_vertices(x, y, 1.0, ang)
            for v in verts:
                dist = _distance_to_hexagon_boundary(v[0], v[1], outer_cx, outer_cy, outer_side, outer_angle)
                if dist < 0:
                    penalty += dist * dist
        
        # 2. Overlap penalty
        # For each pair, compute the minimum distance between the two hexagons.
        # This is hard to compute smoothly.
        # Approximation: Check the distance between centers. If < 2, penalty.
        # But this is not accurate for rotated hexagons.
        # Let's use a simpler penalty: sum of squared overlaps.
        # Or, just use the SAT check and add a penalty if they overlap.
        # But SAT is non-smooth.
        
        # Let's try a different approach:
        # Instead of a penalty, let's use a hard constraint and use a different optimizer.
        # Or, let's just use the local search from the current program, but with a better initial guess.
        
        return penalty

    # This is getting complicated. Let's stick to the current strategy but improve the initial guess.
    # The current program uses a grid. Let's use a tighter grid.
    
    # For 11 hexagons, the optimal packing is likely a hexagonal cluster.
    # 1 in the center, 6 around it, 4 more?
    # 1 + 6 = 7. 7 + 4 = 11.
    # The 4 extra can be placed in the gaps.
    
    # Let's generate this specific configuration.
    sqrt3 = np.sqrt(3)
    pos = []
    ang = []
    
    # Center
    pos.append([0.0, 0.0])
    ang.append(0.0)
    
    # 6 around
    for i in range(6):
        theta = i * np.pi / 3
        x = 2.0 * np.cos(theta)
        y = 2.0 * np.sin(theta)
        pos.append([x, y])
        ang.append(0.0)
    
    # 4 more
    # Place them in the gaps of the outer ring.
    # The gaps are at angles 30, 90, 150, 210, 270, 330.
    # Distance from center to gap center:
    # The gap is between two adjacent hexagons in the outer ring.
    # The center of the gap is at distance 2 * cos(30) = sqrt(3) from the center of one hexagon?
    # No, the distance between centers of adjacent hexagons is 2.
    # The gap center is at the centroid of the triangle formed by the center and two adjacent hexagons.
    # The triangle has sides 2, 2, 2 (equilateral).
    # The centroid is at distance 2 / sqrt(3) from each vertex?
    # No, the altitude is sqrt(3). The centroid is at 2/3 of the altitude.
    # So distance from center = (2/3) * sqrt(3) = 2 * sqrt(3) / 3.
    # Wait, the distance from the center of the hexagon to the gap center.
    # The gap center is at the same distance from the 3 centers.
    # The 3 centers form an equilateral triangle of side 2.
    # The circumradius of this triangle is 2 / sqrt(3).
    # So the gap center is at distance 2 / sqrt(3) from the center of the hexagon.
    
    # But we need to place 4 hexagons in the gaps.
    # There are 6 gaps. We can pick 4 of them.
    # Let's pick the gaps at angles 0, 90, 180, 270.
    # Wait, the gaps are at angles 30, 90, 150, 210, 270, 330.
    # Let's pick 0, 120, 240, 300? No, the gaps are at 30, 90, etc.
    # Let's pick 90, 150, 210, 270.
    
    gap_angles = [90, 150, 210, 270]
    gap_dist = 2.0 / np.sqrt(3) + 1.0 # Distance from center to gap center + radius?
    # No, the distance from the center of the hexagon to the center of the gap is 2 / sqrt(3).
    # But the gap center is the center of the new hexagon.
    # So the new hexagon center is at distance 2 / sqrt(3) from the center of the hexagon.
    # But the center of the hexagon is at distance 2 from the origin.
    # So the new hexagon center is at distance 2 + 2 / sqrt(3) from the origin?
    # No, the gap is between two adjacent hexagons in the outer ring.
    # The center of the gap is at the same distance from the origin as the centers of the outer ring hexagons.
    # Because the triangle is equilateral.
    # So the gap center is at distance 2 from the origin.
    
    # Wait, the centers of the outer ring hexagons are at distance 2 from the origin.
    # The gap center is at the centroid of the triangle formed by the origin and two adjacent outer ring hexagons.
    # The triangle has vertices (0,0), (2,0), (1, sqrt(3)).
    # The centroid is at ((0+2+1)/3, (0+0+sqrt(3))/3) = (1, sqrt(3)/3).
    # The distance from the origin is sqrt(1 + 1/3) = sqrt(4/3) = 2 / sqrt(3).
    # So the gap center is at distance 2 / sqrt(3) from the origin.
    # But the outer ring hexagons are at distance 2.
    # So the gap center is CLOSER to the origin.
    # This means the new hexagon would overlap with the center hexagon.
    # Distance from center hexagon to gap center is 2 / sqrt(3) ~ 1.15.
    # The sum of radii is 1 + 1 = 2.
    # So they would overlap.
    
    # So we cannot place hexagons in the inner gaps.
    # We must place them in the OUTER gaps.
    # The outer gaps are between the outer ring hexagons and the boundary.
    # But we don't have a boundary yet.
    
    # Let's just use the grid from _initial_packing.
    pos, ang = _initial_packing()
    
    # Now, run the local search from the current program.
    # I will copy the local search code from the current program, but with a better initial step size.
    
    outer_cx, outer_cy = 0.0, 0.0
    outer_angle = 0.0
    
    current_side = _min_outer_side(pos, ang, (outer_cx, outer_cy), outer_angle)
    
    best_side = current_side
    best_pos = pos.copy()
    best_ang = ang.copy()
    best_ocx, best_ocy = outer_cx, outer_cy
    best_oor = outer_angle
    
    np.random.seed(42)
    step = 0.1
    iters = 0
    
    while time.time() < DEADLINE - 5.0:
        iters += 1
        
        improved = False
        for i in range(n):
            if time.time() > DEADLINE - 5.0:
                break
            
            for dx in [-step, 0.0, step]:
                for dy in [-step, 0.0, step]:
                    if dx == 0.0 and dy == 0.0:
                        continue
                    new_pos = pos.copy()
                    new_pos[i][0] += dx
                    new_pos[i][1] += dy
                    
                    verts = _hex_vertices(new_pos[i][0], new_pos[i][1], 1.0, ang[i])
                    inside = True
                    for v in verts:
                        if not _point_in_hexagon(v[0], v[1], outer_cx, outer_cy, current_side, outer_angle):
                            inside = False
                            break
                    if not inside:
                        continue
                    
                    overlap = False
                    for j in range(n):
                        if j == i:
                            continue
                        verts_j = _hex_vertices(pos[j][0], pos[j][1], 1.0, ang[j])
                        if _hexagon_overlap(verts, verts_j):
                            overlap = True
                            break
                    if overlap:
                        continue
                    
                    new_side = _min_outer_side(new_pos, ang, (outer_cx, outer_cy), outer_angle)
                    if new_side < current_side - 1e-8:
                        pos = new_pos
                        current_side = new_side
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            for i in range(n):
                if time.time() > DEADLINE - 5.0:
                    break
                for da in [-5.0, 5.0]:
                    new_ang = ang.copy()
                    new_ang[i] = ang[i] + da
                    new_side = _min_outer_side(pos, new_ang, (outer_cx, outer_cy), outer_angle)
                    if new_side < current_side - 1e-8:
                        verts = _hex_vertices(pos[i][0], pos[i][1], 1.0, new_ang[i])
                        inside = True
                        for v in verts:
                            if not _point_in_hexagon(v[0], v[1], outer_cx, outer_cy, current_side, outer_angle):
                                inside = False
                                break
                        if not inside:
                            continue
                        overlap = False
                        for j in range(n):
                            if j == i:
                                continue
                            verts_j = _hex_vertices(pos[j][0], pos[j][1], 1.0, ang[j])
                            if _hexagon_overlap(verts, verts_j):
                                overlap = True
                                break
                        if not overlap:
                            ang = new_ang
                            current_side = new_side
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            step *= 0.8
            if step < 0.001:
                break
        
        if current_side < best_side:
            best_side = current_side
            best_pos = pos.copy()
            best_ang = ang.copy()
            best_ocx, best_ocy = outer_cx, outer_cy
            best_oor = outer_angle
    
    # Final verification
    if not _check_validity(
        [[best_pos[i][0], best_pos[i][1], best_ang[i]] for i in range(N)],
        (best_ocx, best_ocy), best_side, best_oor
    ):
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner_list = [[float(best_pos[i][0]), float(best_pos[i][1]), float(best_ang[i])] for i in range(N)]
    if not _check_validity(inner_list, (best_ocx, best_ocy), best_side, best_oor):
        sqrt3 = np.sqrt(3)
        inner = [[4.0 * k, 0.0, 0.0] for k in range(N)]
        return inner, (0.0, 0.0), 40.0, 0.0
    return inner_list, (float(best_ocx), float(best_ocy)), float(best_side), float(best_oor)

# EVOLVE-BLOCK-END
