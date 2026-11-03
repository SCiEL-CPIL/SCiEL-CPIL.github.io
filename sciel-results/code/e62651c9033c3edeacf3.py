"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

def _hex_vertices_batch(positions, angles, side=1.0):
    """Vectorized: return (N*6, 2) array of all vertices for N hexagons."""
    n = len(positions)
    pos = np.asarray(positions, dtype=np.float64)
    ang = np.radians(np.asarray(angles, dtype=np.float64))
    # Base vertex offsets for unit hexagon: angles 0, 60, 120, 180, 240, 300
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    # For each hexagon i, vertex j: (cx + s*cos(a_i + j*pi/3), cy + s*sin(a_i + j*pi/3))
    # Shape: (n, 6)
    total_angles = ang[:, None] + base_angles[None, :]
    offsets_x = side * np.cos(total_angles)
    offsets_y = side * np.sin(total_angles)
    # Broadcast positions
    verts_x = pos[:, 0:1] + offsets_x  # (n, 6)
    verts_y = pos[:, 1:2] + offsets_y  # (n, 6)
    return np.column_stack([verts_x.ravel(), verts_y.ravel()])  # (n*6, 2)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    s = side
    sqrt3 = np.sqrt(3)
    # Six half-plane constraints for hexagon with vertices at angles 0,60,...,300
    # Constraint 1: lx <= s
    # Constraint 2: lx >= -s
    # Constraint 3: sqrt3*lx + ly <= sqrt3*s
    # Constraint 4: -sqrt3*lx + ly <= sqrt3*s
    # Constraint 5: -sqrt3*lx - ly <= sqrt3*s
    # Constraint 6: sqrt3*lx - ly <= sqrt3*s
    if lx > s + 1e-10 or lx < -s - 1e-10:
        return False
    if sqrt3 * lx + ly > sqrt3 * s + 1e-10:
        return False
    if -sqrt3 * lx + ly > sqrt3 * s + 1e-10:
        return False
    if -sqrt3 * lx - ly > sqrt3 * s + 1e-10:
        return False
    if sqrt3 * lx - ly > sqrt3 * s + 1e-10:
        return False
    return True

def _min_side_vectorized(all_verts, cx, cy, outer_angle_deg):
    """
    Compute minimum outer side length S such that all vertices in all_verts are inside
    the outer hexagon centered at (cx,cy) with rotation outer_angle_deg.
    Uses the half-plane formulation: for a hexagon with side S, a point is inside iff
    all 6 linear constraints are satisfied. The required S is the max over all vertices
    and all 6 constraints of the required S for that constraint.
    """
    verts = np.asarray(all_verts, dtype=np.float64)
    n_pts = len(verts)
    if n_pts == 0:
        return 0.0
    
    dx = verts[:, 0] - cx
    dy = verts[:, 1] - cy
    
    a = np.radians(outer_angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    # Rotate into outer hexagon's local frame
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    # For each of the 6 half-planes, compute the required S:
    # Constraint 1: lx <= S       -> S >= lx
    # Constraint 2: lx >= -S      -> S >= -lx
    # Constraint 3: sqrt3*lx + ly <= sqrt3*S  -> S >= (sqrt3*lx + ly) / sqrt3 = lx + ly/sqrt3
    # Constraint 4: -sqrt3*lx + ly <= sqrt3*S -> S >= (-sqrt3*lx + ly) / sqrt3 = -lx + ly/sqrt3
    # Constraint 5: -sqrt3*lx - ly <= sqrt3*S -> S >= (-sqrt3*lx - ly) / sqrt3 = -lx - ly/sqrt3
    # Constraint 6: sqrt3*lx - ly <= sqrt3*S  -> S >= (sqrt3*lx - ly) / sqrt3 = lx - ly/sqrt3
    
    c1 = lx
    c2 = -lx
    c3 = lx + ly / sqrt3
    c4 = -lx + ly / sqrt3
    c5 = -lx - ly / sqrt3
    c6 = lx - ly / sqrt3
    
    # Required S is the maximum over all vertices and all 6 constraints, but at least 0
    required = np.maximum(
        np.maximum(np.maximum(c1, c2), np.maximum(c3, c4)),
        np.maximum(c5, c6)
    )
    return float(np.max(required))

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

def _check_no_overlap_fast(positions, angles, tol=1e-7):
    """Check that no two inner hexagons overlap. Returns True if valid."""
    n = len(positions)
    sqrt3 = np.sqrt(3)
    # For hexagons of side 1, the minimum center-to-center distance for non-overlap
    # depends on orientation. Use SAT for correctness.
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i]))
    for i in range(n):
        for j in range(i + 1, n):
            if _hexagon_overlap(verts_list[i], verts_list[j], tol):
                return False
    return True

def _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle, tol=1e-6):
    """Find minimum outer side length such that all inner hexagons fit inside the outer hexagon."""
    cx, cy = outer_center
    pos = np.asarray(inner_positions, dtype=np.float64)
    ang = np.asarray(inner_angles, dtype=np.float64)
    all_verts = _hex_vertices_batch(pos, ang, side=1.0)
    return _min_side_vectorized(all_verts, cx, cy, outer_angle)

def _optimal_outer_center_for_side(pos, ang, outer_angle, max_iter=50):
    """Iteratively find optimal outer center that minimizes required side length."""
    # Start with centroid
    cx, cy = np.mean(pos[:, 0]), np.mean(pos[:, 1])
    
    for _ in range(max_iter):
        # Compute required S for current center
        all_verts = _hex_vertices_batch(pos, ang, side=1.0)
        current_s = _min_side_vectorized(all_verts, cx, cy, outer_angle)
        
        # Try shifting center slightly in each direction
        best_s = current_s
        best_cx, best_cy = cx, cy
        step = 0.1
        for dx in [-step, 0, step]:
            for dy in [-step, 0, step]:
                if dx == 0 and dy == 0:
                    continue
                new_s = _min_side_vectorized(all_verts, cx + dx, cy + dy, outer_angle)
                if new_s < best_s - 1e-10:
                    best_s = new_s
                    best_cx, best_cy = cx + dx, cy + dy
        
        if best_cx == cx and best_cy == cy:
            break
        cx, cy = best_cx, best_cy
        step *= 0.7
    
    return cx, cy

def _distance_to_hex_edge(px, py, cx, cy, side, angle_deg):
    """Return signed distance from point to hexagon boundary (positive inside)."""
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    s = side
    sqrt3 = np.sqrt(3)
    
    d1 = s - lx
    d2 = s + lx
    d3 = (sqrt3 * s - (sqrt3 * lx + ly)) / 2.0
    d4 = (sqrt3 * s - (-sqrt3 * lx + ly)) / 2.0
    d5 = (sqrt3 * s - (-sqrt3 * lx - ly)) / 2.0
    d6 = (sqrt3 * s - (sqrt3 * lx - ly)) / 2.0
    
    return min(d1, d2, d3, d4, d5, d6)

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if packing is valid: no overlaps, all inside outer hexagon."""
    cx, cy = outer_center
    verts_list = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        verts_list.append(verts)
    
    # Check no overlaps
    for i in range(len(verts_list)):
        for j in range(i + 1, len(verts_list)):
            if _hexagon_overlap(verts_list[i], verts_list[j]):
                return False
    
    # Check all inside outer hexagon
    outer_verts = _hex_vertices(cx, cy, outer_side, outer_angle)
    for verts in verts_list:
        for v in verts:
            if not _point_in_hexagon(v[0], v[1], cx, cy, outer_side, outer_angle):
                return False
    return True

def _initial_packing():
    """Create initial hexagonal grid packing for N hexagons."""
    positions = []
    angles = []
    sqrt3 = np.sqrt(3)
    
    # Better initial configuration: use hexagonal grid with proper spacing
    # For pointy-top hexagons (vertices at 0,60,120,180,240,300 degrees):
    # Horizontal spacing between centers: 2*sqrt(3)/2 * 2 = 2*sqrt(3) ≈ 3.464 for adjacent columns
    # Actually, for unit hexagons with vertices at angles 0,60,...:
    # The "radius" (center to vertex) = 1
    # The "apothem" (center to edge) = sqrt(3)/2
    # For non-overlapping: center distance >= 2 * apothem = sqrt(3) in perpendicular direction
    # For hex grid: 
    #   Same row: spacing = 2 * apothem * 2 = 2*sqrt(3) if flat-top, or 2 if pointy-top
    # Let me use the standard hex grid for pointy-top hexagons
    
    # Pointy-top hex grid:
    # Row y-spacing: 1.5 (for unit side length hexagons)
    # Column x-spacing: sqrt(3)
    # Odd rows offset by sqrt(3)/2
    
    # Actually for side length 1 hexagons:
    # Width = 2 (vertex to vertex horizontally for pointy-top)
    # Height = sqrt(3) (vertex to vertex vertically for pointy-top)
    # For non-overlapping pointy-top hexagons:
    #   Horizontal spacing: sqrt(3)
    #   Vertical spacing: 1.5
    #   Odd rows offset: sqrt(3)/2
    
    # Let's build a proper hex grid
    row_configs = [
        # (row_y, [x_offsets])
        (0, [0]),
        (1.5, [-sqrt3/2, sqrt3/2]),
        (3.0, [-sqrt3, 0, sqrt3]),
        (4.5, [-sqrt3/2, sqrt3/2]),
        (6.0, [-sqrt3, 0, sqrt3]),
    ]
    
    # Select positions to get exactly N hexagons
    all_positions = []
    for y, x_offsets in row_configs:
        for x_off in x_offsets:
            all_positions.append([x_off, y])
    
    # Take first N
    for i in range(min(N, len(all_positions))):
        positions.append(all_positions[i])
        angles.append(0.0)
    
    # If we need more, add to the next row
    if len(positions) < N:
        next_y = row_configs[-1][0] + 1.5
        x_offs = [-sqrt3/2, sqrt3/2] if (len(row_configs) % 2 == 0) else [-sqrt3, 0, sqrt3]
        for x_off in x_offs:
            if len(positions) >= N:
                break
            positions.append([x_off, next_y])
            angles.append(0.0)
    
    # Center the configuration
    if len(positions) > 0:
        cx = np.mean([p[0] for p in positions])
        cy = np.mean([p[1] for p in positions])
        positions = [[p[0] - cx, p[1] - cy] for p in positions]
    
    return np.array(positions, dtype=np.float64), np.array(angles, dtype=np.float64)

def _smooth_objective(params, outer_center, outer_angle, beta=20.0):
    """
    Smooth log-sum-exp objective that approximates the minimum outer side length.
    params: flat array [x_0, y_0, ..., x_{n-1}, y_{n-1}, ang_0, ..., ang_{n-1}]
    Uses the analytic half-plane formulation with a soft-max.
    """
    n = N
    pos = params[:2*n].reshape(n, 2)
    ang = params[2*n:]
    cx, cy = outer_center
    
    # Compute all vertices
    all_verts = _hex_vertices_batch(pos, ang, side=1.0)
    
    # Transform to outer hexagon local frame
    dx = all_verts[:, 0] - cx
    dy = all_verts[:, 1] - cy
    a = np.radians(outer_angle)
    ca, sa = np.cos(a), np.sin(a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    # 6 constraint values (required S for each)
    c1 = lx
    c2 = -lx
    c3 = lx + ly / sqrt3
    c4 = -lx + ly / sqrt3
    c5 = -lx - ly / sqrt3
    c6 = lx - ly / sqrt3
    
    # Stack all constraints: shape (6, n*6)
    all_c = np.vstack([c1, c2, c3, c4, c5, c6])
    
    # Soft max via log-sum-exp
    max_c = np.max(all_c)
    if max_c > 500.0 / beta:
        return float(max_c)
    shifted = all_c - max_c
    smooth_max = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    # Smooth overlap penalty using center distances
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx_c = pos[i][0] - pos[j][0]
            dy_c = pos[i][1] - pos[j][1]
            dist_c = np.sqrt(dx_c*dx_c + dy_c*dy_c)
            if dist_c < 2.0:
                overlap_penalty += (2.0 - dist_c) ** 2
    
    return float(smooth_max) + 100.0 * overlap_penalty

def _min_side_from_params(params, outer_center, outer_angle):
    """Compute exact minimum outer side from parameter vector."""
    n = N
    pos = params[:2*n].reshape(n, 2)
    ang = params[2*n:]
    return _min_outer_side(pos, ang, outer_center, outer_angle)

def _get_optimal_outer_center(pos, outer_angle):
    """Find optimal outer hexagon center for given inner positions and outer angle."""
    # Use centroid as a good starting point
    cx, cy = np.mean(pos[:, 0]), np.mean(pos[:, 1])
    return cx, cy

def construct():
    global START_TIME, DEADLINE
    
    # Initial packing
    pos, ang = _initial_packing()
    n = N
    
    # Outer hexagon center and angle - use centroid
    outer_cx, outer_cy = _get_optimal_outer_center(pos, 0.0)
    outer_angle = 0.0
    
    # Initial side estimate
    current_side = _min_outer_side(pos, ang, (outer_cx, outer_cy), outer_angle)
    
    best_side = current_side
    best_pos = pos.copy()
    best_ang = ang.copy()
    best_ocx, best_ocy = outer_cx, outer_cy
    best_oor = outer_angle
    
    # Flat parameter vector
    best_params = np.concatenate([best_pos.flatten(), best_ang])
    
    # Phase 1: Smooth optimization using log-sum-exp surrogate
    if time.time() < DEADLINE - 40.0:
        try:
            def smooth_obj(params):
                if time.time() > DEADLINE - 15.0:
                    return best_side + 1000.0
                p = params[:2*n].reshape(n, 2)
                a = params[2*n:]
                # Use centroid for speed in smooth objective
                ocx, ocy = np.mean(p[:, 0]), np.mean(p[:, 1])
                return _smooth_objective(params, (ocx, ocy), 0.0, beta=20.0)
            
            result = minimize(
                smooth_obj,
                best_params.copy(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
            )
            # Evaluate the actual (non-smooth) side at the result
            if result.x is not None:
                cand_pos = result.x[:2*n].reshape(n, 2)
                cand_ang = result.x[2*n:]
                # Use optimized center
                cand_ocx, cand_ocy = _optimal_outer_center_for_side(cand_pos, cand_ang, 0.0)
                candidate_side = _min_outer_side(cand_pos, cand_ang, (cand_ocx, cand_ocy), 0.0)
                if _check_no_overlap_fast(cand_pos, cand_ang):
                    if candidate_side < best_side:
                        best_side = candidate_side
                        best_pos = cand_pos.copy()
                        best_ang = cand_ang.copy()
                        best_ocx, best_ocy = cand_ocx, cand_ocy
                        best_params = result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local search with adaptive step
    step = 0.1
    min_step = 0.001
    while step >= min_step and time.time() < DEADLINE - 10.0:
        improved = False
        for i in range(n):
            if time.time() > DEADLINE - 10.0:
                break
            for dx in [-step, 0.0, step]:
                if time.time() > DEADLINE - 10.0:
                    break
                for dy in [-step, 0.0, step]:
                    if time.time() > DEADLINE - 10.0:
                        break
                    if dx == 0.0 and dy == 0.0:
                        continue
                    new_pos = best_pos.copy()
                    new_pos[i, 0] += dx
                    new_pos[i, 1] += dy
                    
                    if not _check_no_overlap_fast(new_pos, best_ang):
                        continue
                    
                    # Use optimized center for evaluation
                    new_ocx, new_ocy = _optimal_outer_center_for_side(new_pos, best_ang, 0.0)
                    new_side = _min_outer_side(new_pos, best_ang, (new_ocx, new_ocy), 0.0)
                    if new_side < best_side - 1e-9:
                        best_pos = new_pos
                        best_side = new_side
                        best_ocx, best_ocy = new_ocx, new_ocy
                        best_params = np.concatenate([best_pos.flatten(), best_ang])
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            step *= 0.6
    
    # Phase 3: High-precision polish
    if time.time() < DEADLINE - 5.0:
        fine_step = 0.01
        for _ in range(5):
            if time.time() > DEADLINE - 5.0:
                break
            improved = False
            for i in range(n):
                if time.time() > DEADLINE - 5.0:
                    break
                for dx in [-fine_step, 0.0, fine_step]:
                    if time.time() > DEADLINE - 5.0:
                        break
                    for dy in [-fine_step, 0.0, fine_step]:
                        if time.time() > DEADLINE - 5.0:
                            break
                        if dx == 0.0 and dy == 0.0:
                            continue
                        new_pos = best_pos.copy()
                        new_pos[i, 0] += dx
                        new_pos[i, 1] += dy
                        if not _check_no_overlap_fast(new_pos, best_ang):
                            continue
                        new_ocx, new_ocy = _optimal_outer_center_for_side(new_pos, best_ang, 0.0)
                        new_side = _min_outer_side(new_pos, best_ang, (new_ocx, new_ocy), 0.0)
                        if new_side < best_side - 1e-10:
                            best_pos = new_pos
                            best_side = new_side
                            best_ocx, best_ocy = new_ocx, new_ocy
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                fine_step *= 0.5
            if fine_step < 0.0005:
                break
    
    # Final optimization of outer center
    final_ocx, final_ocy = _optimal_outer_center_for_side(best_pos, best_ang, best_oor)
    final_side = _min_outer_side(best_pos, best_ang, (final_ocx, final_ocy), best_oor)
    
    # Final verification
    inner_list = [[float(best_pos[i][0]), float(best_pos[i][1]), float(best_ang[i])] for i in range(N)]
    if not _check_validity(inner_list, (final_ocx, final_ocy), final_side, best_oor):
        inner_list = [[float(pos[i][0]), float(pos[i][1]), float(ang[i])] for i in range(N)]
        fb_ocx, fb_ocy = _optimal_outer_center_for_side(pos, ang, 0.0)
        best_side = _min_outer_side(pos, ang, (fb_ocx, fb_ocy), 0.0)
        best_ocx, best_ocy = fb_ocx, fb_ocy
        best_oor = 0.0
        if not _check_validity(inner_list, (best_ocx, best_ocy), best_side, best_oor):
            inner_list = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner_list, (0.0, 0.0), 40.0, 0.0
        return inner_list, (float(best_ocx), float(best_ocy)), float(best_side), float(best_oor)
    
    return inner_list, (float(final_ocx), float(final_ocy)), float(final_side), float(best_oor)

# EVOLVE-BLOCK-END
