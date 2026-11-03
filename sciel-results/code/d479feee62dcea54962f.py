"""Direct hexagon packing: 11 unit hexagons in a hexagonal lattice pattern.
Uses LP for outer hexagon and Nelder-Mead for refinement with position perturbations."""
import math
import time
import itertools
import numpy as np
from scipy.optimize import minimize, linprog

N = 11


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length `side`,
    centered at (cx, cy), rotated by angle_deg degrees."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side `side` rotated by angle_deg."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * math.sqrt(3) / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True


def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 properly intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _hexagons_disjoint(hex1, hex2, eps=1e-9):
    """Check if two hexagons have disjoint interiors."""
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    r1 = s1
    r2 = s2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (r1 + r2) ** 2 + 1e-12:
        return True
    verts1 = _hex_vertices(cx1, cy1, s1, a1)
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    verts2 = _hex_vertices(cx2, cy2, s2, a2)
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    for i in range(6):
        v1a = verts1[i]
        v1b = verts1[(i + 1) % 6]
        for j in range(6):
            v2a = verts2[j]
            v2b = verts2[(j + 1) % 6]
            if _segments_intersect(v1a, v1b, v2a, v2b):
                return False
    return True


def _all_disjoint(inner_hexes, eps=1e-9):
    """Check if all pairs of hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True


def _inner_in_outer(inner_hex, outer_hex):
    """Check if inner hexagon is fully inside outer hexagon."""
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True


def _min_outer_side_lp(inner_hexes, oangle=0.0):
    """Find minimum side length of outer hexagon using LP.
    Returns (center_x, center_y, side_length)."""
    # Variables: [cx, cy, t] where t is the apothem of outer hex
    # Objective: minimize t
    c = [0, 0, 1]
    
    A_ub = []
    b_ub = []
    
    for cx_h, cy_h, s_h, ang_h in inner_hexes:
        verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
        for vx, vy in verts:
            for k in range(6):
                theta = math.radians(oangle + 30 + 60 * k)
                nx = math.cos(theta)
                ny = math.sin(theta)
                # (vx - cx)*nx + (vy - cy)*ny <= t
                # -nx*cx - ny*cy - t <= -vx*nx - vy*ny
                A_ub.append([-nx, -ny, -1])
                b_ub.append(-vx * nx - vy * ny)
    
    # Also need t >= 0
    A_ub.append([0, 0, -1])
    b_ub.append(0)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, 
                  bounds=[(None, None), (None, None), (0, None)], method='highs')
    
    if res.success:
        t = res.x[2]
        center = (res.x[0], res.x[1])
        side = t * 2.0 / math.sqrt(3)
        return center, side
    else:
        return None, float('inf')


def _min_outer_side_direct(inner_hexes, ox, oy, oangle=0.0):
    """Find minimum side length of outer hexagon centered at (ox, oy) with rotation oangle."""
    max_val = 0.0
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    for cx, cy, s, ang in inner_hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        for vx, vy in verts:
            dx = vx - ox
            dy = vy - oy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                proj = lx * math.cos(theta) + ly * math.sin(theta)
                if proj > max_val:
                    max_val = proj
    return max_val * 2.0 / math.sqrt(3)


def _separation_penalty(hexes, penalty_weight=100.0):
    """Compute a smooth penalty for overlapping hexagons.
    Uses minimum distance between hexagons as a proxy."""
    penalty = 0.0
    n = len(hexes)
    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1, s1, a1 = hexes[i]
            cx2, cy2, s2, a2 = hexes[j]
            dx = cx2 - cx1
            dy = cy2 - cy1
            dist = math.sqrt(dx * dx + dy * dy)
            # For two unit hexagons, they need center distance >= ~1.732 (sqrt(3)) for flat-top
            # but with rotation, the minimum distance can vary. Use a conservative bound.
            min_dist = 1.732  # sqrt(3), minimum center distance for non-overlapping unit hexagons
            if dist < min_dist:
                violation = min_dist - dist
                penalty += penalty_weight * violation * violation
    return penalty


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    # Flat-top hexagon mapping using axial coordinates:
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # Center + ring 1 (6 hexagons)
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    # Ring 2 positions
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    best_side = float('inf')
    best_positions = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_inner_hexes = None

    # Pre-compute all ring2 combinations
    all_combos = list(itertools.combinations(range(len(ring2)), 4))
    
    def combo_score(combo):
        """Score a combination: prefer contiguous arcs."""
        idxs = sorted(combo)
        gaps = []
        for i in range(len(idxs)):
            next_i = (i + 1) % len(idxs)
            gap = (idxs[next_i] - idxs[i]) % len(ring2)
            gaps.append(gap)
        return sum(gaps)
    
    all_combos.sort(key=combo_score)
    
    # Phase 1: Quick scan with coarse grid
    for combo in all_combos:
        if time.time() > deadline - 60.0:
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        
        # Try several outer hexagon orientations
        for oang in np.arange(0.0, 60.0, 3.0):
            center, side = _min_outer_side_lp(inner_hexes, oang)
            if center is not None and side < best_side:
                best_side = side
                best_positions = full_xy
                best_center = center
                best_angle = oang
                best_inner_hexes = inner_hexes
    
    if best_positions is None:
        # Fallback
        full_xy = [axial_to_xy(q, r) for q, r in pos_axial[:7]]
        best_positions = full_xy
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        center, side = _min_outer_side_lp(inner_hexes, 0.0)
        if center is not None:
            best_side = side
            best_center = center
            best_angle = 0.0
            best_inner_hexes = inner_hexes
        else:
            best_side = 4.0
            best_center = (0.0, 0.0)
            best_angle = 0.0
            best_inner_hexes = inner_hexes

    # Phase 2: Optimize outer hexagon parameters (center, angle) for fixed inner positions
    if time.time() < deadline - 30.0:
        inner_hexes = best_inner_hexes
        
        def objective_outer(params):
            cx, cy, ang = params
            # Normalize angle to [0, 60)
            ang = ang % 60.0
            center, side = _min_outer_side_lp(inner_hexes, ang)
            if center is None:
                return 100.0
            return side
        
        x0 = [best_center[0], best_center[1], best_angle]
        
        # Use L-BFGS-B for smooth optimization
        try:
            res = minimize(objective_outer, x0, method='L-BFGS-B',
                          bounds=[(-5, 5), (-5, 5), (0, 60)],
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.fun < best_side:
                best_side = res.fun
                best_center = (res.x[0], res.x[1])
                best_angle = res.x[2] % 60.0
        except Exception:
            pass
        
        # Try multiple starting angles
        if time.time() < deadline - 20.0:
            for start_ang in [0.0, 15.0, 30.0, 45.0]:
                if time.time() > deadline - 10.0:
                    break
                x0 = [best_center[0], best_center[1], start_ang]
                try:
                    res = minimize(objective_outer, x0, method='L-BFGS-B',
                                  bounds=[(-5, 5), (-5, 5), (0, 60)],
                                  options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14})
                    if res.fun < best_side:
                        best_side = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_angle = res.x[2] % 60.0
                except Exception:
                    pass

    # Phase 3: Refine inner positions with small perturbations
    if time.time() < deadline - 15.0 and best_positions is not None:
        # Allow small perturbations of inner positions
        base_positions = np.array(best_positions)
        n_hex = len(base_positions)
        
        def objective_combined(params):
            """Params: [dx1, dy1, dx2, dy2, ..., dx11, dy11, cx, cy, angle]"""
            # Extract perturbations
            dxy = params[:2*n_hex]
            cx, cy, ang = params[2*n_hex], params[2*n_hex+1], params[2*n_hex+2]
            
            # Apply perturbations (small, bounded)
            perturbed = []
            for i in range(n_hex):
                x = base_positions[i][0] + dxy[2*i]
                y = base_positions[i][1] + dxy[2*i+1]
                perturbed.append((x, y, 1.0, 0.0))
            
            # Compute outer hexagon
            center, side = _min_outer_side_lp(perturbed, ang)
            if center is None:
                return 100.0
            
            # Add penalty for overlapping hexagons
            penalty = _separation_penalty(perturbed, penalty_weight=50.0)
            
            return side + penalty
        
        # Initial params: zero perturbations + current best center/angle
        x0 = [0.0] * (2*n_hex) + [best_center[0], best_center[1], best_angle]
        
        # Bounds: small perturbations, center/angle reasonable
        bounds = [(-0.1, 0.1)] * (2*n_hex) + [(-5, 5), (-5, 5), (0, 60)]
        
        try:
            res = minimize(objective_combined, x0, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
            
            if res.fun < best_side + 0.01:  # Allow some slack for penalty
                # Extract results
                dxy = res.x[:2*n_hex]
                new_positions = []
                for i in range(n_hex):
                    x = base_positions[i][0] + dxy[2*i]
                    y = base_positions[i][1] + dxy[2*i+1]
                    new_positions.append((x, y))
                
                new_inner_hexes = [(x, y, 1.0, 0.0) for x, y in new_positions]
                
                # Verify disjointness
                if _all_disjoint(new_inner_hexes):
                    center, side = _min_outer_side_lp(new_inner_hexes, res.x[2*n_hex+2])
                    if center is not None and side < best_side:
                        best_side = side
                        best_positions = new_positions
                        best_center = center
                        best_angle = res.x[2*n_hex+2] % 60.0
                        best_inner_hexes = new_inner_hexes
        except Exception:
            pass

    # Final verification and adjustment
    if best_inner_hexes is not None:
        # Verify all inner hexagons are inside outer
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)
        valid = True
        for h in best_inner_hexes:
            if not _inner_in_outer(h, outer_hex):
                valid = False
                break
        
        if not valid:
            # Binary search for minimum valid side length
            lo, hi = best_side, best_side * 1.1
            for _ in range(50):
                mid = (lo + hi) / 2
                test_outer = (best_center[0], best_center[1], mid, best_angle)
                ok = all(_inner_in_outer(h, test_outer) for h in best_inner_hexes)
                if ok:
                    hi = mid
                else:
                    lo = mid
            best_side = hi

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]

    # Verify disjointness one more time
    if not _all_disjoint(inner_hexes):
        # Fallback to original lattice positions
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in [axial_to_xy(q, r) for q, r in pos_axial[:7]]]
        cx_mean = sum(x for x, y, s, a in inner_hexes) / 7.0
        cy_mean = sum(y for x, y, s, a in inner_hexes) / 7.0
        center, side = _min_outer_side_lp(inner_hexes, 0.0)
        if center is not None:
            best_side = side
            best_center = center
            best_angle = 0.0
        else:
            best_side = 3.0
            best_center = (0.0, 0.0)
            best_angle = 0.0

    # Format output
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
