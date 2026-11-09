"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import math
import time
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of regular hexagon with given center, side, rotation."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts

    def compute_min_side(inner_positions, outer_center, outer_angle):
        """
        Compute minimum outer side to contain all inner hexagons.
        The outer hexagon has center (cx, cy), side length S, and rotation outer_angle.
        A point (px, py) is inside the outer hexagon iff for all 6 outward normals n_k,
        the projection of (px-cx, py-cy) rotated by -outer_angle onto n_k is <= S * sqrt(3)/2.
        So the required side is max over all points and all normals of proj / (sqrt(3)/2).
        """
        cx, cy = outer_center
        max_side = 0.0
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)

        # Pre-compute the 6 outward normals of a unit hexagon (inradius = sqrt(3)/2)
        # The normals point at angles 30, 90, 150, 210, 270, 330 degrees
        normals = []
        for k in range(6):
            theta = math.radians(30 + k * 60)
            normals.append((math.cos(theta), math.sin(theta)))

        for ix, iy, iang in inner_positions:
            inner_verts = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                dx = vx - cx
                dy = vy - cy
                # Rotate to outer hexagon's local frame
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a

                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    # Required side: proj / (sqrt(3)/2)
                    required_side = proj / (sqrt3 / 2)
                    if required_side > max_side:
                        max_side = required_side
        return max_side + 1e-10

    def hex_distance(p1, p2):
        """
        Exact minimum distance between two unit regular hexagons.
        Returns 0 if they overlap, positive if separated.
        Uses Separating Axis Theorem on the convex polygons.
        """
        # Get vertices of both hexagons
        verts1 = hexagon_vertices(p1[0], p1[1], 1.0, p1[2])
        verts2 = hexagon_vertices(p2[0], p2[1], 1.0, p2[2])

        # Collect all edge normals (candidate separating axes)
        axes = []
        for i in range(6):
            for verts in (verts1, verts2):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % 6]
                dx = x2 - x1
                dy = y2 - y1
                # Normal to this edge
                length = math.sqrt(dx * dx + dy * dy)
                if length > 1e-12:
                    axes.append((-dy / length, dx / length))

        # Also add the direction between centers as a candidate axis
        dx_c = p2[0] - p1[0]
        dy_c = p2[1] - p1[1]
        dist_c = math.sqrt(dx_c * dx_c + dy_c * dy_c)
        if dist_c > 1e-12:
            axes.append((dx_c / dist_c, dy_c / dist_c))

        # For each axis, project both polygons and check for separation
        for nx, ny in axes:
            min1 = min(vx * nx + vy * ny for vx, vy in verts1)
            max1 = max(vx * nx + vy * ny for vx, vy in verts1)
            min2 = min(vx * nx + vy * ny for vx, vy in verts2)
            max2 = max(vx * nx + vy * ny for vx, vy in verts2)

            # If there's a gap, the distance is the gap
            gap = max(min2 - max1, min1 - max2)
            if gap > 0:
                return gap

        # No separating axis found -> they overlap
        return 0.0

    def check_overlap(p1, p2):
        """
        Return overlap penalty: 0 if disjoint, positive if overlapping.
        Uses exact SAT-based distance. If distance is 0, they overlap.
        We need a measure of how much they overlap. For the penalty,
        if they are disjoint (distance > 0), penalty is 0.
        If they overlap, we estimate the penetration depth.
        """
        # First quick check: if center distance > 2.0, definitely disjoint
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d = math.sqrt(dx * dx + dy * dy)
        if d >= 2.0:
            return 0.0

        # Use SAT to find minimum distance
        dist = hex_distance(p1, p2)
        if dist > 0:
            return 0.0
        else:
            # They overlap. Estimate penetration depth.
            # A rough estimate: if centers are close, penetration is significant.
            # Use (2.0 - d) as a proxy for penetration depth, clamped to be positive.
            # This is not exact but drives the optimizer to separate them.
            return max(0.0, 2.0 - d)

    def total_penalty(inner_positions):
        """Sum of overlap penalties for all pairs."""
        penalty = 0.0
        n = len(inner_positions)
        for i in range(n):
            for j in range(i + 1, n):
                penalty += check_overlap(inner_positions[i], inner_positions[j])
        return penalty

    def is_disjoint(p1, p2, tol=1e-9):
        """Check if two hexagons are disjoint (no overlap)."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d = math.sqrt(dx * dx + dy * dy)
        if d >= 2.0 - tol:
            return True
        return hex_distance(p1, p2) > tol

    def all_disjoint(positions, tol=1e-9):
        """Check if all hexagons are pairwise disjoint."""
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                if not is_disjoint(positions[i], positions[j], tol):
                    return False
        return True

    # ---- Discrete combinatorial approach using hexagonal lattice ----
    # Generate candidate positions on a hexagonal lattice
    # For a hexagon of side S, the inradius is S*sqrt(3)/2
    # We'll try various outer side lengths and find valid packings

    def generate_lattice_positions(max_radius, step=0.5):
        """Generate candidate positions on a triangular lattice."""
        positions = []
        # Use axial coordinates for hex grid
        for q in range(-int(max_radius / step), int(max_radius / step) + 1):
            for r in range(-int(max_radius / step), int(max_radius / step) + 1):
                # Convert axial to pixel coordinates
                x = step * (q + r / 2.0)
                y = step * (sqrt3 * r)
                if math.sqrt(x*x + y*y) <= max_radius + 1.0:
                    positions.append((x, y))
        return positions

    def try_discrete_packing(outer_side, outer_angle=0.0):
        """
        Try to place 11 hexagons on a discrete lattice inside the outer hexagon.
        Returns (positions, success) where positions is list of [x,y,angle].
        """
        inradius = outer_side * sqrt3 / 2
        
        # Generate candidate positions within the outer hexagon
        max_r = inradius + 1.0
        candidates = generate_lattice_positions(max_r, step=0.5)
        
        # Filter candidates that fit inside the outer hexagon
        # A unit hexagon centered at (x,y) with angle a fits if all its vertices are inside
        valid_candidates = []
        for x, y in candidates:
            # Check if a unit hexagon at this position (angle 0) fits
            verts = hexagon_vertices(x, y, 1.0, 0.0)
            fits = True
            for vx, vy in verts:
                # Check if vertex is inside outer hexagon
                dx = vx
                dy = vy
                # Rotate to outer frame
                angle_rad = math.radians(outer_angle)
                cos_a = math.cos(-angle_rad)
                sin_a = math.sin(-angle_rad)
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                
                # Check against all 6 normals
                for k in range(6):
                    theta = math.radians(30 + k * 60)
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    proj = rx * nx + ry * ny
                    if proj > inradius + 1e-9:
                        fits = False
                        break
                if not fits:
                    break
            if fits:
                valid_candidates.append((x, y))
        
        if len(valid_candidates) < N:
            return None, False
        
        # Greedy placement with backtracking
        # Sort candidates by distance from center (prefer central positions)
        valid_candidates.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
        
        # Try to place hexagons greedily
        placed = []
        used = set()
        
        # First, try to place a central hexagon
        # Find candidate closest to origin
        center_idx = 0
        min_dist = float('inf')
        for i, (x, y) in enumerate(valid_candidates):
            d = x*x + y*y
            if d < min_dist:
                min_dist = d
                center_idx = i
        
        placed.append([valid_candidates[center_idx][0], valid_candidates[center_idx][1], 0.0])
        used.add(center_idx)
        
        # Place remaining 10 hexagons greedily
        for _ in range(N - 1):
            best_idx = -1
            best_score = float('inf')
            for i, (x, y) in enumerate(valid_candidates):
                if i in used:
                    continue
                # Check if this position is disjoint from all placed
                candidate = [x, y, 0.0]
                disjoint = True
                for p in placed:
                    if not is_disjoint(candidate, p):
                        disjoint = False
                        break
                if disjoint:
                    # Score: prefer positions close to center
                    score = x*x + y*y
                    if score < best_score:
                        best_score = score
                        best_idx = i
            
            if best_idx == -1:
                return None, False
            
            placed.append([valid_candidates[best_idx][0], valid_candidates[best_idx][1], 0.0])
            used.add(best_idx)
        
        # Verify all disjoint
        if not all_disjoint(placed):
            return None, False
        
        return placed, True

    def local_refinement(positions, outer_angle=0.0, max_iter=2000):
        """
        Local refinement: try small perturbations to reduce outer side.
        Returns improved positions or original if no improvement.
        """
        current_positions = [list(p) for p in positions]
        current_side = compute_min_side(current_positions, (0.0, 0.0), outer_angle)
        best_positions = [list(p) for p in current_positions]
        best_side = current_side
        
        temp = 0.5
        cooling = 0.995
        min_temp = 0.01
        
        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            
            # Perturb a random hexagon
            idx = random.randint(0, N - 1)
            old_x, old_y, old_ang = current_positions[idx]
            
            # Small perturbation
            dx = random.gauss(0, 0.1)
            dy = random.gauss(0, 0.1)
            da = random.gauss(0, 5.0)
            
            new_x = old_x + dx
            new_y = old_y + dy
            new_ang = old_ang + da
            
            # Check if still disjoint from others
            candidate = [new_x, new_y, new_ang]
            disjoint = True
            for j in range(N):
                if j != idx and not is_disjoint(candidate, current_positions[j]):
                    disjoint = False
                    break
            
            if not disjoint:
                continue
            
            # Compute new side
            trial_positions = [list(p) for p in current_positions]
            trial_positions[idx] = candidate
            trial_side = compute_min_side(trial_positions, (0.0, 0.0), outer_angle)
            
            delta = trial_side - current_side
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_positions[idx] = candidate
                current_side = trial_side
                if current_side < best_side:
                    best_side = current_side
                    best_positions = [list(p) for p in current_positions]
            
            temp *= cooling
        
        return best_positions, best_side

    # ---- Main optimization strategy ----
    # Try various outer side lengths from small to large
    # Use discrete packing to find valid configurations, then refine
    
    best_inner = None
    best_side = float('inf')
    best_angle = 0.0
    
    # Try a range of outer side lengths
    # Theoretical lower bound is around sqrt(3) ≈ 1.732
    # Practical minimum for 11 hexagons is likely around 2.5-3.5
    
    side_candidates = []
    for s in [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]:
        side_candidates.append(s)
    
    # Also try finer steps
    for s in np.arange(2.0, 4.0, 0.1):
        side_candidates.append(float(s))
    
    # Sort and deduplicate
    side_candidates = sorted(set(side_candidates))
    
    for side in side_candidates:
        if time.time() > deadline - 30.0:
            break
        
        # Try different outer angles
        for outer_angle in [0, 15, 30, 45]:
            if time.time() > deadline - 30.0:
                break
            
            positions, success = try_discrete_packing(side, outer_angle)
            if success:
                # Refine locally
                refined_positions, refined_side = local_refinement(positions, outer_angle, max_iter=1000)
                
                if refined_side < best_side:
                    best_side = refined_side
                    best_inner = refined_positions
                    best_angle = outer_angle
                
                # If we found a good packing, we can try smaller sides
                if refined_side < side + 0.1:
                    break
    
    # If no discrete packing found, fall back to continuous optimization
    if best_inner is None:
        # Use the rigid cluster approach as fallback
        def build_cluster(params):
            ring_r = params[0]
            positions = [[0.0, 0.0, 0.0]]
            for k in range(6):
                ang = math.radians(60 * k)
                x = ring_r * math.cos(ang)
                y = ring_r * math.sin(ang)
                positions.append([x, y, 0.0])
            for g in range(4):
                ang_deg = params[1 + 2*g]
                dist = params[2 + 2*g]
                ang = math.radians(ang_deg)
                x = dist * math.cos(ang)
                y = dist * math.sin(ang)
                positions.append([x, y, 0.0])
            return positions
        
        def objective(params, outer_angle=0.0):
            positions = build_cluster(params)
            penalty = 0.0
            n = len(positions)
            for i in range(n):
                for j in range(i + 1, n):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    d = math.sqrt(dx * dx + dy * dy)
                    if d < 2.0:
                        pen = (2.0 - d) ** 2
                        penalty += pen
            side = compute_min_side(positions, (0.0, 0.0), outer_angle)
            return side + 100.0 * penalty
        
        def nelder_mead(f, x0, tol=1e-8, max_iter=5000, step=0.5):
            n = len(x0)
            simplex = [list(x0)]
            for i in range(n):
                pt = list(x0)
                pt[i] = pt[i] + step if pt[i] != 0 else step
                simplex.append(pt)
            values = [f(pt) for pt in simplex]
            for it in range(max_iter):
                order = sorted(range(n + 1), key=lambda i: values[i])
                simplex = [simplex[i] for i in order]
                values = [values[i] for i in order]
                if values[-1] - values[0] < tol:
                    break
                centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
                xr = [centroid[j] + step * (centroid[j] - simplex[-1][j]) for j in range(n)]
                fr = f(xr)
                if fr < values[0]:
                    xe = [centroid[j] + 2 * step * (centroid[j] - simplex[-1][j]) for j in range(n)]
                    fe = f(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        values[-1] = fe
                    else:
                        simplex[-1] = xr
                        values[-1] = fr
                elif fr < values[-2]:
                    simplex[-1] = xr
                    values[-1] = fr
                else:
                    xc = [centroid[j] + 0.5 * step * (simplex[-1][j] - centroid[j]) for j in range(n)]
                    fc = f(xc)
                    if fc < values[-1]:
                        simplex[-1] = xc
                        values[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                            values[i] = f(simplex[i])
            best_idx = min(range(n + 1), key=lambda i: values[i])
            return simplex[best_idx]
        
        ring_r0 = sqrt3 + 0.1
        gap_dist0 = 2 * ring_r0 * math.cos(math.pi / 6) + 0.1
        initial_params = [ring_r0, 30.0, gap_dist0, 150.0, gap_dist0, 210.0, gap_dist0, 330.0, gap_dist0]
        
        if time.time() < deadline - 20.0:
            nm_result = nelder_mead(objective, initial_params, tol=1e-7, max_iter=3000, step=0.3)
            best_params = nm_result
            best_inner = build_cluster(best_params)
            best_side = compute_min_side(best_inner, (0.0, 0.0), 0.0)
            best_angle = 0.0
        
        # Optimize outer angle
        if best_inner is not None and time.time() < deadline - 5.0:
            for outer_angle in range(0, 60, 3):
                side = compute_min_side(best_inner, (0.0, 0.0), outer_angle)
                if side < best_side:
                    best_side = side
                    best_angle = outer_angle
    
    # Final verification and cleanup
    if best_inner is not None:
        # Verify no overlap
        if not all_disjoint(best_inner):
            # Fallback to safe configuration
            safe_inner = []
            for i in range(N):
                if i == 0:
                    safe_inner.append([0.0, 0.0, 0.0])
                else:
                    r = 2.1 * (i // 6 + 1)
                    angle = math.radians(60 * (i % 6))
                    x = r * math.cos(angle)
                    y = r * math.sin(angle)
                    safe_inner.append([x, y, 0.0])
            best_inner = safe_inner
            best_side = compute_min_side(safe_inner, (0.0, 0.0), 0.0)
            best_angle = 0.0
    
    # Ensure we have a valid result
    if best_inner is None:
        # Absolute fallback
        best_inner = []
        for i in range(N):
            if i == 0:
                best_inner.append([0.0, 0.0, 0.0])
            else:
                r = 2.5 * (i // 6 + 1)
                angle = math.radians(60 * (i % 6))
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                best_inner.append([x, y, 0.0])
        best_side = compute_min_side(best_inner, (0.0, 0.0), 0.0)
        best_angle = 0.0
    
    return best_inner, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
