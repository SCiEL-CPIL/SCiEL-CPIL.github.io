"""Combined hexagon packing: circle-packing heuristic with exact hexagon verification."""
import numpy as np
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s limit
    
    sqrt3 = math.sqrt(3)
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of regular hexagon with given center, side, rotation."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside hexagon centered at (cx,cy) with side, angle."""
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        
        apothem = side * sqrt3 / 2
        for k in range(6):
            theta = math.radians(30 + k * 60)
            nx = math.cos(theta)
            ny = math.sin(theta)
            if rx * nx + ry * ny > apothem + 1e-9:
                return False
        return True
    
    def hexagons_disjoint(i1, i2, threshold=1e-6):
        """Check if two unit hexagons (center, angle) are disjoint (no overlap).
        Uses vertex containment and edge intersection checks."""
        x1, y1, a1 = i1
        x2, y2, a2 = i2
        
        dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        # Quick reject: if centers are far enough, definitely disjoint
        if dist > 2.0 + threshold:
            return True
        # Quick accept for same orientation: need dist >= 2
        if abs(a1 - a2) < 1e-6:
            return dist >= 2.0 - threshold
        
        # Check if any vertex of hex1 is inside hex2 or vice versa
        v1 = hexagon_vertices(x1, y1, 1.0, a1)
        v2 = hexagon_vertices(x2, y2, 1.0, a2)
        
        for vx, vy in v1:
            if point_in_hexagon(vx, vy, x2, y2, 1.0, a2):
                return False
        for vx, vy in v2:
            if point_in_hexagon(vx, vy, x1, y1, 1.0, a1):
                return False
        
        # Check edge intersections
        for i in range(6):
            ax, ay = v1[i]
            bx, by = v1[(i+1) % 6]
            for j in range(6):
                cx, cy = v2[j]
                dx, dy = v2[(j+1) % 6]
                if segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                    return False
        return True
    
    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4."""
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        
        a, b, c, d = (p1, p2, p3, p4)
        d1 = cross(c, d, a)
        d2 = cross(c, d, b)
        d3 = cross(a, b, c)
        d4 = cross(a, b, d)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
    
    def all_disjoint(inner_list, threshold=1e-6):
        """Check all pairs of inner hexagons are disjoint."""
        n = len(inner_list)
        for i in range(n):
            for j in range(i+1, n):
                if not hexagons_disjoint(inner_list[i], inner_list[j], threshold):
                    return False
        return True
    
    def min_outer_side(inner_list, outer_center, outer_angle):
        """Binary search for minimum outer side that contains all inner hexagons."""
        cx, cy = outer_center
        lo = 1.0
        hi = 30.0
        
        def fits(side):
            for ix, iy, iang in inner_list:
                verts = hexagon_vertices(ix, iy, 1.0, iang)
                for vx, vy in verts:
                    if not point_in_hexagon(vx, vy, cx, cy, side, outer_angle):
                        return False
            return True
        
        if not fits(hi):
            return None
        if fits(lo):
            return lo
        
        for _ in range(40):
            mid = (lo + hi) / 2
            if fits(mid):
                hi = mid
            else:
                lo = mid
        return hi + 1e-10
    
    def evaluate_packing(inner_list, outer_center, outer_angle):
        """Evaluate a packing configuration. Returns (side, valid) or (None, False)."""
        if not all_disjoint(inner_list):
            return None, False
        side = min_outer_side(inner_list, outer_center, outer_angle)
        if side is None:
            return None, False
        return side, True
    
    # --- Strategy 1: Hexagonal lattice cluster (seed approach, improved) ---
    def axial_to_cartesian(q, r):
        x = 2.0 * q + r
        y = sqrt3 * r
        return x, y
    
    def hex_ring_positions(ring):
        if ring == 0:
            return [(0, 0)]
        positions = []
        q, r = ring, 0
        for side_idx in range(6):
            for step in range(ring):
                positions.append((q, r))
                dq = [1, 0, -1, -1, 0, 1][side_idx]
                dr = [0, -1, -1, 0, 1, 1][side_idx]
                q += dq
                r += dr
        return positions
    
    # Generate hexagonal cluster
    all_positions = [(0, 0)]
    ring = 1
    while len(all_positions) < N:
        all_positions.extend(hex_ring_positions(ring))
        ring += 1
    
    selected = all_positions[:N]
    
    # Try multiple configurations: different outer angles, centered vs not
    best_side = float('inf')
    best_result = None
    
    # Configuration 1: Original positions, centered
    for outer_angle in [0.0, 30.0, 15.0, 45.0, 75.0]:
        if time.time() > deadline - 10:
            break
        inner = []
        for q, r in selected:
            x, y = axial_to_cartesian(q, r)
            inner.append([x, y, 0.0])
        
        # Center the cluster
        cx_mean = sum(p[0] for p in inner) / len(inner)
        cy_mean = sum(p[1] for p in inner) / len(inner)
        centered_inner = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in inner]
        
        side, valid = evaluate_packing(centered_inner, (0.0, 0.0), outer_angle)
        if valid and side < best_side:
            best_side = side
            best_result = (centered_inner, (0.0, 0.0), side, outer_angle)
    
    # --- Strategy 2: Circle-packing-based force-directed optimization ---
    # Use circumcircle radius 1 as proxy, then verify with exact hexagon checks
    
    def force_directed_optimize(initial_positions, outer_center, outer_angle, max_iters=2000):
        """Optimize positions using force-directed method."""
        n = len(initial_positions)
        positions = [list(p[:2]) for p in initial_positions]
        angles = [p[2] for p in initial_positions]
        
        # Initial positions
        pos = np.array(positions, dtype=float)
        
        # Force-directed relaxation
        lr = 0.1
        decay = 0.999
        min_dist = 2.0  # minimum center distance for same orientation
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            forces = np.zeros_like(pos)
            
            # Repulsion between inner hexagons
            for i in range(n):
                for j in range(i+1, n):
                    diff = pos[i] - pos[j]
                    dist = np.linalg.norm(diff)
                    if dist < min_dist - 0.5:
                        if dist < 1e-6:
                            diff = np.random.randn(2) * 0.01
                            dist = np.linalg.norm(diff)
                        force = (min_dist - dist) * 2.0
                        forces[i] += force * diff / dist
                        forces[j] -= force * diff / dist
            
            # Attraction to center (to keep cluster compact)
            cx, cy = outer_center
            for i in range(n):
                to_center = np.array([cx, cy]) - pos[i]
                dist_to_center = np.linalg.norm(to_center)
                if dist_to_center > 1e-6:
                    forces[i] += 0.1 * to_center / dist_to_center
            
            # Update positions
            pos += lr * forces
            lr *= decay
            
            # Check if converged
            if np.max(np.abs(forces)) < 1e-4:
                break
        
        # Build inner list
        inner = []
        for i in range(n):
            inner.append([pos[i][0], pos[i][1], angles[i]])
        
        # Center the cluster
        cx_mean = np.mean([p[0] for p in inner])
        cy_mean = np.mean([p[1] for p in inner])
        centered_inner = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in inner]
        
        return centered_inner
    
    # Try force-directed from hexagonal lattice initial
    if time.time() < deadline - 30:
        initial = []
        for q, r in selected:
            x, y = axial_to_cartesian(q, r)
            initial.append([x, y, 0.0])
        
        # Center initial
        cx_mean = sum(p[0] for p in initial) / len(initial)
        cy_mean = sum(p[1] for p in initial) / len(initial)
        centered_initial = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in initial]
        
        for outer_angle in [0.0, 30.0]:
            if time.time() > deadline - 20:
                break
            optimized = force_directed_optimize(centered_initial, (0.0, 0.0), outer_angle, max_iters=1500)
            side, valid = evaluate_packing(optimized, (0.0, 0.0), outer_angle)
            if valid and side < best_side:
                best_side = side
                best_result = (optimized, (0.0, 0.0), side, outer_angle)
    
    # --- Strategy 3: Grid search with small perturbations ---
    if time.time() < deadline - 25:
        # Start from best found so far or lattice
        if best_result:
            base_inner = [list(p) for p in best_result[0]]
            base_angle = best_result[3]
        else:
            base_inner = []
            for q, r in selected:
                x, y = axial_to_cartesian(q, r)
                base_inner.append([x, y, 0.0])
            cx_mean = sum(p[0] for p in base_inner) / len(base_inner)
            cy_mean = sum(p[1] for p in base_inner) / len(base_inner)
            base_inner = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in base_inner]
            base_angle = 0.0
        
        # Try small random perturbations
        rng = np.random.default_rng(42)
        for trial in range(200):
            if time.time() > deadline - 10:
                break
            perturbed = []
            for p in base_inner:
                dx = rng.normal(0, 0.1)
                dy = rng.normal(0, 0.1)
                perturbed.append([p[0] + dx, p[1] + dy, p[2]])
            
            # Re-center
            cx_mean = sum(p[0] for p in perturbed) / len(perturbed)
            cy_mean = sum(p[1] for p in perturbed) / len(perturbed)
            centered = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in perturbed]
            
            side, valid = evaluate_packing(centered, (0.0, 0.0), base_angle)
            if valid and side < best_side:
                best_side = side
                best_result = (centered, (0.0, 0.0), side, base_angle)
    
    # Fallback
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
