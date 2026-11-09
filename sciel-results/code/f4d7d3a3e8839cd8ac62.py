"""Circle-packing heuristic for hexagon packing: use circumradius 1 circles,
enforce min center distance 1.75 for same-orientation, binary-search outer side."""
import numpy as np
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s limit
    
    sqrt3 = math.sqrt(3)
    MIN_DIST = 1.75  # safe min center distance for same-orientation unit hexagons
    CIRCUMRADIUS = 1.0  # circumradius of unit hexagon
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point is inside hexagon."""
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
    
    def min_outer_side(inner_list, outer_center, outer_angle):
        """Binary search for minimum outer side containing all inner hexagons."""
        cx, cy = outer_center
        lo = 1.0
        hi = 25.0
        
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
    
    def check_disjoint(inner_list, threshold=1e-6):
        """Check all pairs are disjoint using min distance for same orientation."""
        n = len(inner_list)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, a1 = inner_list[i]
                x2, y2, a2 = inner_list[j]
                dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                if abs(a1 - a2) < 1e-6:
                    if dist < MIN_DIST - threshold:
                        return False
                else:
                    # For different orientations, use conservative check
                    if dist < 2.0 - threshold:
                        # Do detailed check
                        v1 = hexagon_vertices(x1, y1, 1.0, a1)
                        v2 = hexagon_vertices(x2, y2, 1.0, a2)
                        for vx, vy in v1:
                            if point_in_hexagon(vx, vy, x2, y2, 1.0, a2):
                                return False
                        for vx, vy in v2:
                            if point_in_hexagon(vx, vy, x1, y1, 1.0, a1):
                                return False
        return True
    
    def evaluate(inner_list, outer_center, outer_angle):
        """Evaluate packing: return (side, valid)."""
        if not check_disjoint(inner_list):
            return None, False
        side = min_outer_side(inner_list, outer_center, outer_angle)
        if side is None:
            return None, False
        return side, True
    
    # --- Generate initial hexagonal lattice positions ---
    def generate_hex_ring(n):
        """Generate n positions in hexagonal ring pattern."""
        positions = [(0, 0)]
        ring = 1
        while len(positions) < n:
            q, r = ring, 0
            for side_idx in range(6):
                for step in range(ring):
                    positions.append((q, r))
                    dq = [1, 0, -1, -1, 0, 1][side_idx]
                    dr = [0, -1, -1, 0, 1, 1][side_idx]
                    q += dq
                    r += dr
            ring += 1
        return positions[:n]
    
    def axial_to_cartesian(q, r, spacing):
        """Convert axial to cartesian with given spacing."""
        x = spacing * (2.0 * q + r)
        y = spacing * sqrt3 * r
        return x, y
    
    axial_pos = generate_hex_ring(N)
    
    best_side = float('inf')
    best_result = None
    
    # --- Strategy 1: Try different spacings and outer angles ---
    for spacing in [1.0, 0.95, 0.9, 0.85, 0.8]:
        if time.time() > deadline - 30:
            break
        for outer_angle in [0.0, 30.0, 15.0]:
            if time.time() > deadline - 25:
                break
            
            inner = []
            for q, r in axial_pos:
                x, y = axial_to_cartesian(q, r, spacing)
                inner.append([x, y, 0.0])
            
            # Center cluster
            cx_mean = sum(p[0] for p in inner) / len(inner)
            cy_mean = sum(p[1] for p in inner) / len(inner)
            centered = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in inner]
            
            side, valid = evaluate(centered, (0.0, 0.0), outer_angle)
            if valid and side < best_side:
                best_side = side
                best_result = (centered, (0.0, 0.0), side, outer_angle)
    
    # --- Strategy 2: Force-directed optimization ---
    def force_optimize(init_pos, max_iters=1500):
        """Force-directed: repel if too close, attract to center."""
        n = len(init_pos)
        pos = np.array([[p[0], p[1]] for p in init_pos], dtype=float)
        
        lr = 0.03
        decay = 0.999
        
        for it in range(max_iters):
            if time.time() > deadline - 40:
                break
            
            forces = np.zeros_like(pos)
            
            # Repulsion
            for i in range(n):
                for j in range(i+1, n):
                    diff = pos[i] - pos[j]
                    dist = np.linalg.norm(diff)
                    if dist < MIN_DIST - 0.2:
                        if dist < 1e-6:
                            diff = np.random.randn(2) * 0.01
                            dist = np.linalg.norm(diff)
                        force = (MIN_DIST - dist) * 2.0
                        forces[i] += force * diff / dist
                        forces[j] -= force * diff / dist
            
            # Attraction to center
            for i in range(n):
                to_center = -pos[i]
                dist_c = np.linalg.norm(to_center)
                if dist_c > 1e-6:
                    forces[i] += 0.03 * to_center / dist_c
            
            pos += lr * forces
            lr *= decay
            
            if np.max(np.abs(forces)) < 1e-4:
                break
        
        result = [[pos[i][0], pos[i][1], 0.0] for i in range(n)]
        cx_mean = np.mean([p[0] for p in result])
        cy_mean = np.mean([p[1] for p in result])
        centered = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in result]
        return centered
    
    if time.time() < deadline - 50:
        if best_result:
            base = [list(p) for p in best_result[0]]
        else:
            base = []
            for q, r in axial_pos:
                x, y = axial_to_cartesian(q, r, 1.0)
                base.append([x, y, 0.0])
            cx_mean = sum(p[0] for p in base) / len(base)
            cy_mean = sum(p[1] for p in base) / len(base)
            base = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in base]
        
        optimized = force_optimize(base, max_iters=1200)
        for outer_angle in [0.0, 30.0]:
            side, valid = evaluate(optimized, (0.0, 0.0), outer_angle)
            if valid and side < best_side:
                best_side = side
                best_result = (optimized, (0.0, 0.0), side, outer_angle)
    
    # --- Strategy 3: Random perturbations ---
    if time.time() < deadline - 35:
        if best_result:
            base = [list(p) for p in best_result[0]]
            base_angle = best_result[3]
        else:
            base = []
            for q, r in axial_pos:
                x, y = axial_to_cartesian(q, r, 1.0)
                base.append([x, y, 0.0])
            cx_mean = sum(p[0] for p in base) / len(base)
            cy_mean = sum(p[1] for p in base) / len(base)
            base = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in base]
            base_angle = 0.0
        
        rng = np.random.default_rng(42)
        for trial in range(500):
            if time.time() > deadline - 20:
                break
            
            perturbed = []
            for p in base:
                dx = rng.normal(0, 0.06)
                dy = rng.normal(0, 0.06)
                perturbed.append([p[0] + dx, p[1] + dy, p[2]])
            
            cx_mean = sum(p[0] for p in perturbed) / len(perturbed)
            cy_mean = sum(p[1] for p in perturbed) / len(perturbed)
            centered = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in perturbed]
            
            side, valid = evaluate(centered, (0.0, 0.0), base_angle)
            if valid and side < best_side:
                best_side = side
                best_result = (centered, (0.0, 0.0), side, base_angle)
    
    # Fallback
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
