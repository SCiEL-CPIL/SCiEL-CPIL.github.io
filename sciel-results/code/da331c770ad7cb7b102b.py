"""Rewritten program: hexagon packing via circle-packing approximation + refinement."""
import math
import time
import numpy as np
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 35s safety margin

    def hexagon_vertices(cx, cy, angle_deg, side=1.0):
        angle_rad = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + math.radians(60 * i)
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def point_in_hexagon(px, py, cx, cy, angle_deg, side=1.0):
        dx = px - cx
        dy = py - cy
        angle_rad = -math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a

        inradius = side * math.sqrt(3) / 2
        max_dot = 0.0
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            dot = lx * math.cos(theta) + ly * math.sin(theta)
            if dot > max_dot:
                max_dot = dot
        return max_dot <= inradius + 1e-9

    def hexagon_overlap(a_x, a_y, a_angle, b_x, b_y, b_angle, eps=1e-9):
        """Check if two unit hexagons overlap (approximate via Minkowski sum / distance)."""
        # For same orientation, minimum center distance = 2.0
        # For different orientations, it's complex. Use vertex-based check:
        # Check if any vertex of A is inside B, or any vertex of B is inside A
        verts_a = hexagon_vertices(a_x, a_y, a_angle)
        verts_b = hexagon_vertices(b_x, b_y, b_angle)
        for vx, vy in verts_a:
            if point_in_hexagon(vx, vy, b_x, b_y, b_angle):
                return True
        for vx, vy in verts_b:
            if point_in_hexagon(vx, vy, a_x, a_y, a_angle):
                return True
        # Also check if centers are very close
        d = math.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2)
        if d < 1.9 - eps:
            return True
        return False

    def all_inside_outer(inner, ocx, ocy, oside, oangle):
        """Check all inner hexagons are inside outer hexagon."""
        for x, y, a in inner:
            verts = hexagon_vertices(x, y, a)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, ocx, ocy, oangle, oside):
                    return False
        return True

    def no_overlaps(inner, eps=1e-9):
        n = len(inner)
        for i in range(n):
            for j in range(i + 1, n):
                if hexagon_overlap(inner[i][0], inner[i][1], inner[i][2],
                                   inner[j][0], inner[j][1], inner[j][2], eps):
                    return False
        return True

    def compute_required_outer(inner, oangle=0.0):
        """Compute minimum outer hexagon (centered at centroid) to contain all inner."""
        xs = [p[0] for p in inner]
        ys = [p[1] for p in inner]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(xs)
        
        # Binary search on outer side length
        lo = 1.0
        hi = 20.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if all_inside_outer(inner, cx, cy, mid, oangle):
                hi = mid
            else:
                lo = mid
        return hi, cx, cy

    # Precompute hexagonal lattice positions (same orientation, spacing 2)
    lattice_points = []
    for r in range(-6, 7):
        for c in range(-6, 7):
            x = 2.0 * c + (r % 2) * 1.0
            y = r * math.sqrt(3)
            lattice_points.append((x, y))
    
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)

    # Strategy 1: Greedy placement on hexagonal lattice
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    rng = random.Random(12345)

    # Try many random subsets of the lattice
    for attempt in range(5000):
        if time.time() > deadline - 10.0:
            break

        # Start from origin, add neighbors greedily
        placed = [(0.0, 0.0)]
        
        # Shuffle lattice points (excluding origin)
        candidates = [p for p in lattice_points if p != (0.0, 0.0)]
        rng.shuffle(candidates)

        for px, py in candidates:
            if len(placed) >= N:
                break
            valid = True
            for qx, qy in placed:
                d = math.sqrt((px - qx)**2 + (py - qy)**2)
                if d < 2.0 - 1e-6:
                    valid = False
                    break
            if valid:
                placed.append((px, py))

        if len(placed) < N:
            continue

        inner = [[x, y, 0.0] for x, y in placed]
        
        if not no_overlaps(inner):
            continue

        side, cx, cy = compute_required_outer(inner, 0.0)
        if side < best_side:
            best_side = side
            best_inner = inner
            best_center = (cx, cy)
            best_angle = 0.0

    # Strategy 2: Local perturbation / annealing on the best found arrangement
    if best_inner is not None and time.time() < deadline - 15.0:
        current_inner = [list(p) for p in best_inner]
        current_side, current_cx, current_cy = compute_required_outer(current_inner, 0.0)
        
        T = 0.5  # temperature
        for step in range(5000):
            if time.time() > deadline - 10.0:
                break
            T *= 0.995
            
            i = rng.randint(0, N - 1)
            old_x, old_y, old_a = current_inner[i]
            
            # Random perturbation
            new_x = old_x + rng.gauss(0, 0.3)
            new_y = old_y + rng.gauss(0, 0.3)
            new_a = old_a + rng.gauss(0, 5.0)
            
            current_inner[i] = [new_x, new_y, new_a]
            
            # Check validity
            if no_overlaps(current_inner):
                new_side, new_cx, new_cy = compute_required_outer(current_inner, 0.0)
                delta_e = new_side - current_side
                if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
                    current_side = new_side
                    current_cx, current_cy = new_cx, new_cy
                    if new_side < best_side:
                        best_side = new_side
                        best_inner = [list(p) for p in current_inner]
                        best_center = (new_cx, new_cy)
                        best_angle = 0.0
                else:
                    current_inner[i] = [old_x, old_y, old_a]
            else:
                current_inner[i] = [old_x, old_y, old_a]

    # Strategy 3: Try different outer orientations
    if time.time() < deadline - 15.0:
        for oangle in [0.0, 30.0, 60.0]:
            # Recompute best for this outer angle
            for attempt in range(1000):
                if time.time() > deadline - 10.0:
                    break
                
                placed = [(0.0, 0.0)]
                candidates = [p for p in lattice_points if p != (0.0, 0.0)]
                rng.shuffle(candidates)

                for px, py in candidates:
                    if len(placed) >= N:
                        break
                    valid = True
                    for qx, qy in placed:
                        d = math.sqrt((px - qx)**2 + (py - qy)**2)
                        if d < 2.0 - 1e-6:
                            valid = False
                            break
                    if valid:
                        placed.append((px, py))

                if len(placed) < N:
                    continue

                inner = [[x, y, 0.0] for x, y in placed]
                if not no_overlaps(inner):
                    continue

                side, cx, cy = compute_required_outer(inner, oangle)
                if side < best_side:
                    best_side = side
                    best_inner = inner
                    best_center = (cx, cy)
                    best_angle = oangle

    # Final validation
    if best_inner is None:
        # Fallback: linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Verify final result
    if not no_overlaps(best_inner) or not all_inside_outer(best_inner, best_center[0], best_center[1], best_side, best_angle):
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
