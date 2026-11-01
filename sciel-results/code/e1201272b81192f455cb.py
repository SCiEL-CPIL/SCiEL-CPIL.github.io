import math
import time
import random
import numpy as np
from itertools import combinations, product

N = 11

def construct():
    deadline = time.time() + 145.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Precompute hex vertices for side 1, angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    base_verts = []
    for i in range(6):
        theta = math.radians(60 * i)
        base_verts.append((math.cos(theta), math.sin(theta)))

    def transform_verts(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        verts = []
        for vx, vy in base_verts:
            rx = vx * ca - vy * sa
            ry = vx * sa + vy * ca
            verts.append((cx + S * rx, cy + S * ry))
        return verts

    def segments_intersect(p1, p2, p3, p4):
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

    def point_in_polygon(p, polygon):
        x, y = p
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def hexagons_disjoint(verts_a, verts_b):
        for v in verts_a:
            if point_in_polygon(v, verts_b):
                return False
        for v in verts_b:
            if point_in_polygon(v, verts_a):
                return False
        for i in range(6):
            a1, a2 = verts_a[i], verts_a[(i + 1) % 6]
            b1, b2 = verts_b[i], verts_b[(i + 1) % 6]
            if segments_intersect(a1, a2, b1, b2):
                return False
        return True

    def compute_outer_hex(inner, ocx, ocy, oangle):
        inner_verts = []
        for (ix, iy, iangle) in inner:
            inner_verts.append(transform_verts(ix, iy, 1.0, iangle))
        
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for iv in inner_verts:
                for v in iv:
                    proj = (v[0] - ocx) * nx + (v[1] - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    def all_hexagons_fit(inner, ocx, ocy, S, oangle):
        outer_verts = transform_verts(ocx, ocy, S, oangle)
        inner_verts = []
        for (ix, iy, iangle) in inner:
            iv = transform_verts(ix, iy, 1.0, iangle)
            inner_verts.append(iv)
        for iv in inner_verts:
            for v in iv:
                if not point_in_polygon(v, outer_verts):
                    return False
        for i in range(len(inner_verts)):
            for j in range(i + 1, len(inner_verts)):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                    return False
        return True

    def local_search(inner, ocx, ocy, oangle, seed):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        rng = random.Random(seed)
        cur_inner = [list(x) for x in inner]
        cur_S = compute_outer_hex(cur_inner, ocx, ocy, oangle)
        cur_valid = all_hexagons_fit(cur_inner, ocx, ocy, cur_S, oangle)
        
        if not cur_valid:
            for _ in range(50):
                idx = rng.randrange(N)
                old_x, old_y, old_a = cur_inner[idx]
                cur_inner[idx][0] = old_x + rng.gauss(0, 0.5)
                cur_inner[idx][1] = old_y + rng.gauss(0, 0.5)
                cur_S = compute_outer_hex(cur_inner, ocx, ocy, oangle)
                cur_valid = all_hexagons_fit(cur_inner, ocx, ocy, cur_S, oangle)
                if cur_valid:
                    break
            if not cur_valid:
                return
        
        if cur_valid and cur_S < best_S:
            best_S = cur_S
            best_inner = [list(x) for x in cur_inner]
            best_center = (ocx, ocy)
            best_angle = oangle
            best_valid = True
        
        for it in range(2000):
            if time.time() > deadline:
                break
            old = [list(x) for x in cur_inner]
            idx = rng.randrange(N)
            cur_inner[idx][0] += rng.gauss(0, 0.15)
            cur_inner[idx][1] += rng.gauss(0, 0.15)
            if rng.random() < 0.03:
                cur_inner[idx][2] = rng.choice([0, 30, 60, 90, 120, 150])
            
            new_S = compute_outer_hex(cur_inner, ocx, ocy, oangle)
            new_valid = all_hexagons_fit(cur_inner, ocx, ocy, new_S, oangle)
            
            if new_valid and new_S < cur_S:
                cur_S = new_S
                if new_S < best_S:
                    best_S = new_S
                    best_inner = [list(x) for x in cur_inner]
                    best_center = (ocx, ocy)
                    best_angle = oangle
                    best_valid = True
            elif new_valid:
                if rng.random() < 0.1:
                    cur_S = new_S
            else:
                cur_inner = old

    # NEW STRATEGY: Discrete Lattice Search with Greedy Best-Fit
    # We search for configurations on a fine triangular lattice.
    # Instead of just taking the N closest points to the origin (which creates a fixed "blob"),
    # we use a greedy algorithm to place hexagons one by one, choosing the position that
    # minimizes the increase in the bounding hexagon size. This allows the shape to adapt
    # to the specific constraints of the outer hexagon.
    
    def try_greedy_lattice(spacing, angle_deg, seed):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        
        # Generate lattice points
        max_coord = 12
        points = []
        a1 = (spacing, 0)
        a2 = (spacing / 2, spacing * math.sqrt(3) / 2)
        
        for i in range(-max_coord, max_coord + 1):
            for j in range(-max_coord, max_coord + 1):
                x = i * a1[0] + j * a2[0]
                y = i * a1[1] + j * a2[1]
                points.append((x, y))
        
        # Sort by distance from origin to prioritize central placements
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        
        # Rotate the lattice
        rad = math.radians(angle_deg)
        ca, sa = math.cos(rad), math.sin(rad)
        
        rotated_points = []
        for x, y in points:
            rx = x * ca - y * sa
            ry = x * sa + y * ca
            rotated_points.append((rx, ry))
        
        # Greedy placement
        # Start with the point closest to origin
        used_indices = set()
        placed_points = [rotated_points[0]]
        used_indices.add(0)
        
        # We need to place N-1 more points
        # To make this efficient, we don't recompute the full outer hex for every candidate.
        # Instead, we approximate the "cost" as the max distance from the centroid of placed points.
        # Or better, we just pick the next point that minimizes the max distance from the current centroid.
        
        for _ in range(N - 1):
            if time.time() > deadline:
                return
                
            # Calculate current centroid
            cx = sum(p[0] for p in placed_points) / len(placed_points)
            cy = sum(p[1] for p in placed_points) / len(placed_points)
            
            best_idx = -1
            best_dist = float('inf')
            
            # Search only in the first few hundred points to keep it fast
            # Since points are sorted by distance from origin, the best candidates are likely nearby
            search_limit = min(200, len(rotated_points))
            
            for idx in range(search_limit):
                if idx in used_indices:
                    continue
                px, py = rotated_points[idx]
                # Distance from centroid
                d = (px - cx)**2 + (py - cy)**2
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            
            if best_idx == -1:
                # Fallback: pick the next unused point
                for idx in range(len(rotated_points)):
                    if idx not in used_indices:
                        best_idx = idx
                        break
            
            if best_idx != -1:
                placed_points.append(rotated_points[best_idx])
                used_indices.add(best_idx)
        
        # Center the configuration
        cx = sum(p[0] for p in placed_points) / N
        cy = sum(p[1] for p in placed_points) / N
        
        inner = []
        for x, y in placed_points:
            inner.append([x - cx, y - cy, 0.0])
        
        # Try a few outer angles
        for oangle in [0, 30, 60]:
            S = compute_outer_hex(inner, 0.0, 0.0, oangle)
            if all_hexagons_fit(inner, 0.0, 0.0, S, oangle):
                if S < best_S:
                    best_S = S
                    best_inner = [list(x) for x in inner]
                    best_center = (0.0, 0.0)
                    best_angle = oangle
                    best_valid = True
                    # Run local search to refine
                    local_search(inner, 0.0, 0.0, oangle, seed)

    # Run lattice search
    # Spacings: sqrt(3) is the touch distance.
    base_spacing = math.sqrt(3)
    spacings = [base_spacing * (1 + i * 0.001) for i in range(5)]
    angles = [0, 15, 30, 45, 60]
    
    seed_counter = 100
    for sp in spacings:
        for ang in angles:
            if time.time() > deadline:
                break
            try_greedy_lattice(sp, ang, seed_counter)
            seed_counter += 1

    # Fallback if lattice search didn't find anything good
    if not best_valid:
        initial_configs = []
        sp = math.sqrt(3)
        hex_config = [[0.0, 0.0, 0.0]]
        for i in range(6):
            theta = math.radians(60 * i)
            hex_config.append([sp * math.cos(theta), sp * math.sin(theta), 0.0])
        for i in [0, 2, 4]:
            theta = math.radians(60 * i + 30)
            hex_config.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
        for i in [1, 3, 5]:
            theta = math.radians(60 * i)
            hex_config.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
        while len(hex_config) < N:
            hex_config.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
        initial_configs.append((hex_config[:N], (0.0, 0.0), 0.0))
        
        for i, (cfg, c, a) in enumerate(initial_configs):
            if time.time() > deadline:
                break
            local_search(cfg, c[0], c[1], a, i * 1000 + 42)

    if not best_valid or best_inner is None:
        sp = math.sqrt(3)
        fallback_inner = []
        for k in range(N):
            fallback_inner.append([(k - N // 2) * sp, 0.0, 0.0])
        S = compute_outer_hex(fallback_inner, 0.0, 0.0, 0.0)
        for _ in range(100):
            if all_hexagons_fit(fallback_inner, 0.0, 0.0, S, 0.0):
                return fallback_inner, (0.0, 0.0), S, 0.0
            for k in range(N):
                fallback_inner[k][0] *= 1.05
            S = compute_outer_hex(fallback_inner, 0.0, 0.0, 0.0)
        return fallback_inner, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle
