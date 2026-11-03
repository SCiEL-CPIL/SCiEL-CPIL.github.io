import math
import time
import numpy as np
import random

N = 11

def _hex_vertices(cx, cy, angle, size=1.0):
    verts = []
    for k in range(6):
        a = angle + k * math.pi / 3
        verts.append((cx + size * math.cos(a), cy + size * math.sin(a)))
    return np.array(verts)

def _min_outer_side(points, outer_angle):
    k_angles = np.arange(6) * math.pi / 3 + outer_angle
    normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
    dots = points @ normals.T
    abs_dots = np.abs(dots)
    max_per_normal = np.max(abs_dots, axis=0)
    max_proj = np.max(max_per_normal)
    return (2.0 / math.sqrt(3)) * max_proj

def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def _poly_overlap(v1, v2, eps=1e-9):
    polys = [v1, v2]
    for v in polys:
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i + 1) % len(v)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            if proj1.max() < proj2.min() + eps or proj2.max() < proj1.min() + eps:
                return False
    return True

def _in_outer(point, outer_center, outer_side, outer_angle):
    p = point - outer_center
    k_angles = np.arange(6) * math.pi / 3 + outer_angle
    normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
    dots = p @ normals.T
    apothem = outer_side * math.sqrt(3) / 2.0
    return np.all(np.abs(dots) <= apothem + 1e-9)

def _valid(inner_centers, inner_angles, outer_center, outer_side, outer_angle, eps=1e-9):
    verts_list = [_hex_vertices(inner_centers[i][0], inner_centers[i][1], inner_angles[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if _poly_overlap(verts_list[i], verts_list[j], eps):
                return False
    for i in range(N):
        for v in verts_list[i]:
            if not _in_outer(v, outer_center, outer_side, outer_angle):
                return False
    return True

def _make_hexagonal_seed(variant=0):
    """Create hexagonal arrangement of 11 hexagons."""
    d = math.sqrt(3)  # distance between centers of adjacent hexagons
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        centers.append((d * math.cos(angle), d * math.sin(angle)))
    
    # Ring 2: 12 positions, pick 4
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2.append((x, y))
    
    random.seed(variant * 100 + 42)
    random.shuffle(ring2)
    centers.extend(ring2[:4])
    
    angles = [0.0] * N
    return centers, angles

def _make_compact_seed(variant=0):
    """Tighter arrangement using axial coordinates."""
    d = math.sqrt(3)
    # Use a 3-4-3 or similar compact pattern
    positions = [
        (0, 0),
        (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2),
    ]
    centers = []
    for q, r in positions:
        x = d * (q + r / 2.0)
        y = d * (math.sqrt(3) / 2.0) * r
        centers.append((x, y))
    angles = [0.0] * N
    return centers, angles

def _simulate_annealing(centers, angles, outer_angle, deadline, safety_margin=10.0):
    """Simulated annealing to minimize outer hexagon side length."""
    start_time = time.time()
    
    best_S = float('inf')
    best_centers = [c[:] for c in centers]
    best_angles = angles[:]
    
    current_centers = [c[:] for c in centers]
    current_angles = angles[:]
    
    # Initial S
    all_points = []
    for i in range(N):
        verts = _hex_vertices(current_centers[i][0], current_centers[i][1], current_angles[i])
        all_points.extend(verts.tolist())
    points_arr = np.array(all_points)
    current_S = _min_outer_side(points_arr, outer_angle)
    best_S = current_S
    best_centers = [c[:] for c in current_centers]
    best_angles = current_angles[:]
    
    T = 0.5
    cooling = 0.999
    step_size = 0.3
    iterations = 0
    max_iter = 50000
    
    while time.time() < deadline - safety_margin and iterations < max_iter:
        iterations += 1
        
        # Pick a random hexagon to move
        i = random.randint(0, N - 1)
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 0.1)
        
        old_cx, old_cy = current_centers[i]
        old_a = current_angles[i]
        current_centers[i][0] += dx
        current_centers[i][1] += dy
        current_angles[i] += da
        
        # Compute new S
        all_points = []
        for j in range(N):
            verts = _hex_vertices(current_centers[j][0], current_centers[j][1], current_angles[j])
            all_points.extend(verts.tolist())
        points_arr = np.array(all_points)
        new_S = _min_outer_side(points_arr, outer_angle)
        
        # Check for overlaps (approximate: same orientation means dist >= 2)
        overlap_penalty = 0.0
        for j in range(N):
            if j == i:
                continue
            # Quick distance check
            dx2 = current_centers[i][0] - current_centers[j][0]
            dy2 = current_centers[i][1] - current_centers[j][1]
            dist = math.sqrt(dx2*dx2 + dy2*dy2)
            if dist < 1.9:  # potential overlap
                v1 = _hex_vertices(current_centers[i][0], current_centers[i][1], current_angles[i])
                v2 = _hex_vertices(current_centers[j][0], current_centers[j][1], current_angles[j])
                if _poly_overlap(v1, v2):
                    overlap_penalty += (2.0 - dist) ** 2 * 100.0
                    break
        
        new_cost = new_S + overlap_penalty
        
        if new_cost < current_S or random.random() < math.exp(-(new_cost - current_S) / T):
            current_S = new_cost
            if new_S < best_S:
                best_S = new_S
                best_centers = [c[:] for c in current_centers]
                best_angles = current_angles[:]
        else:
            # Revert
            current_centers[i][0] = old_cx
            current_centers[i][1] = old_cy
            current_angles[i] = old_a
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    # Final verification
    if _valid(best_centers, best_angles, (0, 0), best_S, outer_angle):
        return best_centers, best_angles, best_S
    
    # If invalid, try to fix by increasing S slightly
    all_points = []
    for i in range(N):
        verts = _hex_vertices(best_centers[i][0], best_centers[i][1], best_angles[i])
        all_points.extend(verts.tolist())
    points_arr = np.array(all_points)
    S_fixed = _min_outer_side(points_arr, outer_angle)
    
    if _valid(best_centers, best_angles, (0, 0), S_fixed, outer_angle):
        return best_centers, best_angles, S_fixed
    
    return None

def construct():
    start_time = time.time()
    deadline = 145.0
    safety_margin = 5.0
    
    best_result = None
    best_S = float('inf')
    
    outer_angles = [0.0, math.pi/6, math.pi/12, -math.pi/6, math.pi/3]
    
    # Try different seeds and outer angles
    for oa in outer_angles:
        if time.time() > deadline - safety_margin:
            break
        
        for variant in range(3):
            if time.time() > deadline - safety_margin:
                break
            
            if variant == 0:
                centers, angles = _make_hexagonal_seed(variant)
            elif variant == 1:
                centers, angles = _make_compact_seed(variant)
            else:
                centers, angles = _make_hexagonal_seed(variant + 10)
            
            result = _simulate_annealing(centers, angles, oa, deadline, safety_margin)
            if result is not None:
                c, a, S = result
                if S < best_S:
                    best_S = S
                    best_result = (c, a, oa)
    
    if best_result is not None:
        c, a, oa = best_result
        inner = [[c[i][0], c[i][1], math.degrees(a[i])] for i in range(N)]
        return inner, (0.0, 0.0), best_S, math.degrees(oa)
    
    # Fallback
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
