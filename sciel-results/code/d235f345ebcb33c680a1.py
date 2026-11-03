"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Combines lattice-based structured packing with full SLSQP continuous optimization.
"""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize, Bounds

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


def compute_outer_side(centers, angles, ocx, ocy, otheta_deg):
    """Compute minimum outer hexagon side for given configuration."""
    otheta = math.radians(otheta_deg)
    max_proj = 0.0
    for k in range(6):
        normal_angle = otheta + math.pi/6 + k * math.pi/3
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        for i in range(len(centers)):
            cx, cy = centers[i]
            angle = math.radians(angles[i])
            for vi in range(6):
                vert_angle = angle + vi * math.pi / 3
                vx = cx + math.cos(vert_angle)
                vy = cy + math.sin(vert_angle)
                proj = (vx - ocx) * nx + (vy - ocy) * ny
                if proj > max_proj:
                    max_proj = proj
    return 2.0 * max_proj / math.sqrt(3)


def check_disjoint(centers, angles, min_dist=math.sqrt(3) - 0.0001):
    """Check if all hexagons are disjoint (center distance >= sqrt(3) for same orientation)."""
    n = len(centers)
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                return False
    return True


def polygons_overlap_sat(A, B):
    """SAT: True if two convex polygons intersect (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                return False
    return True


def check_validity_full(centers, angles, ocx, ocy, outer_side, outer_angle):
    """Check all inner hexagons are inside outer and pairwise non-overlapping."""
    outer = hex_vertices(ocx, ocy, outer_side, outer_angle)
    polys = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        ang = angles[i]
        p = hex_vertices(cx, cy, 1.0, ang)
        polys.append(p)
    for p in polys:
        for (px, py) in p:
            inside = True
            n = len(outer)
            sign = 0
            for i in range(n):
                x1, y1 = outer[i]
                x2, y2 = outer[(i + 1) % n]
                cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                if abs(cross) < 1e-9:
                    continue
                s = 1 if cross > 0 else -1
                if sign == 0:
                    sign = s
                elif s != sign:
                    inside = False
                    break
            if not inside:
                return False
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True


def build_lattice_packing(scale, rot_in_deg, rot_out_deg, pattern='ring12'):
    """Build a hexagonal lattice packing of 12 unit hexagons."""
    a = 2.0 * scale
    positions = []
    
    if pattern == 'ring12':
        positions.append((0.0, 0.0))
        for k in range(6):
            theta = math.radians(k * 60.0)
            x = a * math.cos(theta)
            y = a * math.sin(theta)
            positions.append((x, y))
        for k in range(5):
            theta = math.radians(k * 60.0)
            x = 2.0 * a * math.cos(theta)
            y = 2.0 * a * math.sin(theta)
            positions.append((x, y))
    elif pattern == 'hex6+6':
        for k in range(6):
            theta = math.radians(k * 60.0)
            x = a * math.cos(theta)
            y = a * math.sin(theta)
            positions.append((x, y))
        for k in range(6):
            theta = math.radians(k * 60.0 + 30.0)
            x = 2.0 * a * math.cos(theta)
            y = 2.0 * a * math.sin(theta)
            positions.append((x, y))
    elif pattern == 'rect3x4':
        row_spacing = a * math.sqrt(3) / 2.0
        for row in range(3):
            for col in range(4):
                x = col * a
                y = row * row_spacing
                if row % 2 == 1:
                    x += a / 2.0
                positions.append((x - 1.5 * a, y - row_spacing))
    else:
        positions.append((0.0, 0.0))
        for k in range(6):
            theta = math.radians(k * 60.0)
            x = a * math.cos(theta)
            y = a * math.sin(theta)
            positions.append((x, y))
        for k in range(5):
            theta = math.radians(k * 60.0)
            x = 2.0 * a * math.cos(theta)
            y = 2.0 * a * math.sin(theta)
            positions.append((x, y))
    
    centers = [list(p) for p in positions]
    angles = [rot_in_deg] * 12
    outer_center = (0.0, 0.0)
    outer_side = compute_outer_side(centers, angles, 0.0, 0.0, rot_out_deg)
    
    return centers, angles, outer_center, outer_side, rot_out_deg


def optimize_with_slsqp(centers, angles, max_iters=80):
    """Use SLSQP to optimize positions, angles, and outer hexagon parameters."""
    n = len(centers)
    
    x0 = []
    for i in range(n):
        x0.extend([centers[i][0], centers[i][1], angles[i]])
    x0.extend([0.0, 0.0, 0.0])  # outer center and angle
    
    def objective(x):
        if time.time() > DEADLINE - 30:
            return 1e10
        c = [[x[3*i], x[3*i+1]] for i in range(n)]
        a = [x[3*i+2] for i in range(n)]
        ocx, ocy, otheta = x[3*n], x[3*n+1], x[3*n+2]
        side = compute_outer_side(c, a, ocx, ocy, otheta)
        return side
    
    constraints = []
    min_dist_sq = (math.sqrt(3) - 0.0001)**2
    for i in range(n):
        for j in range(i+1, n):
            def dist_constraint(x, i=i, j=j):
                xi = x[3*i]
                yi = x[3*i+1]
                xj = x[3*j]
                yj = x[3*j+1]
                return (xi - xj)**2 + (yi - yj)**2 - min_dist_sq
            constraints.append({'type': 'ineq', 'fun': dist_constraint})
    
    bounds = []
    for i in range(n):
        bounds.extend([(-6, 6), (-6, 6), (0, 60)])
    bounds.extend([(-3, 3), (-3, 3), (0, 60)])
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options={'maxiter': max_iters, 'ftol': 1e-10})
        if result.success or result.fun < 1e9:
            c = [[result.x[3*i], result.x[3*i+1]] for i in range(n)]
            a = [result.x[3*i+2] for i in range(n)]
            ocx, ocy, otheta = result.x[3*n], result.x[3*n+1], result.x[3*n+2]
            side = compute_outer_side(c, a, ocx, ocy, otheta)
            return c, a, ocx, ocy, otheta, side
    except Exception:
        pass
    
    return None


def polish_outer(centers, angles, initial_ocx, initial_ocy, initial_otheta):
    """High-precision polish of outer hexagon parameters."""
    best_side = compute_outer_side(centers, angles, initial_ocx, initial_ocy, initial_otheta)
    best_ocx, best_ocy, best_otheta = initial_ocx, initial_ocy, initial_otheta
    
    scales = [1.0, 0.1, 0.01, 0.001]
    for scale in scales:
        improved = True
        while improved and time.time() < DEADLINE - 15:
            improved = False
            for dx in [-scale, 0, scale]:
                for dy in [-scale, 0, scale]:
                    for da_deg in [-scale*10, 0, scale*10]:
                        otheta = (best_otheta + da_deg) % 60
                        side = compute_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, otheta)
                        if side < best_side - 1e-12:
                            best_side = side
                            best_ocx += dx
                            best_ocy += dy
                            best_otheta = otheta
                            improved = True
    
    return best_ocx, best_ocy, best_otheta, best_side


def generate_ring_configurations():
    """Generate diverse ring-based initial configurations."""
    eps = 0.001
    d = math.sqrt(3) + eps
    
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*math.sqrt(3)/2.0])
    
    basis = [u, v, v-u, -u, -v, u-v]
    
    ring1_vectors = [u, v, v-u, -u, -v, u-v]
    
    ring2_candidates = []
    for i in range(6):
        p1 = ring1_vectors[i]
        for b in basis:
            p2 = p1 + b
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]), atol=0.01)
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j, atol=0.01):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                duplicate = False
                for pc in ring2_candidates:
                    if np.allclose(p2, pc, atol=0.01):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_candidates.append(p2)
    
    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    
    configs = []
    
    # Config 0: closest 5 ring2 vectors
    ring2_sel = ring2_candidates[:5]
    centers = [[0.0, 0.0]]
    for vec in ring1_vectors:
        centers.append([vec[0], vec[1]])
    for vec in ring2_sel:
        centers.append([vec[0], vec[1]])
    configs.append(centers)
    
    # Additional random selections
    for config_idx in range(1, 5):
        if len(ring2_candidates) >= 5:
            idxs = list(range(len(ring2_candidates)))
            random.seed(config_idx * 12345 + 7)
            random.shuffle(idxs)
            ring2_sel = [ring2_candidates[i] for i in idxs[:5]]
            centers = [[0.0, 0.0]]
            for vec in ring1_vectors:
                centers.append([vec[0], vec[1]])
            for vec in ring2_sel:
                centers.append([vec[0], vec[1]])
            configs.append(centers)
    
    return configs


def construct():
    t0 = time.time()
    
    best_result = None
    best_side = float('inf')
    
    # Generate diverse initial configurations
    lattice_patterns = ['ring12', 'hex6+6', 'rect3x4']
    
    # Try lattice-based configurations first
    for pattern in lattice_patterns:
        if time.time() > DEADLINE - 40:
            break
        
        for scale in [0.9, 1.0, 1.1]:
            if time.time() > DEADLINE - 40:
                break
            
            for rot_in in [0.0, 30.0]:
                if time.time() > DEADLINE - 40:
                    break
                
                centers, angles, outer_center, outer_side, rot_out = build_lattice_packing(
                    scale, rot_in, 0.0, pattern)
                
                if not check_disjoint(centers, angles):
                    continue
                
                # SLSQP optimization
                slsqp_result = optimize_with_slsqp(centers, angles, max_iters=60)
                
                if slsqp_result is not None:
                    c, a, ocx, ocy, otheta, side = slsqp_result
                    
                    # Polish outer hexagon
                    ocx, ocy, otheta, side = polish_outer(c, a, ocx, ocy, otheta)
                    
                    # Verify validity
                    if check_validity_full(c, a, ocx, ocy, side, otheta):
                        if side < best_side:
                            best_side = side
                            best_result = (c, a, ocx, ocy, side, otheta)
    
    # Try ring-based configurations
    ring_configs = generate_ring_configurations()
    for centers in ring_configs:
        if time.time() > DEADLINE - 30:
            break
        
        angles = [30.0] * 12
        
        if not check_disjoint(centers, angles):
            continue
        
        slsqp_result = optimize_with_slsqp(centers, angles, max_iters=60)
        
        if slsqp_result is not None:
            c, a, ocx, ocy, otheta, side = slsqp_result
            
            ocx, ocy, otheta, side = polish_outer(c, a, ocx, ocy, otheta)
            
            if check_validity_full(c, a, ocx, ocy, side, otheta):
                if side < best_side:
                    best_side = side
                    best_result = (c, a, ocx, ocy, side, otheta)
    
    # If we found a good result, return it
    if best_result is not None:
        c, a, ocx, ocy, side, otheta = best_result
        inner = [(c[i][0], c[i][1], a[i]) for i in range(N)]
        outer_center = (ocx, ocy)
        return inner, outer_center, side, otheta
    
    # Fallback: simple hexagonal cluster
    a = 2.0
    positions = [(0.0, 0.0)]
    for k in range(6):
        theta = math.radians(k * 60.0)
        positions.append((a * math.cos(theta), a * math.sin(theta)))
    for k in range(5):
        theta = math.radians(k * 60.0)
        positions.append((2.0 * a * math.cos(theta), 2.0 * a * math.sin(theta)))
    
    centers = [list(p) for p in positions]
    angles = [0.0] * 12
    outer_center = (0.0, 0.0)
    outer_side = compute_outer_side(centers, angles, 0.0, 0.0, 0.0)
    
    inner = [(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
    return inner, outer_center, outer_side, 0.0
