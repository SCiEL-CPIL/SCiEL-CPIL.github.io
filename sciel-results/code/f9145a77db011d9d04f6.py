"""Optimized hexagon packing: 11 unit hexagons in a regular hexagon.
Combines structural adjacency-graph exploration (inspiration) with
smooth L-BFGS-B optimization and multi-start (current program).
"""
import numpy as np
import time
import random
import math
from scipy.optimize import minimize

N = 11
DEADLINE = 150.0
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length 'side'."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _hex_edge_normals(verts):
    """Return unit normals for each edge of a convex polygon (vertices in CCW order)."""
    n = len(verts)
    normals = []
    for i in range(n):
        j = (i + 1) % n
        edge = verts[j] - verts[i]
        length = np.linalg.norm(edge)
        if length < 1e-12:
            normals.append(np.array([1.0, 0.0]))
        else:
            nx = -edge[1] / length
            ny = edge[0] / length
            normals.append(np.array([nx, ny]))
    return np.array(normals)

def _min_enclosing_side(inner_positions, outer_center, outer_angle):
    """Find minimum side length of outer hexagon to contain all inner hexagons."""
    cx, cy = outer_center
    best_side = 0.0
    angle_rad = np.radians(outer_angle)
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            dx = v[0] - cx
            dy = v[1] - cy
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            
            r = np.sqrt(lx*lx + ly*ly)
            if r < 1e-12:
                continue
            
            theta = np.arctan2(ly, lx)
            best_cos = 0.0
            for k in range(6):
                a = np.radians(30 + k * 60)
                d = abs(((theta - a) % (2*np.pi) + np.pi) % (2*np.pi))
                d = min(d, 2*np.pi - d)
                c = np.cos(d)
                if c > best_cos:
                    best_cos = c
            
            if best_cos < 1e-12:
                best_side = max(best_side, 1000.0)
            else:
                required_side = 2.0 * r * best_cos / np.sqrt(3)
                best_side = max(best_side, required_side)
    
    return best_side

def _hex_overlap_penalty(verts1, verts2):
    """Compute smooth overlap penalty between two convex hexagons using SAT."""
    normals1 = _hex_edge_normals(verts1)
    normals2 = _hex_edge_normals(verts2)
    all_normals = np.vstack([normals1, normals2])
    
    total_penalty = 0.0
    for nx, ny in all_normals:
        proj1 = verts1 @ np.array([nx, ny])
        proj2 = verts2 @ np.array([nx, ny])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        total_penalty += overlap * overlap
    
    return total_penalty

def _hexagons_disjoint(verts1, verts2, tol=1e-7):
    """Check if two convex hexagons are disjoint using SAT."""
    normals1 = _hex_edge_normals(verts1)
    normals2 = _hex_edge_normals(verts2)
    all_normals = np.vstack([normals1, normals2])
    
    for nx, ny in all_normals:
        proj1 = verts1 @ np.array([nx, ny])
        proj2 = verts2 @ np.array([nx, ny])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    
    return False

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if configuration is valid: all inner hexagons inside outer, no overlaps."""
    cx, cy = outer_center
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            dx = v[0] - cx
            dy = v[1] - cy
            angle_rad = np.radians(outer_angle)
            ca = np.cos(-angle_rad)
            sa = np.sin(-angle_rad)
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            
            inradius = outer_side * np.sqrt(3) / 2
            for k in range(6):
                a = np.radians(30 + k * 60)
                nx = np.cos(a)
                ny = np.sin(a)
                if lx * nx + ly * ny > inradius + tol:
                    return False
    
    all_verts = [_hex_vertices(ix, iy, 1.0, iangle) for (ix, iy, iangle) in inner_positions]
    for i in range(N):
        for j in range(i+1, N):
            if not _hexagons_disjoint(all_verts[i], all_verts[j], tol=tol):
                return False
    
    return True

def _initial_lattice():
    """Create initial hexagonal lattice arrangement with correct spacing."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 2.0 * np.sqrt(3)
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    dist2 = dist * np.sqrt(3)
    angles2 = [30, 90, 150, 210]
    for a_deg in angles2:
        a = np.radians(a_deg)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    return positions

def _initial_zigzag():
    """Create a zigzag/linear initial arrangement."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 2.0 * np.sqrt(3)
    for i in range(1, N):
        x = i * dist * 0.9
        y = (i % 2) * dist * 0.5
        positions.append([x, y, 0.0])
    return positions

def _initial_circular():
    """Create a circular initial arrangement."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        a = 2 * np.pi * i / (N - 1)
        r = 3.0
        positions.append([r * np.cos(a), r * np.sin(a), 0.0])
    return positions

def _initial_random():
    """Create a random initial configuration."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        r = random.uniform(1.0, 4.0)
        a = random.uniform(0, 2*np.pi)
        angle = random.uniform(0, 360)
        positions.append([r * np.cos(a), r * np.sin(a), angle])
    return positions

def _scipy_optimize(positions, outer_angle, time_budget):
    """Use scipy L-BFGS-B to optimize positions with penalty formulation."""
    init_side = _min_enclosing_side(positions, (0.0, 0.0), outer_angle)
    x0_pos = np.array([p for pos in positions for p in pos])
    x0 = np.concatenate([x0_pos, [init_side]])
    
    angle_rad = np.radians(outer_angle)
    ca = np.cos(angle_rad)
    sa = np.sin(angle_rad)
    outer_normals = []
    for k in range(6):
        a_local = np.radians(30 + 60 * k)
        a_world = a_local + angle_rad
        outer_normals.append(np.array([np.cos(a_world), np.sin(a_world)]))
    
    def objective_with_containment_penalty(x):
        pos = x[:3*N].reshape(N, 3)
        S = x[3*N]
        
        overlap_penalty = 0.0
        verts_list = [_hex_vertices(p[0], p[1], 1.0, p[2]) for p in pos]
        for i in range(N):
            for j in range(i+1, N):
                overlap_penalty += _hex_overlap_penalty(verts_list[i], verts_list[j])
        
        cont_penalty = 0.0
        limit = S * np.sqrt(3) / 2.0
        for i in range(N):
            verts = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, pos[i, 2])
            for n in outer_normals:
                for v in verts:
                    violation = np.dot(v, n) - limit
                    if violation > 0:
                        cont_penalty += violation * violation
        
        return S + 10.0 * overlap_penalty + 100.0 * cont_penalty

    result = minimize(
        objective_with_containment_penalty,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-12}
    )
    
    best_x = result.x
    best_pos = best_x[:3*N].reshape(N, 3)
    best_S = best_x[3*N]
    
    true_S = _min_enclosing_side([tuple(p) for p in best_pos], (0.0, 0.0), outer_angle)
    
    best_pos[:, 2] = best_pos[:, 2] % 360
    
    return [tuple(p) for p in best_pos], true_S

def _try_arrangement(positions, outer_angle):
    """Try to optimize a given arrangement and return the best result."""
    if _remaining_time() < 5.0:
        return None, float('inf')
    
    try:
        opt_positions, opt_value = _scipy_optimize(positions, outer_angle, 10.0)
    except Exception:
        opt_positions = positions
        opt_value = _min_enclosing_side(positions, (0.0, 0.0), outer_angle)
    
    side = _min_enclosing_side(opt_positions, (0.0, 0.0), outer_angle)
    
    if _check_validity(opt_positions, (0.0, 0.0), side, outer_angle, tol=1e-5):
        return opt_positions, side
    
    return None, float('inf')

def _generate_adjacency_structures():
    """Generate diverse adjacency-based initial configurations.
    Uses hexagonal grid structures with different neighbor patterns."""
    structures = []
    
    # Structure 1: Compact hex cluster (center + 6 ring + 4 outer)
    pos = []
    pos.append([0.0, 0.0, 0.0])
    d1 = 2.0 * np.sqrt(3)
    for k in range(6):
        a = np.radians(60 * k)
        pos.append([d1 * np.cos(a), d1 * np.sin(a), 0.0])
    d2 = d1 * np.sqrt(3)
    for a_deg in [30, 90, 150, 210]:
        a = np.radians(a_deg)
        pos.append([d2 * np.cos(a), d2 * np.sin(a), 0.0])
    structures.append(pos)
    
    # Structure 2: Linear chain with alternating offsets
    pos = []
    pos.append([0.0, 0.0, 0.0])
    d = 2.0 * np.sqrt(3)
    for i in range(1, N):
        x = i * d * 0.85
        y = (i % 2) * d * 0.6
        pos.append([x, y, 0.0])
    structures.append(pos)
    
    # Structure 3: Two rings (center + 5 inner + 5 outer)
    pos = []
    pos.append([0.0, 0.0, 0.0])
    d1 = 2.0 * np.sqrt(3)
    for k in range(5):
        a = np.radians(72 * k + 18)
        pos.append([d1 * np.cos(a), d1 * np.sin(a), 0.0])
    d2 = d1 * 1.7
    for k in range(5):
        a = np.radians(72 * k + 54)
        pos.append([d2 * np.cos(a), d2 * np.sin(a), 0.0])
    structures.append(pos)
    
    # Structure 4: Asymmetric cluster
    pos = []
    pos.append([0.0, 0.0, 0.0])
    d1 = 2.0 * np.sqrt(3)
    # 4 neighbors in one direction, 2 in another
    for k in range(4):
        a = np.radians(60 * k)
        pos.append([d1 * np.cos(a), d1 * np.sin(a), 0.0])
    for k in range(2):
        a = np.radians(180 + 60 * k)
        pos.append([d1 * np.cos(a), d1 * np.sin(a), 0.0])
    d2 = d1 * 1.5
    for k in range(4):
        a = np.radians(60 * k + 30)
        pos.append([d2 * np.cos(a), d2 * np.sin(a), 0.0])
    structures.append(pos)
    
    # Structure 5: Stacked rows
    pos = []
    d = 2.0 * np.sqrt(3)
    # Row 1: 4 hexagons
    for i in range(4):
        pos.append([i * d, 0.0, 0.0])
    # Row 2: 4 hexagons offset
    for i in range(4):
        pos.append([i * d + d/2, d * np.sqrt(3)/2, 0.0])
    # Row 3: 3 hexagons
    for i in range(3):
        pos.append([i * d + d, d * np.sqrt(3), 0.0])
    structures.append(pos)
    
    # Structure 6: Random perturbations of lattice
    base = _initial_lattice()
    for _ in range(3):
        perturbed = []
        for p in base:
            new_p = [p[0] + random.uniform(-0.3, 0.3),
                     p[1] + random.uniform(-0.3, 0.3),
                     p[2] + random.uniform(-15, 15)]
            perturbed.append(new_p)
        structures.append(perturbed)
    
    return structures

def construct():
    """Main entry point."""
    random.seed(42)
    np.random.seed(42)
    
    best_result = None
    best_score = float('inf')
    
    outer_angles = [0.0, 30.0, 15.0, 45.0]
    
    # Generate diverse initial configurations
    initial_configs = _generate_adjacency_structures()
    initial_configs.append(_initial_circular())
    initial_configs.append(_initial_random())
    initial_configs.append(_initial_random())
    
    for oa in outer_angles:
        if _remaining_time() < 15.0:
            break
        
        for init_pos in initial_configs:
            if _remaining_time() < 8.0:
                break
            
            positions = [list(p) for p in init_pos]
            
            result, side = _try_arrangement(positions, oa)
            
            if result is not None and side < best_score:
                best_score = side
                best_result = (result, (0.0, 0.0), side, oa)
    
    if best_result is not None:
        return best_result
    
    # Fallback: known valid configuration
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 3.5
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    dist2 = 6.0
    for k in range(4):
        a = np.radians(30 + 60 * k)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    
    side = _min_enclosing_side(positions, (0.0, 0.0), 0.0)
    return positions, (0.0, 0.0), side, 0.0
