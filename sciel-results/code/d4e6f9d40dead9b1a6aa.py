import math
import time
import random
import numpy as np
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

def _point_in_hex(px, py, cx, cy, side, angle_deg, tol=1e-9):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    
    inradius = side * np.sqrt(3) / 2
    for k in range(6):
        a = np.radians(30 + k * 60)
        nx = np.cos(a)
        ny = np.sin(a)
        if lx * nx + ly * ny > inradius + tol:
            return False
    return True

def _hexagons_disjoint(verts1, verts2, tol=1e-7):
    """Check if two convex hexagons are disjoint using SAT. Returns (disjoint, min_sep)."""
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            j = (i + 1) % n
            edge = verts[j] - verts[i]
            nx = -edge[1]
            ny = edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append([nx/norm, ny/norm])
        return normals
    
    normals1 = get_normals(verts1)
    normals2 = get_normals(verts2)
    all_normals = normals1 + normals2
    
    min_sep = float('inf')
    for nx, ny in all_normals:
        proj1 = np.array([v[0]*nx + v[1]*ny for v in verts1])
        proj2 = np.array([v[0]*nx + v[1]*ny for v in verts2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2:
            sep = min2 - max1
        elif max2 < min1:
            sep = min1 - max2
        else:
            return False, 0.0
        
        if sep < min_sep:
            min_sep = sep
            
    return True, min_sep

def _min_enclosing_side(inner_positions, outer_center, outer_angle):
    """Find minimum side length of outer hexagon to contain all inner hexagons."""
    cx, cy = outer_center
    best_side = 0.0
    
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
                required_side = 2.0 * r / (np.sqrt(3) * best_cos)
                best_side = max(best_side, required_side)
    
    return best_side

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if configuration is valid: all inner hexagons inside outer, no overlaps."""
    cx, cy = outer_center
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            if not _point_in_hex(v[0], v[1], cx, cy, outer_side, outer_angle, tol=tol):
                return False
    
    all_verts = [_hex_vertices(ix, iy, 1.0, iangle) for (ix, iy, iangle) in inner_positions]
    for i in range(N):
        for j in range(i+1, N):
            disjoint, sep = _hexagons_disjoint(all_verts[i], all_verts[j], tol=tol)
            if not disjoint:
                return False
    
    return True

def _compute_energy_smooth(positions, outer_side_target=None):
    """
    Compute smooth energy for gradient-based optimization.
    Uses soft penalties that provide meaningful gradients.
    """
    if outer_side_target is None:
        outer_side_target = _min_enclosing_side(positions, (0.0, 0.0), 0.0)
    
    # Smooth overlap penalty using center distances with hexagon-aware metric
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            # Minimum distance for non-overlapping unit hexagons (center to center)
            # For two unit hexagons, minimum separation is 2*sqrt(3) when aligned
            min_dist = 2.0 * np.sqrt(3) * 0.95  # Slightly less to allow tight packing
            if dist < min_dist:
                # Smooth quadratic penalty
                overlap_penalty += (min_dist - dist) ** 2
    
    # Smooth containment penalty
    contain_penalty = 0.0
    for (ix, iy, iangle) in positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            if not _point_in_hex(v[0], v[1], 0.0, 0.0, outer_side_target, 0.0, tol=1e-6):
                dx = v[0]
                dy = v[1]
                r = np.sqrt(dx*dx + dy*dy)
                theta = np.arctan2(dy, dx)
                best_cos = 0.0
                for k in range(6):
                    a = np.radians(30 + k * 60)
                    d = abs(((theta - a) % (2*np.pi) + np.pi) % (2*np.pi))
                    d = min(d, 2*np.pi - d)
                    c = np.cos(d)
                    if c > best_cos:
                        best_cos = c
                if best_cos < 1e-9:
                    required_r = outer_side_target * np.sqrt(3) / 2.0
                else:
                    required_r = outer_side_target * np.sqrt(3) * best_cos / 2.0
                if r > required_r:
                    contain_penalty += (r - required_r) ** 2
    
    # COM penalty
    com_x = np.mean([p[0] for p in positions])
    com_y = np.mean([p[1] for p in positions])
    com_penalty = 0.01 * (com_x**2 + com_y**2)
    
    total_energy = outer_side_target + 50.0 * overlap_penalty + 50.0 * contain_penalty + com_penalty
    return total_energy

def _compute_energy(positions):
    """
    Compute the energy (objective) for a configuration.
    Energy = outer_side + overlap_penalty + containment_penalty
    """
    outer_side = _min_enclosing_side(positions, (0.0, 0.0), 0.0)
    return _compute_energy_smooth(positions, outer_side)

def _gradient_refine(positions, max_time=10.0):
    """
    Use L-BFGS-B to refine positions with smooth energy landscape.
    """
    start_time = time.time()
    
    def _flatten(positions):
        return np.array([p[0] for p in positions] + [p[1] for p in positions] + [p[2] for p in positions])
    
    def _unflatten(flat):
        positions = []
        for i in range(N):
            positions.append([flat[i], flat[i+N], flat[i+2*N]])
        return positions
    
    def _objective(flat):
        if time.time() - start_time > max_time:
            return 1e10
        positions = _unflatten(flat)
        # Keep angles in [0, 60)
        for p in positions:
            p[2] = p[2] % 60.0
        return _compute_energy_smooth(positions)
    
    try:
        x0 = _flatten(positions)
        result = minimize(_objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.success or result.fun < _compute_energy(positions):
            refined_positions = _unflatten(result.x)
            for p in refined_positions:
                p[2] = p[2] % 60.0
            return refined_positions
        else:
            return positions
    except Exception:
        return positions

def _simulate_annealing(initial_positions, max_time=100.0):
    """
    Run simulated annealing to optimize the configuration,
    followed by gradient-based refinement.
    """
    start_time = time.time()
    
    # Convert to list of [x, y, angle]
    positions = [list(p) for p in initial_positions]
    
    # Initial energy
    current_energy = _compute_energy(positions)
    best_positions = [list(p) for p in positions]
    best_energy = current_energy
    
    # Annealing parameters
    T_initial = 3.0
    T_final = 0.005
    cooling_rate = 0.999
    
    T = T_initial
    step_size = 0.8
    
    while T > T_final and (time.time() - start_time) < max_time:
        if _remaining_time() < 5.0:
            break
            
        # Generate a random perturbation
        new_positions = [list(p) for p in positions]
        
        # Choose a random hexagon to move
        idx = random.randint(0, N-1)
        
        # Random displacement with adaptive step size
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        dangle = random.gauss(0, step_size * 0.3)
        
        new_positions[idx][0] += dx
        new_positions[idx][1] += dy
        new_positions[idx][2] = (new_positions[idx][2] + dangle) % 60.0
        
        # Compute new energy
        new_energy = _compute_energy(new_positions)
        
        # Metropolis criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 1e-6 and random.random() < np.exp(-delta_e / T)):
            positions = new_positions
            current_energy = new_energy
            
            # Update best if improved
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = [list(p) for p in positions]
        
        # Cool down
        T *= cooling_rate
        
        # Adaptively reduce step size
        if random.random() < 0.02:
            step_size *= 0.97
    
    # Gradient-based refinement
    if _remaining_time() > 5.0:
        refine_time = min(_remaining_time() - 5.0, 15.0)
        best_positions = _gradient_refine(best_positions, max_time=refine_time)
    
    # Get the minimum enclosing side for the best configuration
    best_side = _min_enclosing_side(best_positions, (0.0, 0.0), 0.0)
    
    return best_positions, (0.0, 0.0), best_side, 0.0

def _initial_hexagonal_lattice():
    """Generate initial positions in a hexagonal lattice pattern."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    
    # First ring: 6 hexagons
    dist1 = 2.0 * np.sqrt(3)
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist1 * np.cos(a), dist1 * np.sin(a), 0.0])
    
    # Second ring: 4 more hexagons to reach 11
    dist2 = dist1 * np.sqrt(3)
    angles2 = [30, 90, 150, 210]
    for a_deg in angles2:
        a = np.radians(a_deg)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    
    return positions

def _initial_random_cluster():
    """Generate random initial positions clustered near origin."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        r = random.uniform(1.5, 3.5)
        a = random.uniform(0, 2*np.pi)
        angle = random.uniform(0, 60)
        positions.append([r * np.cos(a), r * np.sin(a), angle])
    return positions

def _initial_spiral():
    """Generate positions in a spiral pattern."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    
    for i in range(1, N):
        r = 1.8 * np.sqrt(i)
        a = i * 2.4  # Golden angle approximation in radians
        angle = 0.0
        positions.append([r * np.cos(a), r * np.sin(a), angle])
    
    return positions

def _try_configuration(initial_positions, max_time=30.0):
    """Try to optimize a configuration using simulated annealing + gradient refinement."""
    try:
        result = _simulate_annealing(initial_positions, max_time=max_time)
        inner, center, side, cangle = result
        
        # Verify validity
        if _check_validity(inner, center, side, cangle, tol=1e-5):
            return inner, center, side, cangle
        else:
            # Try to fix by expanding side length
            for expand in [0.02, 0.05, 0.1, 0.2]:
                side_try = side + expand
                if _check_validity(inner, center, side_try, cangle, tol=1e-5):
                    return inner, center, side_try, cangle
            return None
    except Exception:
        return None

def construct():
    start_time = time.time()
    
    # Different initial configurations to try
    initializations = [
        _initial_hexagonal_lattice,
        _initial_random_cluster,
        _initial_spiral
    ]
    
    best_result = None
    best_side = float('inf')
    
    # Time allocation: spend most time on first (best) init, less on others
    time_budgets = [50.0, 30.0, 20.0]
    
    for i, init_func in enumerate(initializations):
        if _remaining_time() < 8.0:
            break
        
        init_pos = init_func()
        
        # Run simulated annealing + gradient refinement
        result = _try_configuration(init_pos, max_time=time_budgets[i])
        
        if result is not None:
            inner, center, side, cangle = result
            if side < best_side:
                best_side = side
                best_result = result
    
    # If we have time, do additional refinement on the best result
    if best_result is not None and _remaining_time() > 10.0:
        inner, center, side, cangle = best_result
        refined = _try_configuration(inner, max_time=_remaining_time() - 3.0)
        if refined is not None:
            inner_r, center_r, side_r, cangle_r = refined
            if side_r < best_side:
                best_side = side_r
                best_result = refined
    
    # Fallback if no valid result found
    if best_result is None:
        # Simple linear arrangement
        inner = [[3.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        center = (0.0, 0.0)
        side = 20.0
        angle = 0.0
        return inner, center, side, angle
    
    return best_result
