"""Packing 12 unit hexagons in a regular hexagon via simulated annealing + binary search."""
import numpy as np
import time
import math

N = 12
DEADLINE_MARGIN = 5.0  # seconds before 150s limit to stop

def _hex_verts(cx, cy, side, angle_rad):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation (radians)."""
    a = angle_rad
    return np.array([
        [cx + side * math.cos(a + math.radians(60 * i)),
         cy + side * math.sin(a + math.radians(60 * i))]
        for i in range(6)
    ])

def _is_inside_hexagon(pts, center, side, angle_rad):
    """Check if all points are inside the outer hexagon. Returns True if all inside."""
    verts = _hex_verts(center[0], center[1], side, angle_rad)
    n = len(verts)
    for pt in pts:
        inside = True
        for i in range(n):
            v1 = verts[i]
            v2 = verts[(i + 1) % n]
            edge = v2 - v1
            rel = pt - v1
            cross = edge[0] * rel[1] - edge[1] * rel[0]
            if cross < -1e-9:
                inside = False
                break
        if not inside:
            return False
    return True

def _sat_overlap_depth(h1, h2):
    """Minimum penetration depth between two convex polygons using SAT. Returns 0 if disjoint."""
    min_depth = float('inf')
    axes = []
    for poly in [h1, h2]:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
    
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2 + 1e-10 or max2 <= min1 + 1e-10:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
    
    return min_depth if min_depth < float('inf') else 0.0

def _check_validity(params, center, side, angle_rad):
    """
    Hard validity check: all inner hexagons must be inside outer hexagon
    and pairwise disjoint. Returns (is_valid, max_violation).
    """
    inner_verts = []
    for i in range(N):
        cx, cy, ang = params[3*i], params[3*i+1], params[3*i+2]
        inner_verts.append(_hex_verts(cx, cy, 1.0, ang))
    
    # Containment check
    for iv in inner_verts:
        if not _is_inside_hexagon(iv, center, side, angle_rad):
            # Compute how far outside
            outer_verts = _hex_verts(center[0], center[1], side, angle_rad)
            n_out = len(outer_verts)
            for pt in iv:
                for i in range(n_out):
                    v1 = outer_verts[i]
                    v2 = outer_verts[(i + 1) % n_out]
                    edge = v2 - v1
                    edge_len = np.linalg.norm(edge)
                    if edge_len < 1e-12:
                        continue
                    rel = pt - v1
                    cross = edge[0] * rel[1] - edge[1] * rel[0]
                    if cross < -1e-9:
                        return False, abs(cross) / edge_len
            return False, 1.0
    
    # Pairwise disjointness
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(inner_verts[i], inner_verts[j])
            if depth > 1e-9:
                return False, depth
    
    return True, 0.0

def _compute_violation(params, center, side, angle_rad):
    """Compute total violation magnitude (for SA energy)."""
    inner_verts = []
    for i in range(N):
        cx, cy, ang = params[3*i], params[3*i+1], params[3*i+2]
        inner_verts.append(_hex_verts(cx, cy, 1.0, ang))
    
    total_viol = 0.0
    
    # Containment
    outer_verts = _hex_verts(center[0], center[1], side, angle_rad)
    n_out = len(outer_verts)
    for iv in inner_verts:
        for pt in iv:
            for i in range(n_out):
                v1 = outer_verts[i]
                v2 = outer_verts[(i + 1) % n_out]
                edge = v2 - v1
                edge_len = np.linalg.norm(edge)
                if edge_len < 1e-12:
                    continue
                rel = pt - v1
                cross = edge[0] * rel[1] - edge[1] * rel[0]
                if cross < -1e-9:
                    total_viol += abs(cross) / edge_len
    
    # Pairwise
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(inner_verts[i], inner_verts[j])
            if depth > 1e-9:
                total_viol += depth
    
    return total_viol

def _make_hex_grid(nrows, ncols):
    """Create hex grid positions centered at origin."""
    positions = []
    for r in range(nrows):
        for c in range(ncols):
            x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
            y = r * math.sqrt(3)
            positions.append([x, y])
    arr = np.array(positions)
    centroid = arr.mean(axis=0)
    return (arr - centroid).tolist()

def _make_ring():
    """1 center + 6 inner ring + 5 outer ring."""
    positions = [[0.0, 0.0]]
    for i in range(6):
        ang = math.radians(60 * i)
        positions.append([2.0 * math.cos(ang), 2.0 * math.sin(ang)])
    for i in range(6):
        ang = math.radians(60 * i + 30)
        positions.append([3.464 * math.cos(ang), 3.464 * math.sin(ang)])
    return positions[:12]

def _make_compact():
    """Compact 2x6 arrangement."""
    positions = []
    for r in range(2):
        for c in range(6):
            x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
            y = r * math.sqrt(3)
            positions.append([x, y])
    arr = np.array(positions)
    centroid = arr.mean(axis=0)
    return (arr - centroid).tolist()

def _make_linear():
    """Linear arrangement in a zigzag."""
    positions = []
    for i in range(12):
        row = i // 4
        col = i % 4
        x = col * 2.0 + (1.0 if row % 2 == 1 else 0.0)
        y = row * math.sqrt(3)
        positions.append([x, y])
    arr = np.array(positions)
    centroid = arr.mean(axis=0)
    return (arr - centroid).tolist()

def _random_params(scale=4.0):
    """Generate random initial parameters."""
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = np.random.uniform(-scale, scale)
        params[3*i+1] = np.random.uniform(-scale, scale)
        params[3*i+2] = np.random.uniform(0, math.pi)
    return params

def _simulated_annealing(init_params, center, side, angle_rad, 
                          max_iter=5000, t_start=1.0, t_end=0.01,
                          deadline=None):
    """
    Simulated annealing to find valid packing.
    Returns (params, is_valid) of best found.
    """
    current = init_params.copy()
    current_energy = _compute_violation(current, center, side, angle_rad)
    
    best = current.copy()
    best_energy = current_energy
    best_valid = current_energy < 1e-8
    
    n_params = len(current)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Temperature schedule
        t = t_start * (t_end / t_start) ** (it / max_iter)
        
        # Generate perturbation
        candidate = current.copy()
        # Perturb 1-3 random hexagons
        n_perturb = np.random.randint(1, 4)
        for _ in range(n_perturb):
            idx = np.random.randint(0, N)
            dx = np.random.normal(0, 0.3)
            dy = np.random.normal(0, 0.3)
            da = np.random.normal(0, 0.2)
            candidate[3*idx] += dx
            candidate[3*idx+1] += dy
            candidate[3*idx+2] = (candidate[3*idx+2] + da) % math.pi
        
        candidate_energy = _compute_violation(candidate, center, side, angle_rad)
        
        # Metropolis criterion
        delta_e = candidate_energy - current_energy
        if delta_e < 0 or np.random.random() < math.exp(-delta_e / max(t, 1e-10)):
            current = candidate
            current_energy = candidate_energy
            
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy
                if best_energy < 1e-8:
                    best_valid = True
        
        if best_valid and best_energy < 1e-10:
            # Found valid, try to verify and break early
            is_v, _ = _check_validity(best, center, side, angle_rad)
            if is_v:
                break
    
    # Final check
    is_v, _ = _check_validity(best, center, side, angle_rad)
    return best, is_v

def _binary_search_side(init_params, center, angle_rad, 
                        lo=3.0, hi=6.0, n_iter=15, 
                        sa_iters=3000, deadline=None):
    """
    Binary search for minimum side length where a valid packing exists.
    Returns (best_side, best_params) or (None, None).
    """
    best_side = hi
    best_params = None
    
    # First check if hi works
    valid_params, is_valid = _simulated_annealing(
        init_params, center, hi, angle_rad, 
        max_iter=sa_iters, deadline=deadline
    )
    if is_valid:
        best_side = hi
        best_params = valid_params
    else:
        # Try harder
        valid_params, is_valid = _simulated_annealing(
            init_params, center, hi, angle_rad,
            max_iter=sa_iters * 2, t_start=1.5, deadline=deadline
        )
        if is_valid:
            best_side = hi
            best_params = valid_params
        else:
            return None, None
    
    # Binary search
    for _ in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        mid = (lo + hi) / 2
        if mid >= best_side - 1e-4:
            break
        
        test_params, is_valid = _simulated_annealing(
            init_params if best_params is None else best_params,
            center, mid, angle_rad,
            max_iter=sa_iters, deadline=deadline
        )
        
        if is_valid:
            hi = mid
            best_side = mid
            best_params = test_params
        else:
            lo = mid
    
    return best_side, best_params

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN  # 145s
    
    np.random.seed(42)
    
    # Initial configurations
    configs = [
        _make_hex_grid(3, 4),
        _make_hex_grid(4, 3),
        _make_hex_grid(2, 6),
        _make_ring(),
        _make_compact(),
        _make_linear(),
    ]
    
    # Also add some random starts
    for _ in range(3):
        configs.append(_random_params(3.0).reshape(N, 3).tolist())
    
    best_side = float('inf')
    best_params = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    outer_angles = [0.0, math.radians(30)]
    
    for angle_deg in outer_angles:
        if time.time() > deadline - 10:
            break
        angle_rad = angle_deg
        
        for config in configs:
            if time.time() > deadline - 10:
                break
            
            # Convert to params array
            init_params = np.zeros(3 * N)
            for i in range(N):
                init_params[3*i] = config[i][0]
                init_params[3*i+1] = config[i][1]
                init_params[3*i+2] = 0.0
            
            # Binary search for minimum side
            side, params = _binary_search_side(
                init_params, (0.0, 0.0), angle_rad,
                lo=3.0, hi=6.0, n_iter=12,
                sa_iters=4000, deadline=deadline
            )
            
            if side is not None and side < best_side:
                best_side = side
                best_params = params
                best_center = (0.0, 0.0)
                best_angle = angle_deg
        
        # Try with slightly offset centers
        if time.time() > deadline - 10:
            continue
        
        for dx in [0.0, 0.2, -0.2]:
            for dy in [0.0, 0.2, -0.2]:
                if time.time() > deadline - 10:
                    break
                center = (dx, dy)
                
                for config in configs[:4]:
                    if time.time() > deadline - 10:
                        break
                    
                    init_params = np.zeros(3 * N)
                    for i in range(N):
                        init_params[3*i] = config[i][0]
                        init_params[3*i+1] = config[i][1]
                        init_params[3*i+2] = 0.0
                    
                    side, params = _binary_search_side(
                        init_params, center, angle_rad,
                        lo=3.0, hi=6.0, n_iter=10,
                        sa_iters=3000, deadline=deadline
                    )
                    
                    if side is not None and side < best_side:
                        best_side = side
                        best_params = params
                        best_center = center
                        best_angle = angle_deg
    
    # If we found something, try to polish with more SA iterations
    if best_params is not None and time.time() < deadline - 5:
        # Try to tighten further
        for _ in range(3):
            if time.time() > deadline - 3:
                break
            new_side = best_side - 0.05
            if new_side < 3.0:
                break
            params, is_valid = _simulated_annealing(
                best_params, best_center, new_side, best_angle,
                max_iter=5000, t_start=0.5, deadline=deadline
            )
            if is_valid:
                best_side = new_side
                best_params = params
    
    # Fallback if nothing found
    if best_params is None:
        # Use a known valid configuration with generous outer hex
        config = _make_hex_grid(3, 4)
        inner = [[float(config[i][0]), float(config[i][1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 5.5, 0.0
    
    # Build output
    inner = []
    for i in range(N):
        x = float(best_params[3*i])
        y = float(best_params[3*i+1])
        ang_deg = math.degrees(float(best_params[3*i+2])) % 360
        inner.append([x, y, ang_deg])
    
    return inner, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_angle)

if __name__ == "__main__":
    result = construct()
    print(f"Result: side={result[2]:.4f}, angle={result[3]:.2f}")
