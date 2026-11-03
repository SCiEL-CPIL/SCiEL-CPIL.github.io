import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 3.0


def hex_vertices(cx, cy, angle_deg=0.0):
    """Return the 6 vertices of a unit regular hexagon."""
    a = math.radians(angle_deg)
    pts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        pts.append((cx + math.cos(theta), cy + math.sin(theta)))
    return pts


def min_enclosing_hexagon_side(pts, angle_deg=0.0):
    """
    Find the minimum side length S of a regular hexagon (with given rotation angle_deg)
    that encloses all points in pts. Returns (S, cx, cy).
    
    Uses the fact that a regular hexagon with apothem D has side S = 2*D/sqrt(3) = D / APOTHEM * ... 
    Wait: apothem = S * sqrt(3) / 2, so S = 2 * apothem / sqrt(3) = apothem / APOTHEM * ... 
    Actually APOTHEM = sqrt(3)/2 for side 1, so apothem = S * sqrt(3)/2, thus S = 2*apothem/sqrt(3).
    
    The hexagon with center (cx, cy) and apothem D contains point (px, py) iff
    for all 6 edge normals n_k: (px - cx, py - cy) . n_k <= D
    
    So we need to find (cx, cy, D) minimizing D such that:
    (px_i - cx) * nx_k + (py_i - cy) * ny_k <= D  for all i, k
    which is equivalent to:
    px_i * nx_k - cx * nx_k + py_i * ny_k - cy * ny_k <= D
    => -cx * nx_k - cy * ny_k - D <= -px_i * nx_k - py_i * ny_k
    => cx * nx_k + cy * ny_k + D >= px_i * nx_k + py_i * ny_k
    => cx * nx_k + cy * ny_k + D - (px_i * nx_k + py_i * ny_k) >= 0
    
    Actually let me reformulate as:
    For each point i and each normal k:
    (px_i - cx)*nx_k + (py_i - cy)*ny_k <= D
    
    This is a linear constraint. Variables: cx, cy, D (D >= 0).
    Minimize D.
    """
    if len(pts) == 0:
        return 1.0, 0.0, 0.0
    if len(pts) == 1:
        px, py = pts[0]
        return 1.0, px, py
    
    a = math.radians(angle_deg)
    normals = []
    for k in range(6):
        theta = a + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    
    # Variables: x = [cx, cy, D]
    # Minimize D => c = [0, 0, 1]
    c = np.array([0.0, 0.0, 1.0])
    
    A_ub = []
    b_ub = []
    for nx, ny in normals:
        for px, py in pts:
            # (px - cx)*nx + (py - cy)*ny <= D
            # => px*nx - cx*nx + py*ny - cy*ny <= D
            # => -cx*nx - cy*ny - D <= -(px*nx + py*ny)
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            # Also the reverse: (px - cx)*nx + (py - cy)*ny >= -D
            # => cx*nx + cy*ny - D <= px*nx + py*ny
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            D = res.x[2]
            cx = res.x[0]
            cy = res.x[1]
            S = D / APOTHEM  # since apothem = S * sqrt(3)/2 = S * APOTHEM, so S = D / APOTHEM
            return S, cx, cy
        else:
            return float('inf'), 0.0, 0.0
    except Exception:
        return float('inf'), 0.0, 0.0


def hexagon_overlap_penalty(x1, y1, x2, y2):
    """
    Penalty for two unit hexagons at given positions.
    Returns 0 if no overlap, positive if overlap.
    For same orientation, two unit hexagons don't overlap if center distance >= 2*APOTHEM = sqrt(3) ~ 1.732
    For different orientations, the threshold varies but is at most 2 (circumradius) and at least sqrt(3) (apothem distance).
    We use a smooth penalty based on distance.
    """
    dx = x1 - x2
    dy = y1 - y2
    d = math.sqrt(dx * dx + dy * dy)
    # Two unit hexagons: maximum extent from center is 1.0 (vertex), minimum is APOTHEM ~ 0.866
    # For same orientation (flat-top), edge-to-edge distance is 2*APOTHEM = sqrt(3) ~ 1.732
    # For worst case (vertex-to-vertex aligned), they can overlap up to distance 2.0
    # Use a penalty that is 0 when d >= sqrt(3) and increases as d decreases
    thresh = math.sqrt(3)  # ~1.732
    if d >= thresh:
        return 0.0
    excess = thresh - d
    return excess * excess * 10.0


def compute_total_penalty(positions, angles):
    """Compute total overlap penalty for a set of hexagons."""
    n = len(positions)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            total += hexagon_overlap_penalty(x1, y1, x2, y2)
    return total


def force_directed_relaxation(positions, angles, max_iter=5000, dt=0.02, 
                               repulsion_strength=5.0, attraction_strength=0.5,
                               target_radius=2.5):
    """
    Force-directed relaxation to minimize overlap and keep the cluster compact.
    """
    n = len(positions)
    pos = np.array(positions, dtype=float)
    ang = np.array(angles, dtype=float)
    
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        forces = np.zeros_like(pos)
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 1e-10:
                    dx = (np.random.random() - 0.5) * 0.01
                    dy = (np.random.random() - 0.5) * 0.01
                    dist = math.sqrt(dx * dx + dy * dy)
                
                # Repulsion when overlapping
                thresh = math.sqrt(3)
                if dist < thresh:
                    # Strong repulsion
                    force_mag = repulsion_strength * (thresh - dist) ** 2 / (dist + 1e-10)
                elif dist < 2.5:
                    # Mild repulsion to maintain spacing
                    force_mag = repulsion_strength * 0.05 * (2.5 - dist) / (dist + 1e-10)
                else:
                    force_mag = 0.0
                
                fx = force_mag * dx / dist
                fy = force_mag * dy / dist
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
        
        # Attraction toward center to keep cluster compact
        cx = np.mean(pos[:, 0])
        cy = np.mean(pos[:, 1])
        for i in range(n):
            dx = cx - pos[i, 0]
            dy = cy - pos[i, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0.01:
                # Pull toward center, stronger for farther points
                pull = attraction_strength * dist / (dist + 1e-10)
                forces[i, 0] += pull * dx / dist
                forces[i, 1] += pull * dy / dist
        
        # Update
        pos += dt * forces
        dt *= 0.997
    
    return pos


def evaluate_configuration(positions, angles):
    """
    Evaluate a configuration: return (side_length, center, angle) of minimal enclosing hexagon.
    Also check overlap.
    """
    pts = np.array(positions)
    best_S = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try several outer hexagon angles
    for angle_deg in np.linspace(0, 60, 25):
        S, cx, cy = min_enclosing_hexagon_side(pts, angle_deg)
        if S < best_S:
            best_S = S
            best_center = (cx, cy)
            best_angle = angle_deg
    
    # Check overlap
    penalty = compute_total_penalty(positions, angles)
    if penalty > 0.01:
        best_S = float('inf')
    
    return best_S, best_center, best_angle


def generate_initial_configurations():
    """
    Generate multiple initial configurations for the 11 hexagons.
    """
    configs = []
    
    # Config 1: Hex grid pattern (3 rows)
    # Hex grid: row 0 has 4, row 1 has 4, row 2 has 3
    hex_grid = []
    # Row 0 (y=0): 4 hexagons
    for k in range(4):
        hex_grid.append((k * math.sqrt(3), 0.0))
    # Row 1 (y=1.5): 4 hexagons, offset by sqrt(3)/2
    for k in range(4):
        hex_grid.append((k * math.sqrt(3) + math.sqrt(3) / 2, 1.5))
    # Row 2 (y=3.0): 3 hexagons
    for k in range(3):
        hex_grid.append((k * math.sqrt(3) + math.sqrt(3) / 2, 3.0))
    configs.append(hex_grid[:11])
    
    # Config 2: Ring + center + extras
    ring1 = [(0.0, 0.0)]
    for k in range(6):
        theta = k * math.pi / 3.0
        ring1.append((math.sqrt(3) * math.cos(theta), math.sqrt(3) * math.sin(theta)))
    # Add 4 more on a second ring
    for k in range(4):
        theta = math.pi / 6 + k * math.pi / 3.0
        ring1.append((2 * math.sqrt(3) * math.cos(theta), 2 * math.sqrt(3) * math.sin(theta)))
    configs.append(ring1[:11])
    
    # Config 3: Compact cluster
    compact = [(0.0, 0.0)]
    for k in range(6):
        theta = k * math.pi / 3.0
        compact.append((1.732 * math.cos(theta), 1.732 * math.sin(theta)))
    for k in range(4):
        theta = math.pi / 6 + k * math.pi / 3.0
        compact.append((2.8 * math.cos(theta), 2.8 * math.sin(theta)))
    configs.append(compact[:11])
    
    # Config 4: Random cluster
    np.random.seed(42)
    random_pts = [(np.random.random() * 4 - 2, np.random.random() * 4 - 2) for _ in range(N)]
    configs.append(random_pts)
    
    return configs


def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: return a simple configuration
        fallback = [(0.0, 0.0)]
        for k in range(6):
            theta = k * math.pi / 3.0
            fallback.append((math.sqrt(3) * math.cos(theta), math.sqrt(3) * math.sin(theta)))
        for k in range(4):
            theta = math.pi / 6 + k * math.pi / 3.0
            fallback.append((2 * math.sqrt(3) * math.cos(theta), 2 * math.sqrt(3) * math.sin(theta)))
        pts = np.array(fallback)
        S, cx, cy = min_enclosing_hexagon_side(pts, 0.0)
        inner = [[float(x), float(y), 0.0] for x, y in fallback]
        return inner, [float(cx), float(cy)], float(S), 0.0
    
    best_S = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Generate initial configurations
    initial_configs = generate_initial_configurations()
    
    for init_config in initial_configs:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Use all same orientation (0 degrees) for simplicity
        angles = [0.0] * N
        
        # Force-directed relaxation
        pos = np.array(init_config, dtype=float)
        pos = force_directed_relaxation(pos, angles, max_iter=3000, dt=0.02,
                                        repulsion_strength=5.0, attraction_strength=0.3)
        
        # Evaluate
        S, center, angle = evaluate_configuration(pos, angles)
        if S < best_S:
            best_S = S
            best_positions = pos.copy()
            best_angles = angles.copy()
            best_center = center
            best_angle = angle
    
    # If we have a good configuration, try to improve with local search
    if best_positions is not None:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            # Local search: try small perturbations
            step_size = 0.1
            for it in range(100):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                
                improved = False
                for i in range(N):
                    for dx in [-step_size, 0, step_size]:
                        for dy in [-step_size, 0, step_size]:
                            if dx == 0 and dy == 0:
                                continue
                            new_pos = best_positions.copy()
                            new_pos[i, 0] += dx
                            new_pos[i, 1] += dy
                            S_test, c_test, a_test = evaluate_configuration(new_pos, best_angles)
                            if S_test < best_S:
                                best_S = S_test
                                best_positions = new_pos
                                best_center = c_test
                                best_angle = a_test
                                improved = True
                if not improved:
                    step_size *= 0.7
                    if step_size < 0.01:
                        break
    
    # If still no good configuration, use fallback
    if best_positions is None or best_S == float('inf'):
        fallback = [(0.0, 0.0)]
        for k in range(6):
            theta = k * math.pi / 3.0
            fallback.append((math.sqrt(3) * math.cos(theta), math.sqrt(3) * math.sin(theta)))
        for k in range(4):
            theta = math.pi / 6 + k * math.pi / 3.0
            fallback.append((2 * math.sqrt(3) * math.cos(theta), 2 * math.sqrt(3) * math.sin(theta)))
        best_positions = np.array(fallback)
        best_angles = [0.0] * N
        S, cx, cy = min_enclosing_hexagon_side(best_positions, 0.0)
        best_S = S
        best_center = (cx, cy)
        best_angle = 0.0
    
    # Final: compute the exact minimal enclosing hexagon
    S_final, cx_final, cy_final = min_enclosing_hexagon_side(best_positions, best_angle)
    
    inner = []
    for i in range(N):
        x, y = best_positions[i]
        inner.append([float(x), float(y), float(best_angles[i])])
    
    center = [float(cx_final), float(cy_final)]
    side = float(S_final)
    rotation = float(best_angle)
    
    return inner, center, side, rotation
