import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull

N = 11

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # Unit hexagon vertices centered at origin, angle 0
    unit_hex_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    # Normals of outer hexagon (centered at 0,0, rotation 0)
    # Faces are perpendicular to vectors at angles 0, 60, ...
    normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    def get_hex_verts(x, y, angle_deg):
        """Get vertices of hexagon at (x, y) with given rotation."""
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        verts = unit_hex_verts @ R.T
        verts += np.array([x, y])
        return verts

    def calc_side(centers, angles):
        """Calculate required outer hexagon side length."""
        max_apothem = 0.0
        for k in range(N):
            verts = get_hex_verts(centers[k, 0], centers[k, 1], angles[k])
            # Project all vertices onto all normals
            for v in verts:
                for n in normals:
                    proj = np.dot(v, n)
                    if proj > max_apothem:
                        max_apothem = proj
        # Side S = 2 * apothem / sqrt(3)
        return 2.0 * max_apothem / np.sqrt(3)

    def hexagon_overlap_test(centers, angles, eps=1e-7):
        """
        Check if any two hexagons overlap using vertex-in-polygon test.
        Returns True if no overlap.
        """
        def point_in_hexagon(px, py, cx, cy, angle_deg):
            """Check if point (px,py) is inside hexagon centered at (cx,cy) with rotation."""
            # Transform point to hexagon's local frame
            dx = px - cx
            dy = py - cy
            angle_rad = np.deg2rad(-angle_deg)
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            lx = c * dx - s * dy
            ly = s * dx + c * dy
            # Check against unit hexagon vertices
            for k in range(6):
                v1 = unit_hex_verts[k]
                v2 = unit_hex_verts[(k + 1) % 6]
                # Cross product to check if point is on the correct side of each edge
                ex = v2[0] - v1[0]
                ey = v2[1] - v1[1]
                cross = ex * (ly - v1[1]) - ey * (lx - v1[0])
                if cross > eps:
                    return False
            return True

        for i in range(N):
            for j in range(i + 1, N):
                verts_i = get_hex_verts(centers[i, 0], centers[i, 1], angles[i])
                verts_j = get_hex_verts(centers[j, 0], centers[j, 1], angles[j])
                # Check if any vertex of i is inside j
                for v in verts_i:
                    if point_in_hexagon(v[0], v[1], centers[j, 0], centers[j, 1], angles[j]):
                        return False
                # Check if any vertex of j is inside i
                for v in verts_j:
                    if point_in_hexagon(v[0], v[1], centers[i, 0], centers[i, 1], angles[i]):
                        return False
        return True

    # Generate candidate configurations using hexagonal lattice
    d = np.sqrt(3)  # Distance between centers of adjacent hexagons in lattice

    # Hexagonal lattice positions within a reasonable range
    # Centers at (x, y) where x = a * sqrt(3) + b * sqrt(3)/2, y = b * 1.5
    # for integers a, b

    candidate_positions = []
    for b in range(-3, 4):  # y coordinate index
        for a in range(-3, 4):  # x coordinate index
            x = a * d + b * d / 2
            y = b * 1.5
            candidate_positions.append((x, y))

    # Convert to numpy array
    candidate_positions = np.array(candidate_positions)
    n_candidates = len(candidate_positions)

    best_side = float('inf')
    best_centers = None
    best_angles = None

    def objective(centers, angles):
        """Objective function: side length + penalty for overlaps."""
        side = calc_side(centers, angles)
        
        # Penalty for overlaps using distance heuristic (fast)
        penalty = 0.0
        min_dist = np.sqrt(3)
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 1000
        
        return side + penalty

    def local_search(centers, angles, max_iters=10000, step_init=0.1):
        """Local search to minimize side length while avoiding overlaps."""
        current_centers = centers.copy()
        current_angles = angles.copy()
        current_obj = objective(current_centers, current_angles)
        best_centers = current_centers.copy()
        best_angles = current_angles.copy()
        best_obj = current_obj
        step = step_init
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            idx = np.random.randint(N)
            dx = np.random.normal(0, step)
            dy = np.random.normal(0, step)
            dangle = np.random.normal(0, step * 5)
            
            new_centers = current_centers.copy()
            new_angles = current_angles.copy()
            new_centers[idx, 0] += dx
            new_centers[idx, 1] += dy
            new_angles[idx] += dangle
            
            new_obj = objective(new_centers, new_angles)
            
            if new_obj < current_obj:
                current_centers = new_centers
                current_angles = new_angles
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_centers = current_centers.copy()
                    best_angles = current_angles.copy()
            
            # Decay step size
            step *= 0.9995
        
        return best_centers, best_angles, best_obj

    # Generate multiple initial configurations
    np.random.seed(42)
    
    initial_configs = []
    
    # Config 1: Hexagonal cluster centered at origin
    centers1 = []
    centers1.append((0, 0))
    for k in range(6):
        angle = k * np.pi / 3
        centers1.append((d * np.cos(angle), d * np.sin(angle)))
    for k in range(6):
        angle = k * np.pi / 3 + np.pi / 6
        x = 2 * d * np.cos(angle)
        y = 2 * d * np.sin(angle)
        if len(centers1) < N:
            centers1.append((x, y))
    
    while len(centers1) < N:
        angle = np.random.uniform(0, 2 * np.pi)
        r = 2.5 * d
        centers1.append((r * np.cos(angle), r * np.sin(angle)))
    
    centers1 = np.array(centers1[:N])
    angles1 = np.zeros(N)
    initial_configs.append((centers1, angles1))
    
    # Config 2: Linear arrangement
    centers2 = np.array([(i * d - (N - 1) * d / 2, 0) for i in range(N)])
    angles2 = np.zeros(N)
    initial_configs.append((centers2, angles2))
    
    # Config 3: Two rows
    centers3 = []
    for i in range(N):
        if i < 6:
            centers3.append((i * d - 2.5 * d, 0))
        else:
            centers3.append(((i - 6) * d - 1.5 * d + d / 2, 1.5))
    centers3 = np.array(centers3)
    angles3 = np.zeros(N)
    initial_configs.append((centers3, angles3))
    
    # Config 4: Random scatter
    centers4 = np.random.uniform(-3 * d, 3 * d, (N, 2))
    angles4 = np.random.uniform(-10, 10, N)
    initial_configs.append((centers4, angles4))
    
    # Config 5: Tighter hexagonal pattern
    centers5 = []
    centers5.append((-d / 2, 0))
    centers5.append((d / 2, 0))
    centers5.append((-d, 1.5))
    centers5.append((0, 1.5))
    centers5.append((d, 1.5))
    centers5.append((-d / 2, 3.0))
    centers5.append((d / 2, 3.0))
    centers5.append((-d, -1.5))
    centers5.append((0, -1.5))
    centers5.append((d, -1.5))
    centers5.append((0, -3.0))
    
    centers5 = np.array(centers5[:N])
    angles5 = np.zeros(N)
    initial_configs.append((centers5, angles5))
    
    # Config 6: Another hexagonal arrangement
    centers6 = []
    # Center
    centers6.append((0, 0))
    # First ring
    for k in range(6):
        angle = k * np.pi / 3
        centers6.append((d * np.cos(angle), d * np.sin(angle)))
    # Second ring - pick 4 from 12 possible positions
    for k in [0, 2, 4, 6]:
        angle = k * np.pi / 3
        x = 2 * d * np.cos(angle)
        y = 2 * d * np.sin(angle)
        centers6.append((x, y))
    
    centers6 = np.array(centers6[:N])
    angles6 = np.zeros(N)
    initial_configs.append((centers6, angles6))
    
    # Run local search from each initial configuration
    for config_idx, (init_centers, init_angles) in enumerate(initial_configs):
        if time.time() > deadline - 5:
            break
        
        try:
            result_centers, result_angles, result_obj = local_search(
                init_centers, init_angles, max_iters=15000, step_init=0.15
            )
            
            # Check if the result is valid (no overlaps) using strict test
            if hexagon_overlap_test(result_centers, result_angles):
                side = calc_side(result_centers, result_angles)
                if side < best_side:
                    best_side = side
                    best_centers = result_centers.copy()
                    best_angles = result_angles.copy()
        except Exception:
            pass
    
    # If we haven't found a good solution, try combinatorial search on lattice
    if best_side > 4.0 or best_centers is None:
        # Sort candidate positions by distance from origin
        dists = np.linalg.norm(candidate_positions, axis=1)
        sorted_indices = np.argsort(dists)
        
        # Try subsets of the closest positions
        for start_idx in range(0, min(30, n_candidates - N + 1), 2):
            if time.time() > deadline - 3:
                break
            
            window_start = sorted_indices[start_idx]
            window_end = min(window_start + N + 10, n_candidates)
            candidate_subset = sorted_indices[window_start:window_end]
            
            if len(candidate_subset) < N:
                continue
            
            if len(candidate_subset) <= 14:
                for combo in itertools.combinations(range(len(candidate_subset)), N):
                    indices = candidate_subset[list(combo)]
                    centers = candidate_positions[indices]
                    angles = np.zeros(N)
                    
                    if not hexagon_overlap_test(centers, angles):
                        continue
                    
                    side = calc_side(centers, angles)
                    if side < best_side:
                        best_side = side
                        best_centers = centers.copy()
                        best_angles = angles.copy()
    
    # Final refinement: local search on the best solution
    if best_centers is not None and time.time() < deadline - 5:
        current_centers = best_centers.copy()
        current_angles = best_angles.copy()
        current_obj = objective(current_centers, current_angles)
        
        step_size = 0.05
        for it in range(5000):
            if time.time() > deadline - 2:
                break
            
            idx = np.random.randint(N)
            dx = np.random.normal(0, step_size)
            dy = np.random.normal(0, step_size)
            dangle = np.random.normal(0, step_size * 10)
            
            new_centers = current_centers.copy()
            new_angles = current_angles.copy()
            new_centers[idx, 0] += dx
            new_centers[idx, 1] += dy
            new_angles[idx] += dangle
            
            new_obj = objective(new_centers, new_angles)
            
            if new_obj < current_obj:
                current_centers = new_centers
                current_angles = new_angles
                current_obj = new_obj
            
            step_size *= 0.999
        
        if current_obj < objective(best_centers, best_angles):
            best_centers = current_centers
            best_angles = current_angles
            best_side = calc_side(best_centers, best_angles)
    
    # Construct output
    if best_centers is None:
        # Fallback: widely spaced configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = []
    for i in range(N):
        x, y = best_centers[i, 0], best_centers[i, 1]
        ang = best_angles[i]
        inner.append([float(x), float(y), float(ang)])
    
    return inner, (0.0, 0.0), float(best_side), 0.0
