import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
EPS = 1e-9

def hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def hex_edge_normals(angle_deg):
    """Return the 6 outward edge normals of a regular hexagon rotated by angle_deg."""
    a = math.radians(angle_deg)
    normals = []
    for i in range(6):
        theta = a + math.pi / 6 + i * math.pi / 3
        normals.append([math.cos(theta), math.sin(theta)])
    return np.array(normals)

def min_enclosing_side(centers, angles, side=1.0):
    """Compute minimal outer hexagon side length and center to contain all inner hexagons.
    
    Returns (side_length, center_x, center_y, outer_angle_deg)
    """
    # Collect all vertices
    all_verts = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = hex_vertices(cx, cy, side, angles[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Try different outer hexagon orientations
    best_side = float('inf')
    best_center = (0, 0)
    best_angle = 0.0
    
    for outer_angle_deg in np.linspace(0, 60, 61):
        a = math.radians(outer_angle_deg)
        normals = []
        for i in range(6):
            theta = a + math.pi / 6 + i * math.pi / 3
            normals.append([math.cos(theta), math.sin(theta)])
        normals = np.array(normals)
        
        # For each normal direction, find max and min projection
        dx = all_verts[:, 0]
        dy = all_verts[:, 1]
        
        max_proj = np.zeros(6)
        min_proj = np.zeros(6)
        for i in range(6):
            proj = dx * normals[i, 0] + dy * normals[i, 1]
            max_proj[i] = proj.max()
            min_proj[i] = proj.min()
        
        # The center should be at the midpoint of the bounding box in each direction
        # For a regular hexagon, the constraint is:
        # For each normal i: (px - cx)*n_i.x + (py - cy)*n_i.y <= s * sqrt(3)/2
        # So: cx*n_i.x + cy*n_i.y >= max_proj[i] - s*sqrt(3)/2
        # And: cx*n_i.x + cy*n_i.y <= min_proj[i] + s*sqrt(3)/2
        
        # The minimal s is determined by the tightest constraint
        # For a given center, s >= max_i (max_proj[i] - center.dot(n_i)) / (sqrt(3)/2)
        # Also: s >= max_i (center.dot(n_i) - min_proj[i]) / (sqrt(3)/2)
        
        # Optimal center is where the max of these is minimized
        # This is a linear programming problem, but we can approximate
        
        # Simple approach: center at centroid
        cx_c = all_verts[:, 0].mean()
        cy_c = all_verts[:, 1].mean()
        
        required_s = 0
        for i in range(6):
            proj_max = (all_verts[:, 0] - cx_c) * normals[i, 0] + (all_verts[:, 1] - cy_c) * normals[i, 1]
            proj_min = proj_max  # recompute
            proj_vals = (all_verts[:, 0] - cx_c) * normals[i, 0] + (all_verts[:, 1] - cy_c) * normals[i, 1]
            pmax = proj_vals.max()
            pmin = proj_vals.min()
            required_s = max(required_s, pmax, -pmin)
        
        side_val = required_s / (math.sqrt(3) / 2)
        
        if side_val < best_side:
            best_side = side_val
            best_center = (cx_c, cy_c)
            best_angle = outer_angle_deg
    
    return best_side, best_center[0], best_center[1], best_angle

def hex_overlap(c1, a1, c2, a2, side=1.0):
    """Check if two hexagons overlap using SAT.
    c1, c2: centers as (x,y) tuples
    a1, a2: angles in degrees
    Returns True if they overlap (interior intersection).
    """
    x1, y1 = c1
    x2, y2 = c2
    
    # Get vertices
    v1 = hex_vertices(x1, y1, side, a1)
    v2 = hex_vertices(x2, y2, side, a2)
    
    # Get edge normals for both
    n1 = hex_edge_normals(a1)
    n2 = hex_edge_normals(a2)
    
    # SAT: project both polygons onto each normal
    # If there's a gap on any normal, no overlap
    
    # Project v1 onto n1
    for i in range(6):
        proj1 = v1[:, 0] * n1[i, 0] + v1[:, 1] * n1[i, 1]
        proj2 = v2[:, 0] * n1[i, 0] + v2[:, 1] * n1[i, 1]
        if proj1.max() < proj2.min() - EPS or proj2.max() < proj1.min() - EPS:
            return False
    
    # Project v1 onto n2
    for i in range(6):
        proj1 = v1[:, 0] * n2[i, 0] + v1[:, 1] * n2[i, 1]
        proj2 = v2[:, 0] * n2[i, 0] + v2[:, 1] * n2[i, 1]
        if proj1.max() < proj2.min() - EPS or proj2.max() < proj1.min() - EPS:
            return False
    
    return True

def check_validity(centers, angles, outer_side, outer_center, outer_angle):
    """Check if the configuration is valid:
    1. All inner hexagons are disjoint
    2. All inner hexagons are inside the outer hexagon
    """
    # Check disjointness
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if hex_overlap(centers[i], angles[i], centers[j], angles[j], 1.0):
                return False
    
    # Check containment in outer hexagon
    outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    def point_in_polygon(px, py, poly):
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    for i in range(len(centers)):
        verts = hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
        for v in verts:
            if not point_in_polygon(v[0], v[1], outer_verts):
                return False
    
    return True

def optimize_outer(centers, angles):
    """Optimize outer hexagon to minimize side length."""
    side, cx, cy, angle = min_enclosing_side(centers, angles)
    return side, (cx, cy), angle

def try_arrangement(centers, angles, deadline):
    """Try to optimize an arrangement and return result if valid."""
    start = time.time()
    
    # Optimize outer
    side, center, angle = optimize_outer(centers, angles)
    
    # Add small safety margin
    side *= 1.0005
    
    # Verify
    if check_validity(centers, angles, side, center, angle):
        return side, centers, angles, center, angle
    
    # If invalid due to containment, increase side
    side *= 1.01
    if check_validity(centers, angles, side, center, angle):
        return side, centers, angles, center, angle
    
    return None

def get_valid_fallback():
    """Return a known valid fallback configuration."""
    sqrt3 = math.sqrt(3)
    
    # Hexagonal lattice arrangement
    axial_positions = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2),
    ]
    
    centers = []
    for q, r in axial_positions:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers.append((x, y))
    
    angles = [0.0] * N
    
    side, center, angle = optimize_outer(centers, angles)
    side *= 1.001
    
    return side, centers, angles, center, angle

def force_directed_relax(centers, angles, deadline, iterations=200):
    """Apply force-directed relaxation to push overlapping hexagons apart."""
    centers = [list(c) for c in centers]
    angles = list(angles)
    
    start = time.time()
    k = 0.5  # spring constant
    damping = 0.9
    
    velocities = [[0, 0] for _ in range(N)]
    
    for it in range(iterations):
        if time.time() > deadline:
            break
        
        forces = [[0, 0] for _ in range(N)]
        
        for i in range(N):
            for j in range(i + 1, N):
                if hex_overlap(tuple(centers[i]), angles[i], tuple(centers[j]), angles[j], 1.0):
                    # Compute separation vector
                    dx = centers[j][0] - centers[i][0]
                    dy = centers[j][1] - centers[i][1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < EPS:
                        dx, dy = 1, 0
                        dist = 1
                    else:
                        dx /= dist
                        dy /= dist
                    
                    # Push apart
                    f = k * (1.0 / max(dist, 0.1))
                    forces[i][0] -= dx * f
                    forces[i][1] -= dy * f
                    forces[j][0] += dx * f
                    forces[j][1] += dy * f
        
        # Update positions
        for i in range(N):
            velocities[i][0] = (velocities[i][0] + forces[i][0]) * damping
            velocities[i][1] = (velocities[i][1] + forces[i][1]) * damping
            centers[i][0] += velocities[i][0]
            centers[i][1] += velocities[i][1]
    
    return [tuple(c) for c in centers], angles

def generate_initial_configurations():
    """Generate diverse initial configurations."""
    configs = []
    sqrt3 = math.sqrt(3)
    
    # 1. Hexagonal lattice
    axial_positions = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2),
    ]
    centers = []
    for q, r in axial_positions:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers.append((x, y))
    configs.append((centers, [0.0] * N))
    
    # 2. Linear zigzag
    centers = []
    for i in range(N):
        x = i * sqrt3 * 1.01
        y = (0.5 if i % 2 == 0 else -0.5) * 1.5
        centers.append((x, y))
    configs.append((centers, [0.0] * N))
    
    # 3. Circular arrangement
    centers = []
    r = 2.0
    for i in range(N):
        theta = 2 * math.pi * i / N
        centers.append((r * math.cos(theta), r * math.sin(theta)))
    configs.append((centers, [0.0] * N))
    
    # 4. Random
    np.random.seed(42)
    centers = []
    for i in range(N):
        x = np.random.uniform(-3, 3)
        y = np.random.uniform(-3, 3)
        centers.append((x, y))
    configs.append((centers, [0.0] * N))
    
    # 5. Hexagonal lattice with slight perturbation
    centers = []
    for q, r in axial_positions:
        x = 1.5 * q + np.random.uniform(-0.05, 0.05)
        y = sqrt3 * (r + 0.5 * q) + np.random.uniform(-0.05, 0.05)
        centers.append((x, y))
    configs.append((centers, [0.0] * N))
    
    return configs

def simulate_annealing(centers, angles, deadline, iterations=500):
    """Simulated annealing to optimize the arrangement."""
    centers = [list(c) for c in centers]
    angles = list(angles)
    
    start = time.time()
    
    # Initial temperature
    T = 0.5
    T_min = 0.001
    cooling_rate = 0.995
    
    best_side = float('inf')
    best_centers = [tuple(c) for c in centers]
    best_angles = list(angles)
    
    for it in range(iterations):
        if time.time() > deadline:
            break
        
        # Create a perturbation
        new_centers = [list(c) for c in centers]
        new_angles = list(angles)
        
        # Perturb 1-2 hexagons
        n_perturb = np.random.randint(1, 3)
        indices = np.random.choice(N, n_perturb, replace=False)
        
        for idx in indices:
            new_centers[idx][0] += np.random.uniform(-0.3, 0.3)
            new_centers[idx][1] += np.random.uniform(-0.3, 0.3)
            new_angles[idx] += np.random.uniform(-15, 15)
        
        # Evaluate
        side, center, angle = optimize_outer([tuple(c) for c in new_centers], new_angles)
        side *= 1.0005
        
        # Check validity
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if hex_overlap(tuple(new_centers[i]), new_angles[i], tuple(new_centers[j]), new_angles[j], 1.0):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Check containment
            outer_verts = hex_vertices(center[0], center[1], side, angle)
            for i in range(N):
                verts = hex_vertices(new_centers[i][0], new_centers[i][1], 1.0, new_angles[i])
                for v in verts:
                    # Point in polygon test
                    inside = False
                    n = len(outer_verts)
                    j = n - 1
                    for k in range(n):
                        xi, yi = outer_verts[k]
                        xj, yj = outer_verts[j]
                        if ((yi > v[1]) != (yj > v[1])) and (v[0] < (xj - xi) * (v[1] - yi) / (yj - yi) + xi):
                            inside = not inside
                        j = k
                    if not inside:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            if side < best_side:
                best_side = side
                best_centers = [tuple(c) for c in new_centers]
                best_angles = list(new_angles)
            # Accept move
            centers = new_centers
            angles = new_angles
        else:
            # Reject move (or accept with probability based on temperature if it's only slightly invalid)
            # For simplicity, just reject
            pass
        
        T *= cooling_rate
    
    return best_side, best_centers, best_angles

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_result = None
    best_side = float('inf')
    
    # Generate initial configurations
    configs = generate_initial_configurations()
    
    for centers, angles in configs:
        if time.time() > deadline:
            break
        
        # First, try force-directed relaxation to fix overlaps
        centers_relaxed, angles_relaxed = force_directed_relax(centers, angles, deadline, iterations=100)
        
        # Try the arrangement
        result = try_arrangement(centers_relaxed, angles_relaxed, deadline)
        if result is not None:
            side, c, a, center, angle = result
            if side < best_side:
                best_side = side
                best_result = (c, a, center, angle)
        
        # Then try simulated annealing from this configuration
        if time.time() > deadline:
            break
        
        sa_side, sa_centers, sa_angles = simulate_annealing(centers_relaxed, angles_relaxed, deadline, iterations=300)
        
        if sa_side < float('inf'):
            result = try_arrangement(sa_centers, sa_angles, deadline)
            if result is not None:
                side, c, a, center, angle = result
                if side < best_side:
                    best_side = side
                    best_result = (c, a, center, angle)
    
    # If no valid result found, use fallback
    if best_result is None:
        side, centers, angles, center, angle = get_valid_fallback()
        inner = [[float(cx), float(cy), float(ang)] for (cx, cy), ang in zip(centers, angles)]
        return inner, (float(center[0]), float(center[1])), float(side), float(angle)
    
    centers, angles, center, angle = best_result
    inner = [[float(cx), float(cy), float(ang)] for (cx, cy), ang in zip(centers, angles)]
    
    return inner, (float(center[0]), float(center[1])), float(best_side), float(angle)
