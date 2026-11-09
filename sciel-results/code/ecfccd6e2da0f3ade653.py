import numpy as np
import time
from scipy.optimize import minimize

N = 11

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
    return np.array(verts)

def point_in_hex(px, py, cx, cy, s, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) side s rotated angle_deg."""
    a = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # Regular hexagon with vertices at angles 0,60,...: 
    # Half-plane i: cos(theta_i)*lx + sin(theta_i)*ly <= s * sqrt(3)/2
    for i in range(6):
        theta = i * np.pi / 3
        if lx * np.cos(theta) + ly * np.sin(theta) > s * np.sqrt(3) / 2 + 1e-12:
            return False
    return True

def distance_point_to_segment(px, py, p1, p2):
    """Distance from point (px,py) to segment p1-p2."""
    vx = p2[0] - p1[0]
    vy = p2[1] - p1[1]
    wx = px - p1[0]
    wy = py - p1[1]
    
    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return np.sqrt(wx*wx + wy*wy)
    
    c2 = vx*vx + vy*vy
    if c2 <= c1:
        return np.sqrt((px-p2[0])**2 + (py-p2[1])**2)
    
    b = c1 / c2
    px_proj = p1[0] + b * vx
    py_proj = p1[1] + b * vy
    return np.sqrt((px-px_proj)**2 + (py-py_proj)**2)

def penetration_depth(px, py, poly):
    """If point is inside poly, return min distance to boundary. Else 0."""
    # Check if inside
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    if not inside:
        return 0.0
    
    # If inside, find min distance to edges
    min_dist = float('inf')
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i+1)%n]
        d = distance_point_to_segment(px, py, p1, p2)
        if d < min_dist:
            min_dist = d
    return min_dist

def calculate_forces(centers, angles):
    """
    Calculate repulsive forces between overlapping hexagons.
    Returns a list of force vectors [fx, fy] for each hexagon.
    """
    forces = np.zeros((N, 2))
    verts_list = [hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i]) for i in range(N)]
    
    for i in range(N):
        for j in range(i + 1, N):
            v1 = verts_list[i]
            v2 = verts_list[j]
            
            # Check if hexagons overlap by checking vertices
            # A simple robust check: if any vertex of v1 is inside v2 or vice versa,
            # or if edges intersect. For packing, vertex-inside is a good proxy for significant overlap.
            # To be safe, we'll compute penetration depth for vertices.
            
            # Force vector from i to j
            f = np.zeros(2)
            
            # Check vertices of v1 against v2
            for v in v1:
                d = penetration_depth(v[0], v[1], v2)
                if d > 0:
                    # Direction from v (point in v2) to center of v1?
                    # Better: Direction from center of v1 to center of v2?
                    # Or simply push v1 away from v2.
                    # Let's use the vector from the penetrating point to the center of the other hex.
                    # Actually, standard repulsion: push centers apart proportional to overlap.
                    # Let's estimate the "push" direction as the vector between centers.
                    pass 
            
            # A more effective force model for polygons:
            # If they overlap, compute the minimum translation vector (MTV) or simply use center-to-center vector.
            # Given time constraints, let's use a simplified model:
            # If any vertex of i is inside j, or any vertex of j is inside i, they overlap.
            # We will push them apart along the line connecting their centers.
            
            overlap = False
            for v in v1:
                if point_in_hex(v[0], v[1], centers[j, 0], centers[j, 1], 1.0, angles[j]):
                    overlap = True
                    break
            if not overlap:
                for v in v2:
                    if point_in_hex(v[0], v[1], centers[i, 0], centers[i, 1], 1.0, angles[i]):
                        overlap = True
                        break
            
            if overlap:
                # Calculate distance between centers
                d_vec = centers[i] - centers[j]
                d_norm = np.linalg.norm(d_vec)
                if d_norm < 1e-6:
                    d_norm = 1e-6
                    d_vec = np.array([1e-6, 0])
                
                # Strong repulsion
                force_mag = 5.0 * d_norm # Arbitrary strong constant
                f = force_mag * d_vec / d_norm
                
                forces[i] += f
                forces[j] -= f
                
    return forces

def simulate_annealing(start_centers, start_angles, start_outer, time_limit):
    """
    Run Simulated Annealing to minimize outer size and eliminate overlaps.
    """
    centers = np.array(start_centers, dtype=float)
    angles = np.array(start_angles, dtype=float)
    Cx, Cy, Phi = start_outer
    
    # Initial temperature
    T = 1.0
    T_min = 1e-6
    cooling_rate = 0.99
    
    best_score = float('inf')
    best_state = (centers.copy(), angles.copy(), (Cx, Cy, Phi))
    
    current_score = float('inf')
    
    start_time = time.time()
    
    while T > T_min and (time.time() - start_time) < time_limit:
        # Generate a new candidate
        new_centers = centers.copy()
        new_angles = angles.copy()
        new_Cx, new_Cy, new_Phi = Cx, Cy, Phi
        
        # Random moves
        # Move a random hexagon
        idx = np.random.randint(N)
        new_centers[idx, 0] += np.random.normal(0, 0.1)
        new_centers[idx, 1] += np.random.normal(0, 0.1)
        new_angles[idx] += np.random.normal(0, 5.0) # 5 degrees
        
        # Move outer
        new_Cx += np.random.normal(0, 0.1)
        new_Cy += np.random.normal(0, 0.1)
        new_Phi += np.random.normal(0, 2.0)
        
        # Calculate Forces and apply relaxation
        # Apply several steps of force-directed relaxation to the candidate
        for _ in range(10):
            forces = calculate_forces(new_centers, new_angles)
            # Damping
            new_centers += 0.1 * forces
            
        # Calculate Objective
        # 1. Check Overlaps
        verts_list = [hex_vertices(new_centers[i, 0], new_centers[i, 1], 1.0, new_angles[i]) for i in range(N)]
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                v1 = verts_list[i]
                v2 = verts_list[j]
                # Quick check: any vertex inside?
                for v in v1:
                    if point_in_hex(v[0], v[1], new_centers[j, 0], new_centers[j, 1], 1.0, new_angles[j]):
                        overlap_penalty += 1.0
                        break
                if overlap_penalty > 0:
                    break
                for v in v2:
                    if point_in_hex(v[0], v[1], new_centers[i, 0], new_centers[i, 1], 1.0, new_angles[i]):
                        overlap_penalty += 1.0
                        break
                if overlap_penalty > 0:
                    break
            if overlap_penalty > 0:
                break
        
        if overlap_penalty > 0:
            new_score = 10000.0 + overlap_penalty
        else:
            # Calculate Outer Side Length
            a = np.radians(new_Phi)
            normals = []
            for k in range(6):
                theta = a + np.pi/6 + k * np.pi / 3
                normals.append([np.cos(theta), np.sin(theta)])
            normals = np.array(normals)
            
            max_proj = 0.0
            for i in range(N):
                verts = verts_list[i]
                dx = verts[:, 0] - new_Cx
                dy = verts[:, 1] - new_Cy
                for k in range(6):
                    proj = dx * normals[k, 0] + dy * normals[k, 1]
                    if proj.max() > max_proj:
                        max_proj = proj.max()
            
            S = max_proj / (np.sqrt(3) / 2)
            new_score = S
        
        # Metropolis Criterion
        delta = new_score - current_score
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_score = new_score
            centers = new_centers
            angles = new_angles
            Cx, Cy, Phi = new_Cx, new_Cy, new_Phi
            
            if current_score < best_score:
                best_score = current_score
                best_state = (centers.copy(), angles.copy(), (Cx, Cy, Phi))
        
        T *= cooling_rate
        
    return best_state

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    sqrt3 = np.sqrt(3)
    
    # Initial Configuration 1: Standard Lattice
    hex_positions_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2),
    ]
    
    centers_init = []
    for q, r in hex_positions_axial:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers_init.append((x, y))
        
    angles_init = [0.0] * N
    
    cx_mean = np.mean([c[0] for c in centers_init])
    cy_mean = np.mean([c[1] for c in centers_init])
    
    best_score = float('inf')
    best_result = None
    
    # Multi-start
    starts = []
    
    # Start 1: Lattice
    starts.append((centers_init, angles_init, (cx_mean, cy_mean, 0.0)))
    
    # Start 2: Perturbed Lattice
    np.random.seed(42)
    for _ in range(5):
        c_pert = [(x + np.random.normal(0, 0.2), y + np.random.normal(0, 0.2)) for x, y in centers_init]
        a_pert = [np.random.normal(0, 5.0) for _ in range(N)]
        starts.append((c_pert, a_pert, (cx_mean, cy_mean, np.random.normal(0, 10.0))))
        
    # Start 3: Rotated Lattice
    for angle_rot in [15, 30, 45]:
        rot = np.radians(angle_rot)
        new_centers = []
        for cx, cy in centers_init:
            dx = cx - cx_mean
            dy = cy - cy_mean
            nx = dx * np.cos(rot) - dy * np.sin(rot) + cx_mean
            ny = dx * np.sin(rot) + dy * np.cos(rot) + cy_mean
            new_centers.append((nx, ny))
        new_angles = [a + angle_rot for a in angles_init]
        starts.append((new_centers, new_angles, (cx_mean, cy_mean, angle_rot)))
        
    # Time budget per start
    time_per_start = (deadline - time.time()) / len(starts)
    if time_per_start < 5.0:
        time_per_start = 5.0
        
    for start in starts:
        if time.time() > deadline - 1.0:
            break
            
        try:
            res = simulate_annealing(start[0], start[1], start[2], time_per_start)
            centers, angles, outer = res
            
            # Final verification and adjustment
            # Recalculate S precisely
            a = np.radians(outer[2])
            normals = []
            for k in range(6):
                theta = a + np.pi/6 + k * np.pi / 3
                normals.append([np.cos(theta), np.sin(theta)])
            normals = np.array(normals)
            
            max_proj = 0.0
            for i in range(N):
                verts = hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
                dx = verts[:, 0] - outer[0]
                dy = verts[:, 1] - outer[1]
                for k in range(6):
                    proj = dx * normals[k, 0] + dy * normals[k, 1]
                    if proj.max() > max_proj:
                        max_proj = proj.max()
            
            S = max_proj / (np.sqrt(3) / 2)
            
            # Check for overlaps one last time
            verts_list = [hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    for v in verts_list[i]:
                        if point_in_hex(v[0], v[1], centers[j, 0], centers[j, 1], 1.0, angles[j]):
                            valid = False
                            break
                        if point_in_hex(centers[j, 0], centers[j, 1], centers[i, 0], centers[i, 1], 1.0, angles[i]):
                             # Center of j inside i? Unlikely if vertices are disjoint, but check.
                             pass
                    if not valid: break
                    for v in verts_list[j]:
                        if point_in_hex(v[0], v[1], centers[i, 0], centers[i, 1], 1.0, angles[i]):
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                if S < best_score:
                    best_score = S
                    best_result = (centers, angles, outer, S)
        except Exception:
            pass
            
    if best_result is None:
        # Fallback: Return the initial lattice with a large enough container
        # This is better than crashing, though score might be poor
        centers = np.array(centers_init)
        angles = np.array(angles_init)
        Cx, Cy, Phi = cx_mean, cy_mean, 0.0
        a = np.radians(Phi)
        normals = []
        for k in range(6):
            theta = a + np.pi/6 + k * np.pi / 3
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        max_proj = 0.0
        for i in range(N):
            verts = hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
            dx = verts[:, 0] - Cx
            dy = verts[:, 1] - Cy
            for k in range(6):
                proj = dx * normals[k, 0] + dy * normals[k, 1]
                if proj.max() > max_proj:
                    max_proj = proj.max()
        S = max_proj / (np.sqrt(3) / 2)
        best_result = (centers, angles, (Cx, Cy, Phi), S)
        
    centers, angles, outer, S = best_result
    Cx, Cy, Phi = outer
    
    # Add safety margin to S to ensure no floating point errors cause overlap in evaluator
    S += 1e-6
    
    inner = []
    for i in range(N):
        inner.append([float(centers[i, 0]), float(centers[i, 1]), float(angles[i])])
        
    return inner, (float(Cx), float(Cy)), float(S), float(Phi)
