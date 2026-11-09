import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
EPS = 1e-9

def hex_vertices(cx, cy, s, angle_deg=0.0):
    """Vertices of regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        verts.append([cx + s * math.cos(theta), cy + s * math.sin(theta)])
    return np.array(verts)

def point_in_hex(px, py, cx, cy, s, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) side s rotated angle_deg."""
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    ca, sa = math.cos(-a), math.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    for i in range(6):
        theta = i * math.pi / 3
        if lx * math.cos(theta) + ly * math.sin(theta) > s * math.sqrt(3) / 2 + 1e-12:
            return False
    return True

def _point_in_poly(px, py, poly):
    """Ray casting point in polygon test."""
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

def _seg_intersect(p1, p2, p3, p4):
    """Check if line segments p1-p2 and p3-p4 intersect."""
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

def hexagons_disjoint(v1, v2, s, eps=1e-9):
    """Check if two hexagons (vertex arrays) are disjoint (no interior intersection)."""
    # Check if any vertex of one is inside the other
    for v in v1:
        if _point_in_poly(v[0], v[1], v2):
            return False
    for v in v2:
        if _point_in_poly(v[0], v[1], v1):
            return False
    # Check edge intersections
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j+1)%6]
            if _seg_intersect(p1, p2, p3, p4):
                return False
    return True

def _min_enclosing_side(centers, angles, outer_center, outer_angle):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    sqrt3 = math.sqrt(3)
    a = math.radians(outer_angle)
    # Face normals for regular hexagon with rotation a
    normals = []
    for i in range(6):
        theta = a + math.pi/6 + i * math.pi / 3
        normals.append([math.cos(theta), math.sin(theta)])
    normals = np.array(normals)
    
    # Get all vertices of all inner hexagons
    all_verts = []
    for k in range(N):
        cx, cy = centers[k]
        ang = angles[k]
        verts = hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    dx = all_verts[:, 0] - outer_center[0]
    dy = all_verts[:, 1] - outer_center[1]
    
    max_proj = np.zeros(6)
    min_proj = np.zeros(6)
    for i in range(6):
        proj = dx * normals[i, 0] + dy * normals[i, 1]
        max_proj[i] = proj.max()
        min_proj[i] = proj.min()
    
    required_s = np.maximum(max_proj, -min_proj) / (sqrt3 / 2)
    return required_s.max()

def _check_validity(centers, angles, outer_center, outer_side, outer_angle, margin=1e-8):
    """Check if configuration is valid: inner hexagons disjoint and inside outer."""
    # Check inner hexagons are disjoint
    inner_verts = [hex_vertices(centers[k][0], centers[k][1], 1.0, angles[k]) for k in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if not hexagons_disjoint(inner_verts[i], inner_verts[j], 1.0):
                return False
    
    # Check all inner vertices are inside outer hexagon
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + math.pi/6 + i * math.pi / 3
        normals.append([math.cos(theta), math.sin(theta)])
    normals = np.array(normals)
    
    all_verts = np.vstack(inner_verts)
    dx = all_verts[:, 0] - outer_center[0]
    dy = all_verts[:, 1] - outer_center[1]
    
    for i in range(6):
        proj = dx * normals[i, 0] + dy * normals[i, 1]
        if proj.max() > outer_side * math.sqrt(3) / 2 + margin:
            return False
        if -proj.min() > outer_side * math.sqrt(3) / 2 + margin:
            return False
    
    return True

def _initial_lattice():
    """Generate initial hexagonal lattice configuration."""
    sqrt3 = math.sqrt(3)
    hex_positions_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2),
    ]
    centers = []
    for q, r in hex_positions_axial:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers.append((x, y))
    centers = np.array(centers)
    angles = [0.0] * N
    return centers, angles

def _objective(params):
    """Objective function for optimization: outer side length."""
    centers = params[:2*N].reshape(N, 2)
    angles = params[2*N:3*N]
    
    # Compute center of mass for outer hexagon
    all_verts = []
    for k in range(N):
        cx, cy = centers[k]
        ang = angles[k]
        verts = hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    outer_center = (all_verts[:, 0].mean(), all_verts[:, 1].mean())
    
    # Try a few outer rotations to find best
    best_side = float('inf')
    for outer_angle in [0, 15, 30, 45, 60]:
        side = _min_enclosing_side(centers, angles, outer_center, outer_angle)
        if side < best_side:
            best_side = side
    
    # Add penalty for overlaps (use soft penalty)
    inner_verts = [hex_vertices(centers[k][0], centers[k][1], 1.0, angles[k]) for k in range(N)]
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            # Check distance between centers
            d = np.linalg.norm(centers[i] - centers[j])
            # Minimum distance for disjoint hexagons (edge to edge) is sqrt(3)
            min_dist = math.sqrt(3)
            if d < min_dist - EPS:
                penalty += (min_dist - d) ** 2
    
    return best_side + 1000.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration: hexagonal lattice
    centers, angles = _initial_lattice()
    
    # Compute initial best
    all_verts = []
    for k in range(N):
        verts = hex_vertices(centers[k][0], centers[k][1], 1.0, angles[k])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    cx_mean = all_verts[:, 0].mean()
    cy_mean = all_verts[:, 1].mean()
    
    best_side = float('inf')
    best_center = (cx_mean, cy_mean)
    best_angle = 0.0
    best_centers = centers.copy()
    best_angles = angles.copy()
    
    sqrt3 = math.sqrt(3)
    
    # Optimize outer rotation for initial config
    for outer_angle in np.linspace(0, 60, 121):
        side = _min_enclosing_side(centers, angles, (cx_mean, cy_mean), outer_angle)
        if side < best_side:
            best_side = side
            best_center = (cx_mean, cy_mean)
            best_angle = outer_angle
    
    # Simulated annealing to optimize inner positions
    current_centers = centers.copy()
    current_angles = angles.copy()
    current_side = best_side
    
    # Build initial parameter vector
    def build_params(centers_arr, angles_arr):
        params = np.zeros(3 * N)
        params[:2*N] = centers_arr.flatten()
        params[2*N:3*N] = angles_arr
        return params
    
    current_params = build_params(current_centers, current_angles)
    best_params = current_params.copy()
    
    # Simulated annealing
    T = 0.5
    cooling = 0.995
    step_size = 0.1
    max_iterations = 5000
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Generate perturbation
        new_params = current_params.copy()
        
        # Perturb a random hexagon's position
        k = np.random.randint(N)
        new_params[2*k] += np.random.normal(0, step_size)
        new_params[2*k+1] += np.random.normal(0, step_size)
        
        # Occasionally perturb angle
        if np.random.random() < 0.3:
            new_params[2*N + k] += np.random.normal(0, step_size * 0.5)
        
        new_centers = new_params[:2*N].reshape(N, 2)
        new_angles = new_params[2*N:3*N]
        
        # Compute new objective
        new_side = _objective(new_params)
        
        # Check validity (disjointness)
        valid = True
        inner_verts = [hex_vertices(new_centers[k2][0], new_centers[k2][1], 1.0, new_angles[k2]) for k2 in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j], 1.0):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Reject invalid configurations
            continue
        
        # Accept/reject based on simulated annealing
        delta = new_side - current_side
        if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-10)):
            current_params = new_params
            current_centers = new_centers
            current_angles = new_angles
            current_side = new_side
            
            if new_side < best_side:
                best_side = new_side
                best_params = new_params.copy()
                best_centers = new_centers.copy()
                best_angles = new_angles.copy()
        
        # Cool down
        T *= cooling
        if T < 1e-4:
            T = 1e-4
    
    # Use best found configuration
    final_centers = best_centers
    final_angles = best_angles
    
    # Recompute optimal outer hexagon
    all_verts = []
    for k in range(N):
        verts = hex_vertices(final_centers[k][0], final_centers[k][1], 1.0, final_angles[k])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    cx_final = all_verts[:, 0].mean()
    cy_final = all_verts[:, 1].mean()
    
    best_final_side = float('inf')
    best_final_angle = 0.0
    
    for outer_angle in np.linspace(0, 60, 241):
        side = _min_enclosing_side(final_centers, final_angles, (cx_final, cy_final), outer_angle)
        if side < best_final_side:
            best_final_side = side
            best_final_angle = outer_angle
    
    # Add safety margin
    best_final_side *= 1.0005
    
    # Final verification
    inner_verts = [hex_vertices(final_centers[k][0], final_centers[k][1], 1.0, final_angles[k]) for k in range(N)]
    all_check_verts = np.vstack(inner_verts)
    
    a = math.radians(best_final_angle)
    normals = []
    for i in range(6):
        theta = a + math.pi/6 + i * math.pi / 3
        normals.append([math.cos(theta), math.sin(theta)])
    normals = np.array(normals)
    
    dx = all_check_verts[:, 0] - cx_final
    dy = all_check_verts[:, 1] - cy_final
    
    max_violation = 0
    for i in range(6):
        proj = dx * normals[i, 0] + dy * normals[i, 1]
        max_violation = max(max_violation, proj.max() - best_final_side * sqrt3 / 2)
        max_violation = max(max_violation, -proj.min() - best_final_side * sqrt3 / 2)
    
    if max_violation > 1e-9:
        best_final_side += max_violation / (sqrt3 / 2) * 1.0005
    
    # Build inner list
    inner = [[float(final_centers[k][0]), float(final_centers[k][1]), float(final_angles[k])] for k in range(N)]
    
    return inner, (float(cx_final), float(cy_final)), float(best_final_side), float(best_final_angle)
