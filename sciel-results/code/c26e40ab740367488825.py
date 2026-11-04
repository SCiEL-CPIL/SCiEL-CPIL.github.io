import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hex_vertices(center, angle_deg):
    """Return vertices of a unit regular hexagon."""
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    # Vertices at angles 0, 60, ... 300
    angles = np.arange(0, 360, 60) * np.pi / 180
    # Rotate vertices
    x = np.cos(angles + angle_rad)
    y = np.sin(angles + angle_rad)
    # Translate
    x += center[0]
    y += center[1]
    return np.column_stack((x, y))

def polygons_intersect(v1, v2):
    """
    Check if two convex polygons intersect using Separating Axis Theorem.
    v1, v2: Nx2 arrays of vertices.
    """
    def project(p, axis):
        return np.dot(p, axis)
    
    axes = []
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i+1)%len(v1)]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        axes.append(normal)
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i+1)%len(v2)]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        axes.append(normal)
        
    for axis in axes:
        proj1 = project(v1, axis)
        proj2 = project(v2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            return False
    return True

def support_function_hex(center, angle_deg, direction):
    """
    Support function of a unit hexagon at center with rotation angle_deg
    in the direction of the unit vector 'direction'.
    """
    angle_rad = np.deg2rad(angle_rad) # Wait, angle_deg is passed
    # The support function of a regular hexagon of side 1 is:
    # h(theta) = cos(theta - phi) / (2 * cos(pi/6)) ? No.
    # For a regular polygon with circumradius R=1, apothem r=sqrt(3)/2.
    # h(alpha) = r / cos((alpha - phi) mod 60 - 30)?
    # Simpler: compute max dot product of vertices with direction.
    verts = hex_vertices(center, angle_deg)
    return np.max(np.dot(verts, direction))

def calculate_min_side_length(positions, angles_deg, container_angle_deg):
    """
    Calculate the side length of the smallest regular hexagon
    with rotation container_angle_deg that contains all hexes.
    """
    max_width = 0.0
    for k in range(6):
        # Normal directions of the container hexagon
        angle_rad = np.deg2rad(container_angle_deg + k * 60)
        n1 = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        n2 = -n1
        
        # Max projection in direction n1
        max_proj_1 = -np.inf
        min_proj_1 = np.inf
        for i in range(N):
            h1 = support_function_hex(positions[i], angles_deg[i], n1)
            # Support in direction -n1 is equivalent to -min projection in n1
            # Or we can just compute support in -n1
            h2 = support_function_hex(positions[i], angles_deg[i], n2)
            
            max_proj_1 = max(max_proj_1, h1)
            min_proj_1 = min(min_proj_1, -h2) # -h2 is min proj in n1 direction? 
            # Actually, width is max(h(n)) - min(h(n)) = max(h(n)) + max(h(-n))
            # So width_i = h_i(n) + h_i(-n)
            # But we need the width of the UNION.
            # Width of union in direction n is max_i(h_i(n)) - min_i(min_proj_i(n))
            # min_proj_i(n) = -h_i(-n)
            
        # Let's re-calculate properly
        max_h_n = -np.inf
        min_h_n = np.inf
        for i in range(N):
            h_n = support_function_hex(positions[i], angles_deg[i], n1)
            h_neg_n = support_function_hex(positions[i], angles_deg[i], n2)
            # The extent of hex i in direction n1 is [center_i . n1 - h_i(-n1), center_i . n1 + h_i(n1)]?
            # No, support function h(n) gives max(v . n).
            # min(v . n) = -h(-n).
            # So range is [-h_i(-n1), h_i(n1)] relative to center? No, relative to origin.
            # Vertices are at center + v.
            # max((c+v).n) = c.n + h_v(n).
            # min((c+v).n) = c.n - h_v(-n).
            
            max_val = np.dot(positions[i], n1) + h_n
            min_val = np.dot(positions[i], n1) - h_neg_n
            
            max_h_n = max(max_h_n, max_val)
            min_h_n = min(min_h_n, min_val)
            
        width = max_h_n - min_h_n
        # Side length S relates to width W by W = S * sqrt(3)
        S_candidate = width / np.sqrt(3)
        if S_candidate > max_width:
            max_width = S_candidate
            
    return max_width

def cost_function(params):
    # params: [x1, y1, a1, x2, y2, a2, ..., x11, y11, a11, container_angle]
    positions = np.array(params[0:2*N]).reshape(N, 2)
    angles_deg = np.array(params[2*N:3*N])
    container_angle_deg = params[3*N]
    
    # Penalty for overlaps
    verts_list = [hex_vertices(positions[i], angles_deg[i]) for i in range(N)]
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if polygons_intersect(verts_list[i], verts_list[j]):
                penalty += 10.0 # Large penalty
                
    if penalty > 0:
        return penalty + calculate_min_side_length(positions, angles_deg, container_angle_deg)
        
    return calculate_min_side_length(positions, angles_deg, container_angle_deg)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # Leave 20s margin
    
    # Initial configuration: 7 in a flower, 4 in bays
    # Flower centers:
    # 0: (0,0)
    # 1-6: at distance sqrt(3) from center, angles 0, 60, ...
    # But to fit in side 2, they are at distance 2?
    # Let's use the side 2 corner configuration:
    # Centers at distance 2, angles 0, 60, ...
    # This fits in side 2 hexagon.
    
    init_positions = []
    init_angles = [0.0] * N
    
    # Center
    init_positions.append([0.0, 0.0])
    
    # 6 Corners
    for k in range(6):
        angle = k * 60
        rad = np.deg2rad(angle)
        x = 2.0 * np.cos(rad)
        y = 2.0 * np.sin(rad)
        init_positions.append([x, y])
        
    # 4 Bay hexes
    # Bays are at 30, 90, 150, 210, 270, 330
    # We place them at 30, 150, 270, 330 (symmetric)
    # Initial guess for radius: slightly larger than 2?
    # Distance to bay center approx 2.4?
    bay_angles = [30, 150, 270, 330]
    for angle in bay_angles:
        rad = np.deg2rad(angle)
        x = 2.4 * np.cos(rad)
        y = 2.4 * np.sin(rad)
        init_positions.append([x, y])
        
    init_positions = np.array(init_positions)
    
    # Flatten for optimizer
    init_params = np.hstack((init_positions.flatten(), init_angles, [0.0]))
    
    best_params = init_params
    best_cost = cost_function(init_params)
    
    # Try a few different initial rotations for the container to break symmetry
    # And maybe perturb the positions slightly
    
    results = [(best_cost, best_params)]
    
    # Local Search
    # Use Nelder-Mead
    options = {'maxiter': 500, 'fatol': 1e-6}
    
    for attempt in range(5):
        # Perturb initial guess
        if attempt == 0:
            p0 = init_params
        else:
            p0 = init_params + np.random.normal(0, 0.1, size=init_params.shape)
            # Keep container angle small
            p0[-1] = 0.0
            
        try:
            res = minimize(cost_function, p0, method='Nelder-Mead', options=options)
            if res.fun < best_cost:
                best_cost = res.fun
                best_params = res.x
                results.append((best_cost, best_params))
        except Exception:
            pass
            
        if time.time() > deadline:
            break
            
    # Extract best result
    positions = np.array(best_params[0:2*N]).reshape(N, 2)
    angles_deg = np.array(best_params[2*N:3*N])
    container_angle_deg = best_params[3*N]
    
    # Final check and cleanup
    # Ensure angles are within 0-360
    angles_deg = angles_deg % 360.0
    
    # Construct return values
    inner = []
    for i in range(N):
        inner.append([float(positions[i][0]), float(positions[i][1]), float(angles_deg[i])])
        
    center = (0.0, 0.0)
    side = float(best_cost)
    angle = float(container_angle_deg)
    
    return inner, center, side, angle

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
