"""
Strategy: Simulated Annealing with direct outer-hexagon minimization.
Instead of optimizing the outer hexagon parameters alongside the inner ones, 
we treat the outer hexagon as the minimal container for the current inner 
configuration. This reduces the dimensionality and allows the SA to focus 
on packing the inner hexagons tightly. The outer side length is computed 
by scanning the rotation angle of the outer hexagon to find the minimum 
size that contains all inner hexagons.
"""
import numpy as np
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        apothem = side * np.sqrt(3) / 2
        # Normals to the sides of a regular hexagon centered at origin
        # Sides are at 30, 90, 150, 210, 270, 330 degrees
        normal_angles = np.radians(np.arange(30, 360, 60))
        
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def polygons_disjoint_sat(v1, v2):
        """
        Check if two convex polygons are disjoint using Separating Axis Theorem.
        Returns True if disjoint, False if overlapping.
        """
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        return False

    def min_outer_side(inner_positions, outer_angle_deg):
        """
        Calculate the minimum side length of a regular hexagon with given angle
        and center (0,0) that contains all inner hexagons.
        """
        # Collect all vertices of all inner hexagons
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        
        if not all_verts:
            return 0.0
        
        all_verts = np.vstack(all_verts)
        
        # Normals for the outer hexagon at angle outer_angle_deg
        base_normals = np.radians(np.arange(30, 360, 60))
        outer_rad = np.radians(outer_angle_deg)
        normals = base_normals + outer_rad
        
        # Calculate max projection
        max_proj = 0.0
        for na in normals:
            nx = np.cos(na)
            ny = np.sin(na)
            projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        
        # S * sqrt(3)/2 >= max_proj  => S >= max_proj / (sqrt(3)/2)
        return max_proj / (np.sqrt(3) / 2)

    def get_min_outer_config(inner_positions):
        """
        Find the best outer hexagon (center, side, angle) for given inner positions.
        """
        # Scan outer angle from 0 to 60 degrees (symmetry)
        best_s = float('inf')
        best_theta = 0.0
        
        # Scan theta
        for theta in np.arange(0, 60, 1.0):
            s = min_outer_side(inner_positions, theta)
            if s < best_s:
                best_s = s
                best_theta = theta
        
        # Refine theta around best
        for theta in np.arange(best_theta - 1.0, best_theta + 1.0, 0.1):
            s = min_outer_side(inner_positions, theta)
            if s < best_s:
                best_s = s
                best_theta = theta
                
        return best_s, best_theta

    def objective(params):
        """
        params: 3*N values for inner hexagons (x, y, angle).
        Outer hexagon is assumed centered at (0,0).
        Returns the side length of the minimal enclosing outer hexagon.
        """
        inner_positions = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner_positions.append((x, y, a))
        
        # Check disjointness
        disjoint_ok = True
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                ci = (inner_positions[i][0], inner_positions[i][1])
                cj = (inner_positions[j][0], inner_positions[j][1])
                ai = inner_positions[i][2]
                aj = inner_positions[j][2]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                if not polygons_disjoint_sat(v1, v2):
                    disjoint_ok = False
                    penalty += 10.0
        
        s, theta = get_min_outer_config(inner_positions)
        
        if not disjoint_ok:
            return s + penalty
        
        return s

    # Initial configuration: Hexagonal lattice
    init_params = np.zeros(3 * N)
    
    # Center
    init_params[0:3] = [0.0, 0.0, 0.0]
    
    # Ring 1
    for k in range(6):
        angle = np.radians(60 * k)
        idx = (k + 1) * 3
        init_params[idx:idx+3] = [2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0]
        
    # Ring 2 (5 hexagons)
    for k in range(5):
        angle = np.radians(60 * k + 30)
        idx = (k + 7) * 3
        init_params[idx:idx+3] = [3.5 * np.cos(angle), 3.5 * np.sin(angle), 0.0]
    
    # Simulated Annealing
    np.random.seed(42)
    
    current_params = init_params.copy()
    current_score = objective(current_params)
    
    best_params = current_params.copy()
    best_score = current_score
    
    T = 1.0  # Initial temperature
    cooling = 0.995
    
    max_iters = 5000
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Perturb
        new_params = current_params.copy()
        n_move = np.random.randint(1, 3)
        indices = np.random.choice(N, n_move, replace=False)
        
        for idx in indices:
            new_params[3*idx] += np.random.uniform(-0.5, 0.5)
            new_params[3*idx+1] += np.random.uniform(-0.5, 0.5)
            new_params[3*idx+2] += np.random.uniform(-30, 30)
            
        new_score = objective(new_params)
        
        if new_score < current_score or np.random.rand() < np.exp(-(new_score - current_score) / T):
            current_params = new_params
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
                
        T *= cooling
        if T < 0.001:
            T = 0.001
            
    # Extract best inner positions
    inner = []
    for i in range(N):
        inner.append((best_params[3*i], best_params[3*i+1], best_params[3*i+2]))
        
    # Compute final outer side and angle
    s, theta = get_min_outer_config(inner)
    
    center = (0.0, 0.0)
    side = s
    angle = theta
    
    # Validation check
    valid = True
    for cx, cy, ang in inner:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, center[0], center[1], side, angle):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        side *= 1.01
        
    return inner, center, side, angle

# EVOLVE-BLOCK-END
