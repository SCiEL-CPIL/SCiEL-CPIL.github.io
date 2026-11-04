import math
import time
import numpy as np
import random
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    sqrt3 = math.sqrt(3)

    def get_hex_verts(cx, cy, side, angle_rad):
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle_rad=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle_rad that contains all inner hexagons.
        angles are in radians.
        """
        theta_rad = outer_angle_rad
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def sat_gap(poly1, poly2):
        """
        Compute the minimum gap (positive if separated, negative if overlapping)
        between two convex polygons using SAT.
        Returns the maximum separation (minimum gap) over all axes.
        If any axis has positive separation, they are disjoint.
        """
        min_gap = float('inf')
        
        # Axes from poly1
        for i in range(6):
            edge = poly1[(i+1)%6] - poly1[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-9: continue
            axis = normal / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap # Separated
        
        # Axes from poly2
        for i in range(6):
            edge = poly2[(i+1)%6] - poly2[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-9: continue
            axis = normal / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap # Separated
        
        return min_gap

    def is_valid_config(positions, angles, tol=1e-6):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    def exact_objective(params):
        """
        Exact objective function using SAT gap for penalties.
        """
        if time.time() > deadline - 5:
            return 1e6
        
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        penalty = 0.0
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    # Quadratic penalty for overlap
                    penalty += gap * gap
        
        base_score = calc_min_outer_side(positions, angles, 0.0)
        
        # Large weight for penalty to ensure we prioritize valid configs
        return base_score + 100.0 * penalty

    # --- Robust Initialization ---
    # Generate a valid configuration using a hexagonal lattice pattern
    def generate_valid_start():
        # Hexagonal lattice coordinates for side length 1 hexagons
        # Distance between centers in lattice: sqrt(3) for flat-top, or similar
        # Let's use a simple grid and filter, or just use a known good pattern
        # A 3x4 grid of hexagons works well.
        
        positions = []
        angles = [0.0] * N
        
        # Arrange in 3 rows of 4
        # Row 0: y=0, x = -3, -1, 1, 3 (spacing 2*apothem? No, center-to-center for touching is 2*apothem = sqrt(3) if flat-to-flat? 
        # For pointy-top, touching side-to-side distance is 2*apothem = sqrt(3).
        # Let's use spacing sqrt(3) + small epsilon to ensure separation.
        spacing = math.sqrt(3) + 0.1
        
        for r in range(3):
            for c in range(4):
                x = (c - 1.5) * spacing
                y = (r - 1) * 1.5 # Vertical spacing for hex grid is 1.5 * side? 
                # Actually, for pointy-top, vertical distance between rows is 1.5 * side = 1.5
                # Horizontal distance is sqrt(3) * side = sqrt(3)
                # Let's stick to a simple rectangular grid with large enough spacing to be safe initially
                pass
        
        # Simpler: Place them in a circle or grid with large spacing, then let optimizer compress.
        # Or use the lattice points from the previous fallback but ensure they are valid.
        lattice_points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * 1.5
                y = i * sqrt3 / 2.0 + j * sqrt3
                lattice_points.append((x, y))
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        
        init_positions = np.array(lattice_points[:12])
        init_positions -= np.mean(init_positions, axis=0)
        init_angles = np.zeros(N)
        
        # Verify validity
        if is_valid_config(init_positions, init_angles):
             return np.concatenate([init_positions.flatten(), init_angles])
        
        # If not valid (should not happen with this lattice), spread them out more
        for i in range(N):
            init_positions[i] *= 1.5
        return np.concatenate([init_positions.flatten(), init_angles])

    best_params = generate_valid_start()
    best_score = exact_objective(best_params)
    
    # Optimization Loop
    n_starts = 5
    for start_idx in range(n_starts):
        if time.time() > deadline - 20:
            break
        
        # Perturb the current best or use a new random start
        if start_idx == 0:
            x0 = best_params.copy()
        else:
            # Random start with large spread
            positions = np.random.uniform(-5, 5, (N, 2))
            angles = np.random.uniform(0, math.pi/3, N)
            x0 = np.concatenate([positions.flatten(), angles])
        
        try:
            res = minimize(
                exact_objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True}
            )
            
            # Check if the result is better and valid
            if res.fun < best_score:
                new_positions = res.x[:2*N].reshape(N, 2)
                new_angles = res.x[2*N:]
                
                # Strict validation
                if is_valid_config(new_positions, new_angles):
                    final_side = calc_min_outer_side(new_positions, new_angles, 0.0)
                    # The objective includes penalty, so if valid, penalty is 0.
                    # res.fun should be close to final_side.
                    if final_side < best_score:
                        best_score = final_side
                        best_params = res.x.copy()
                        
        except Exception as e:
            pass

    # Final Verification and Return
    final_positions = best_params[:2*N].reshape(N, 2)
    final_angles = best_params[2*N:]
    
    # One last check to ensure we don't return an invalid config
    if not is_valid_config(final_positions, final_angles):
        # Fallback to the known valid start
        valid_start = generate_valid_start()
        final_positions = valid_start[:2*N].reshape(N, 2)
        final_angles = valid_start[2*N:]
        
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        angle_deg = math.degrees(final_angles[i])
        inner.append([final_positions[i][0], final_positions[i][1], angle_deg])
    
    return inner, (0.0, 0.0), final_side, 0.0

if __name__ == "__main__":
    construct()
