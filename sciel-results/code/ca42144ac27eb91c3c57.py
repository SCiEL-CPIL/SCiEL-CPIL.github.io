import numpy as np
import time
import math
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions.
        """
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        # The support function of a regular hexagon of side S centered at origin 
        # in directions 30, 90, 150, 210, 270, 330 degrees is S * sqrt(3) / 2.
        # We need max_k (v . n_k) <= S * sqrt(3) / 2 for all vertices v.
        # Thus S >= 2/sqrt(3) * max_k (v . n_k).
        
        # Normals n_k for hexagon with flat top/bottom (vertices at 0, 60, ...)
        # Vertices are at angles 0, 60, 120...
        # Normals are at 30, 90, 150...
        # n_0 = (cos(30), sin(30)) = (sqrt(3)/2, 1/2)
        # n_1 = (cos(90), sin(90)) = (0, 1)
        # n_2 = (cos(150), sin(150)) = (-sqrt(3)/2, 1/2)
        # n_3 = (cos(210), sin(210)) = (-sqrt(3)/2, -1/2)
        # n_4 = (cos(270), sin(270)) = (0, -1)
        # n_5 = (cos(330), sin(330)) = (sqrt(3)/2, -1/2)
        
        L1 = x * (sqrt3/2) + y * (1/2)
        L2 = y
        L3 = x * (-sqrt3/2) + y * (1/2)
        L4 = x * (-sqrt3/2) + y * (-1/2)
        L5 = -y
        L6 = x * (sqrt3/2) + y * (-1/2)
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        max_proj = np.max(all_L)
        return (2.0 / sqrt3) * max_proj

    def check_validity(positions, tol=1e-4):
        """
        Check if hexagons are disjoint.
        Returns True if valid, False otherwise.
        Uses SAT (Separating Axis Theorem) for hexagon-hexagon collision.
        """
        verts_list = [hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0) for i in range(N)]
        
        for i in range(N):
            for j in range(i + 1, N):
                v1 = verts_list[i]
                v2 = verts_list[j]
                
                # Quick bounding box check
                if (np.max(v1[:,0]) + tol < np.min(v2[:,0]) or 
                    np.max(v2[:,0]) + tol < np.min(v1[:,0]) or
                    np.max(v1[:,1]) + tol < np.min(v2[:,1]) or
                    np.max(v2[:,1]) + tol < np.min(v1[:,1])):
                    continue

                # SAT check
                # Since all inner hexagons have the same orientation (0 degrees),
                # we only need to check 6 unique normal directions.
                
                # Generate normals for hexagon 1
                for k in range(6):
                    edge = v1[(k+1)%6] - v1[k]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10: continue
                    normal /= norm
                    
                    proj1 = np.dot(v1, normal)
                    proj2 = np.dot(v2, normal)
                    
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    # Check for separation
                    if max1 + tol < min2 or max2 + tol < min1:
                        break
                else:
                    # No separating axis found, they overlap.
                    return False
        return True

    def generate_positions(r1, r2, angle_deg):
        """
        Generate positions for 1 center + 6 ring1 + 5 ring2.
        r1: radius of first ring
        r2: radius of second ring
        angle_deg: rotation of the entire configuration (affects ring2 offset relative to ring1? 
                  Actually, let's define angle as the angular offset of the second ring relative to the first).
        """
        positions = np.zeros((N, 2))
        # Center
        positions[0] = [0.0, 0.0]
        
        # Ring 1: 6 hexagons
        for i in range(6):
            theta = np.radians(60 * i)
            positions[i+1, 0] = r1 * np.cos(theta)
            positions[i+1, 1] = r1 * np.sin(theta)
            
        # Ring 2: 5 hexagons
        # To maximize packing, the 5 hexagons should be placed in the "gaps" or aligned optimally.
        # A common symmetric arrangement for 5 items on a ring is to space them by 72 degrees.
        # However, due to the 6-fold symmetry of the inner core, the outer ring might break symmetry 
        # or align with specific gaps.
        # Let's assume the 5 hexagons are placed at angles: offset + k * 72 degrees.
        # The 'angle_deg' parameter will control the offset of this 5-ring.
        
        offset_rad = np.radians(angle_deg)
        for i in range(5):
            theta = offset_rad + np.radians(72 * i)
            positions[i+7, 0] = r2 * np.cos(theta)
            positions[i+7, 1] = r2 * np.sin(theta)
            
        return positions

    def objective(params):
        """
        Objective function for optimization.
        Returns (side_length, is_valid)
        """
        r1, r2, angle = params
        if r1 < 0 or r2 < 0:
            return float('inf'), False
            
        pos = generate_positions(r1, r2, angle)
        if not check_validity(pos):
            return float('inf'), False
            
        side = required_outer_side(pos)
        return side, True

    def nelder_mead(func, x0, step=0.1, max_iter=1000, tol=1e-6):
        """
        Simple Nelder-Mead implementation.
        func takes a 1D array and returns a scalar (infinity if invalid).
        """
        n = len(x0)
        simplex = [x0.copy()]
        # Initialize simplex
        for i in range(n):
            x = x0.copy()
            x[i] += step
            simplex.append(x)
            
        f = [func(x) for x in simplex]
        
        for _ in range(max_iter):
            # Sort simplex by function value
            order = np.argsort(f)
            simplex = [simplex[i] for i in order]
            f = [f[i] for i in order]
            
            # Check convergence
            if abs(f[-1] - f[0]) < tol:
                break
                
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + (centroid - simplex[-1])
            fr = func(xr)
            
            if fr < f[0]:
                # Expansion
                xe = centroid + 2 * (centroid - simplex[-1])
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    f[-1] = fe
                else:
                    simplex[-1] = xr
                    f[-1] = fr
            elif fr < f[-2]:
                simplex[-1] = xr
                f[-1] = fr
            else:
                # Contraction
                xc = centroid + 0.5 * (simplex[-1] - centroid)
                fc = func(xc)
                if fc < f[-1]:
                    simplex[-1] = xc
                    f[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        f[i] = func(simplex[i])
                        
        # Return best point
        best_idx = np.argmin(f)
        return simplex[best_idx], f[best_idx]

    best_side = float('inf')
    best_params = None
    best_positions = None

    # Phase 1: Global Grid Search
    # r1 (distance center to ring1)
    # For unit hexagons, the "kissing" distance is roughly sqrt(3) (side to side) or 2 (vertex to vertex).
    # Let's search r1 from 1.5 to 3.0
    # r2 (distance center to ring2)
    # Let's search r2 from 2.5 to 4.5
    # angle (offset of ring2)
    # 0 to 72 (periodicity)
    
    r1_vals = np.linspace(1.5, 3.0, 16)
    r2_vals = np.linspace(2.5, 4.5, 16)
    angle_vals = np.linspace(0, 72, 13)
    
    # We want to find the best valid configuration.
    # To save time, we can prune obviously invalid or large configurations.
    
    for r1 in r1_vals:
        if time.time() > deadline - 10:
            break
        for r2 in r2_vals:
            if time.time() > deadline - 10:
                break
            for angle in angle_vals:
                pos = generate_positions(r1, r2, angle)
                # Quick validity check
                if check_validity(pos):
                    side = required_outer_side(pos)
                    if side < best_side:
                        best_side = side
                        best_params = [r1, r2, angle]
                        best_positions = pos.copy()

    if best_params is None:
        # Fallback if grid search found nothing (should not happen with these ranges)
        # Try a very large configuration
        pos = generate_positions(3.0, 5.0, 0.0)
        if check_validity(pos):
            best_side = required_outer_side(pos)
            best_params = [3.0, 5.0, 0.0]
            best_positions = pos.copy()
        else:
            # Last resort linear
            best_positions = np.array([[i * 3.0, 0.0] for i in range(N)])
            best_side = required_outer_side(best_positions)
            return [list(map(float, [best_positions[i,0], best_positions[i,1], 0.0])) for i in range(N)], (0.0, 0.0), float(best_side), 0.0

    # Phase 2: Local Refinement with Nelder-Mead
    # Start from the best grid point
    x0 = np.array(best_params)
    
    # Run Nelder-Mead
    # We need a function that returns inf if invalid to guide NM away from collisions
    def nm_func(params):
        if time.time() > deadline - 5:
            return float('inf')
        return objective(params)[0]
        
    x_opt, f_opt = nelder_mead(nm_func, x0, step=0.1, max_iter=200, tol=1e-5)
    
    if f_opt < best_side:
        best_side = f_opt
        best_params = x_opt
        best_positions = generate_positions(best_params[0], best_params[1], best_params[2])
        
    # Final check and return
    if best_positions is None:
        # Should not happen
        best_positions = np.array([[i * 3.0, 0.0] for i in range(N)])
        best_side = required_outer_side(best_positions)

    # Ensure the final result is valid
    if not check_validity(best_positions):
        # If NM drifted into an invalid state (should be prevented by inf penalty, but just in case)
        # Revert to best grid point
        best_positions = generate_positions(best_params[0], best_params[1], best_params[2])
        if not check_validity(best_positions):
             # Fallback to a known valid large config
             best_positions = generate_positions(3.0, 5.0, 0.0)
        best_side = required_outer_side(best_positions)

    final_side = required_outer_side(best_positions)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
