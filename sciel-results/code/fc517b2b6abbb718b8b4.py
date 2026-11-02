import numpy as np
import time
import math
import random

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices and normals
    angles = np.arange(6) * np.pi / 3
    unit_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    normal_angles = angles + np.pi / 6
    unit_normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))

    def get_hex_verts(x, y, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return unit_verts @ R.T + np.array([x, y])

    def get_hex_normals(theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return unit_normals @ R.T

    def sat_max_separation(v1, n1, v2, n2):
        all_normals = np.vstack((n1, n2))
        proj1 = v1 @ all_normals.T
        proj2 = v2 @ all_normals.T
        min1 = np.min(proj1, axis=0)
        max1 = np.max(proj1, axis=0)
        min2 = np.min(proj2, axis=0)
        max2 = np.max(proj2, axis=0)
        
        # Separation on each axis
        sep1 = min2 - max1
        sep2 = min1 - max2
        overlap_depth = np.minimum(max1, max2) - np.maximum(min1, min2)
        sep3 = -overlap_depth
        
        separations = np.maximum(sep1, np.maximum(sep2, sep3))
        return np.max(separations)

    def compute_objective(params, penalty_weight=10.0, eps=1e-5):
        """
        Compute outer side length + penalty for overlaps.
        Uses a soft penalty for overlaps to allow smooth gradient-like search.
        """
        max_extent = 0.0
        verts_list = []
        normals_list = []
        
        for i in range(N):
            x, y, theta = params[3*i], params[3*i+1], params[3*i+2]
            verts = get_hex_verts(x, y, theta)
            normals = get_hex_normals(theta)
            verts_list.append(verts)
            normals_list.append(normals)
            
            proj = verts @ unit_normals.T
            extents = np.max(np.abs(proj), axis=0)
            local_max = np.max(extents)
            if local_max > max_extent:
                max_extent = local_max
        
        outer_side = 2.0 * max_extent / np.sqrt(3)
        
        # Compute overlap penalties
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                sep = sat_max_separation(verts_list[i], normals_list[i], verts_list[j], normals_list[j])
                if sep < 0:
                    # Overlapping
                    # Smooth penalty: square of overlap depth
                    penalty += (-sep) ** 2
        
        return outer_side + penalty_weight * penalty

    def initial_flower():
        params = np.zeros(3*N)
        params[0:3] = [0, 0, 0]
        # 6 neighbors at distance ~1.732 (sqrt(3)) for edge-to-edge contact
        for i in range(6):
            theta = i * np.pi / 3
            x = 1.732 * np.cos(theta)
            y = 1.732 * np.sin(theta)
            params[3 + 3*i : 6 + 3*i] = [x, y, 0]
        # 5 outer ones. In a hex grid, the next ring is at distance sqrt(3)*2 = 3.464?
        # Actually, the centers of the second ring in a hexagonal close packing are at distance 2*sqrt(3) from center?
        # Let's check. Center (0,0). Neighbors at (sqrt(3), 0), (sqrt(3)/2, 1.5), etc.
        # The "corners" of the hexagon of 7 are at distance 2*sqrt(3).
        # But we only have 5 more. Let's place them at the 5 vertices of the hexagon of 7, skipping one?
        # Or just place them further out.
        for i in range(5):
            theta = i * 2 * np.pi / 5 + np.pi / 5
            x = 3.0 * np.cos(theta)
            y = 3.0 * np.sin(theta)
            idx = 7 + i
            params[3*idx : 3*idx+3] = [x, y, 0]
        return params

    # Simulated Annealing
    def simulate_annealing(initial_params, start_temp, end_temp, steps, deadline):
        current_params = initial_params.copy()
        current_cost = compute_objective(current_params)
        best_params = current_params.copy()
        best_cost = current_cost
        
        # Adaptive step size
        step_size = 1.0
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Cooling
            alpha = start_temp / end_temp
            temp = end_temp * (alpha ** (step / steps))
            
            # Propose new state
            # Perturb a random hexagon
            idx = np.random.randint(N)
            new_params = current_params.copy()
            
            # Move position
            delta_x = np.random.normal(0, step_size)
            delta_y = np.random.normal(0, step_size)
            new_params[3*idx] += delta_x
            new_params[3*idx+1] += delta_y
            
            # Rotate
            if np.random.rand() < 0.3:
                delta_theta = np.random.normal(0, 0.2)
                new_params[3*idx+2] += delta_theta
            
            new_cost = compute_objective(new_params)
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or np.random.rand() < math.exp(-delta_cost / temp):
                current_params = new_params
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_params = current_params.copy()
                    # Reduce step size on improvement
                    step_size *= 0.95
            else:
                # Increase step size slightly on rejection to explore
                step_size *= 1.01
                step_size = min(step_size, 2.0)

        return best_params, best_cost

    # Initial guesses
    initial_guesses = [
        initial_flower(),
    ]
    
    # Add some random perturbations to initial guesses for diversity
    for _ in range(4):
        p = initial_flower() + np.random.normal(0, 0.5, 3*N)
        initial_guesses.append(p)
        
    # Also try a ring configuration
    p_ring = np.zeros(3*N)
    for i in range(12):
        theta = i * 2 * np.pi / 12
        p_ring[3*i] = 2.0 * np.cos(theta)
        p_ring[3*i+1] = 2.0 * np.sin(theta)
        p_ring[3*i+2] = 0
    initial_guesses.append(p_ring)

    best_params_overall = None
    best_cost_overall = np.inf

    # Run SA for each guess
    # We have ~145s. Let's divide time among guesses.
    num_guesses = len(initial_guesses)
    time_per_guess = (deadline - time.time()) / num_guesses
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        
        start_sa = time.time()
        # Estimate steps. Each step takes ~0.1ms? Let's assume 1ms to be safe.
        # 145s / 12 guesses = 12s per guess. 12000 steps.
        steps = 10000
        best_params, best_cost = simulate_annealing(guess, 5.0, 0.01, steps, deadline)
        
        if best_cost < best_cost_overall:
            best_cost_overall = best_cost
            best_params_overall = best_params

    if best_params_overall is None:
        best_params_overall = initial_flower()
        best_cost_overall = compute_objective(best_params_overall)

    # Extract inner hexagons
    inner = []
    for i in range(N):
        x, y, theta = best_params_overall[3*i], best_params_overall[3*i+1], best_params_overall[3*i+2]
        inner.append([float(x), float(y), float(np.degrees(theta))])
    
    # Calculate the actual minimum side length for the final configuration
    # We need to find the minimum outer side length that contains all these hexagons.
    # This is equivalent to finding the minimum enclosing hexagon.
    # The objective function computed this as `outer_side`.
    # However, the penalty might have inflated the "cost" if there were overlaps.
    # Since we want the raw value (outer side length), we should recompute it without penalty.
    
    # Recompute outer side length for the best params
    max_extent = 0.0
    for i in range(N):
        x, y, theta = best_params_overall[3*i], best_params_overall[3*i+1], best_params_overall[3*i+2]
        verts = get_hex_verts(x, y, theta)
        proj = verts @ unit_normals.T
        extents = np.max(np.abs(proj), axis=0)
        local_max = np.max(extents)
        if local_max > max_extent:
            max_extent = local_max
    
    side = float(2.0 * max_extent / np.sqrt(3))
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, side, angle

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side: {side}")
