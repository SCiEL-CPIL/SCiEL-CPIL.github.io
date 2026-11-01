import numpy as np
import time
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def sat_disjoint(p1, p2, eps=1e-5):
        """
        Returns True if two convex polygons are disjoint (allowing touching with eps tolerance).
        """
        for i in range(len(p1)):
            j = (i + 1) % len(p1)
            edge = p1[j] - p1[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        for i in range(len(p2)):
            j = (i + 1) % len(p2)
            edge = p2[j] - p2[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        return False

    def check_validity(positions, angles):
        """
        positions: (12, 2) array of (x, y)
        angles: (12,) array of angles in radians
        Returns True if all are pairwise disjoint.
        """
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            a = np.degrees(angles[i])
            verts = hexagon_vertices(x, y, 1.0, a)
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_disjoint(vertices_list[i], vertices_list[j]):
                    return False
        return True

    def min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon (centered at 0,0, rotated by outer_angle)
        that contains all vertices of the inner hexagons.
        """
        # Outer hexagon normals (unit vectors pointing to side midpoints)
        # For a hexagon with vertex at angle 0, sides are at angles 30, 90, 150, 210, 270, 330
        # Normals point to 0, 60, 120, 180, 240, 300? 
        # Let's use the definition: distance from center to side is S * sqrt(3)/2.
        # The 6 normals are at angles: outer_angle + 30, outer_angle + 90, ... (degrees)
        # Wait, standard hexagon vertex at (S, 0). Side opposite is x = -S. Normal (-1, 0).
        # Side between (S,0) and (S/2, S*sqrt(3)/2). Midpoint (3S/4, S*sqrt(3)/4).
        # Vector from origin to midpoint: (3S/4, S*sqrt(3)/4). Angle 30 deg.
        # So normals are at 30, 90, 150, 210, 270, 330 degrees relative to vertex 0.
        
        normal_angles = np.radians(outer_angle + np.array([30, 90, 150, 210, 270, 330]))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for i in range(N):
            x, y = positions[i]
            a = angles[i]
            verts = hexagon_vertices(x, y, 1.0, np.degrees(a))
            # Project all vertices onto all 6 normals
            # proj = verts @ normals.T -> (6, 6)
            projs = verts.dot(normals.T)
            # For each normal direction k, the constraint is max(proj) <= S * sqrt(3)/2
            # So S >= max(proj) / (sqrt(3)/2)
            # We take the max over all vertices and all normals.
            # Actually, we need the max projection for each normal, then max over normals.
            max_projs_per_normal = np.max(projs, axis=1)
            current_max = np.max(max_projs_per_normal)
            if current_max > max_proj:
                max_proj = current_max
                
        return max_proj / (SQRT3 / 2.0)

    def objective_and_grad(params):
        """
        params: (36,) array [x1, y1, a1, ..., x12, y12, a12]
        Returns (loss, gradient)
        Loss = outer_side^2 + lambda * sum(overlap^2)
        """
        if time.time() > deadline:
            return 1e10, np.zeros(3*N)
            
        positions = params[0:2*N].reshape((N, 2))
        angles = params[2*N:3*N]
        
        # 1. Compute outer side length
        outer_side = min_outer_side(positions, angles, 0.0)
        base_loss = outer_side ** 2
        
        # 2. Compute collision penalty (smooth approximation)
        # For each pair, compute minimum distance between convex polygons.
        # Since we don't have a fast smooth distance function, we use a "soft" SAT penalty.
        # If separated, penalty is 0. If overlapping, penalty is depth^2.
        # To make it differentiable, we can approximate the "depth" using the minimum projection gap.
        # Gap = min_axis (min_proj_1 - max_proj_2) OR (min_proj_2 - max_proj_1)
        # If Gap < 0, they overlap by |Gap|.
        # We use sum of squared negative gaps.
        
        penalty = 0.0
        grad_positions = np.zeros((N, 2))
        grad_angles = np.zeros(N)
        
        # Precompute vertices for efficiency? 
        # For N=12, 66 pairs. 66 * (12 verts) is small.
        
        vertices_list = []
        for i in range(N):
            vertices_list.append(hexagon_vertices(positions[i,0], positions[i,1], 1.0, np.degrees(angles[i])))
        
        # Gradients for SAT are tricky. 
        # Alternative: Use a simpler penalty based on center distance if close, 
        # or stick to a coarse grid search + local polish?
        # Given the complexity of analytic gradients for SAT, 
        # let's use a finite difference approach for the gradient? 
        # No, that's slow.
        
        # Better Strategy: 
        # Use a "Repulsive Force" model which is easier to differentiate.
        # Force = k * (min_dist_threshold - actual_dist)^2 if overlapping.
        # But actual_dist is hard.
        
        # Let's try a different smooth objective:
        # Sum over pairs of max(0, R_i + R_j - dist_ij)^2 
        # This is not accurate for hexagons but provides a smooth landscape to get close.
        # Then refine with a local search that uses exact SAT.
        
        # Let's implement the "soft" center-distance penalty for the gradient part,
        # and rely on the structure of the problem (hexagonal symmetry) to help.
        
        # Actually, let's just use a heuristic "soft" overlap metric:
        # For each pair, check the 6 directions.
        # This is getting complicated. Let's use a simpler proxy:
        # Penalty = sum( max(0, 2*sqrt(3) - dist_ij)^2 )
        # 2*sqrt(3) is the distance between centers of touching hexagons (side 1).
        # This assumes they are aligned. If rotated, it's an approximation.
        # However, it provides a smooth gradient to push them apart.
        
        lambda_rep = 100.0
        repulsive_loss = 0.0
        
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[j,0] - positions[i,0]
                dy = positions[j,1] - positions[i,1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = SQRT3 # Approximation for touching sides
                if dist < min_dist:
                    overlap = min_dist - dist
                    repulsive_loss += overlap * overlap
                    # Gradient w.r.t positions
                    # d(overlap^2)/d(dist) = 2*overlap * (-1)
                    # d(dist)/dx_i = -dx/dist
                    # d(overlap)/dx_i = -d(dist)/dx_i = dx/dist
                    # d(overlap^2)/dx_i = 2*overlap * (dx/dist)
                    factor = 2.0 * overlap / dist
                    grad_positions[i,0] -= factor * dx
                    grad_positions[i,1] -= factor * dy
                    grad_positions[j,0] += factor * dx
                    grad_positions[j,1] += factor * dy
        
        total_loss = base_loss + lambda_rep * repulsive_loss
        
        # Gradient of base_loss (outer_side^2)
        # This is harder. outer_side is a max of linear functions (projections).
        # d(outer_side^2)/d(x) = 2 * outer_side * d(outer_side)/d(x)
        # d(outer_side)/d(x) is determined by the "active" constraint (the vertex/normal pair giving the max projection).
        
        # Let's find the active constraint
        normal_angles = np.radians(30 + np.array([0, 60, 120, 180, 240, 300])) # Wait, 30, 90...
        normal_angles = np.radians(np.array([30, 90, 150, 210, 270, 330]))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj_val = 0.0
        active_vertex_idx = -1
        active_normal_idx = -1
        active_hex_idx = -1
        
        for i in range(N):
            verts = vertices_list[i]
            projs = verts.dot(normals.T)
            max_p = np.max(projs)
            if max_p > max_proj_val:
                max_proj_val = max_p
                # Find which vertex and normal
                local_max_idx = np.argmax(projs)
                active_vertex_idx = local_max_idx % 6
                active_normal_idx = local_max_idx // 6
                active_hex_idx = i
                
        # Gradient of max_proj_val w.r.t position of active hexagon
        # proj = v_x * n_x + v_y * n_y
        # v = c + R * [cos(a + theta_v), sin(a + theta_v)]
        # d(proj)/d(c_x) = n_x
        # d(proj)/d(c_y) = n_y
        # d(proj)/d(a) = R * (-sin(a + theta_v) * n_x + cos(a + theta_v) * n_y)
        
        if active_hex_idx != -1:
            n = normals[active_normal_idx]
            grad_positions[active_hex_idx, 0] += 2.0 * outer_side * n[0]
            grad_positions[active_hex_idx, 1] += 2.0 * outer_side * n[1]
            
            # Angle gradient
            theta_v = np.radians(active_vertex_idx * 60)
            a = angles[active_hex_idx]
            # d(proj)/d(a) = -sin(a+theta_v)*n_x + cos(a+theta_v)*n_y
            d_proj_da = -np.sin(a + theta_v) * n[0] + np.cos(a + theta_v) * n[1]
            grad_angles[active_hex_idx] += 2.0 * outer_side * d_proj_da
            
        # Combine gradients
        grad_flat = np.concatenate([grad_positions.flatten(), grad_angles])
        
        # Add gradient of repulsive loss (already in grad_positions)
        # Note: lambda_rep is multiplied in loss, so multiply grad by lambda_rep
        grad_flat[0:2*N] += lambda_rep * grad_positions.flatten()
        
        return total_loss, grad_flat

    # Initial Guess: Honeycomb
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    init_positions = []
    init_angles = []
    for q, r in coords:
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        init_positions.append([x, y])
        init_angles.append(0.0)
        
    init_positions = np.array(init_positions)
    init_angles = np.array(init_angles)
    
    init_params = np.concatenate([init_positions.flatten(), init_angles])
    
    # Optimize
    best_params = init_params
    best_loss = objective_and_grad(init_params)[0]
    
    try:
        res = minimize(
            objective_and_grad,
            init_params,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8, 'maxls': 50}
        )
        
        if res.fun < best_loss:
            best_params = res.x
            best_loss = res.fun
            
    except Exception as e:
        pass

    # Extract best configuration
    best_positions = best_params[0:2*N].reshape((N, 2))
    best_angles = best_params[2*N:3*N]
    
    # Verify validity
    if not check_validity(best_positions, best_angles):
        # Fallback to initial if invalid (should rarely happen with soft constraints)
        best_positions = init_positions
        best_angles = init_angles
        
    # Calculate final outer side
    final_side = min_outer_side(best_positions, best_angles, 0.0)
    
    # Construct output
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i,0]), float(best_positions[i,1]), float(np.degrees(best_angles[i]))])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END