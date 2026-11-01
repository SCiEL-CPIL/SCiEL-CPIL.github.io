import numpy as np
import time
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute basis vectors for hexagon vertices
    # A regular hexagon with side length 1 has circumradius 1.
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees.
    base_angles = np.radians(np.arange(0, 360, 60))
    base_cx = np.cos(base_angles)
    base_sy = np.sin(base_angles)
    
    # Minkowski sum of two unit hexagons (for collision detection)
    # The Minkowski difference of two convex polygons is also convex.
    # For two hexagons, the Minkowski sum of a hexagon and a hexagon is a 12-gon.
    # However, for the "smooth L1 approximation of the maximum support function",
    # we can use the fact that two convex polygons A and B are disjoint iff
    # max_{u in U} (h_A(u) - h_B(u)) < 0 is not quite right.
    # Actually, A and B are disjoint iff max_{u in U} (h_{A-B}(u)) < 0.
    # h_{A-B}(u) = h_A(u) - h_{-B}(u) = h_A(u) - h_B(-u).
    # For centrally symmetric hexagons, h_B(-u) = h_B(u).
    # So we need max_{u in U} (h_A(u) - h_B(u)) < 0.
    # h_A(u) = max_{v in A} v dot u.
    # For a hexagon centered at c with rotation theta, h_A(u) = c dot u + R(theta) dot u,
    # where R(theta) is the support function of the unit hexagon.
    # The unit hexagon support function is the max of dot products with its 6 vertices.
    # Let V be the 6x2 matrix of unit hexagon vertices.
    # h_unit(u) = max(V @ u).
    # This max is non-differentiable. We approximate it with log-sum-exp.
    
    V = np.column_stack((base_cx, base_sy)) # 6x2

    def smooth_max(vals, beta=100.0):
        """Smooth approximation of max using log-sum-exp."""
        m = np.max(vals)
        if m > 1e6:
            return m
        # beta controls sharpness. Higher beta -> closer to max.
        # log(sum(exp(beta * (vals - m)))) / beta + m
        return m + np.log(np.sum(np.exp(beta * (vals - m)))) / beta

    def hex_support(cx, cy, theta, u):
        """
        Compute the smooth support function of a hexagon centered at (cx, cy)
        with rotation theta in direction u (2D vector).
        h(c, theta, u) = c dot u + smooth_max(R(theta) V dot u)
        """
        # Rotate vertices
        # R(theta) is 2x2. We want (R(theta) v) dot u = v dot (R(-theta) u)
        # So we can rotate u instead of vertices, which is faster.
        ct = np.cos(theta)
        st = np.sin(theta)
        # R(-theta) u
        u_rot_x = ct * u[0] + st * u[1]
        u_rot_y = -st * u[0] + ct * u[1]
        u_rot = np.array([u_rot_x, u_rot_y])
        
        # Project vertices onto rotated u
        projections = V @ u_rot
        h_unit = smooth_max(projections)
        
        return cx * u[0] + cy * u[1] + h_unit

    def collision_penalty(params):
        """
        Compute the smooth collision penalty.
        params: array of shape (N, 3) -> [x, y, theta]
        Returns sum of soft violations.
        """
        positions = params[:, :2]
        thetas = params[:, 2]
        
        # We need to check all pairs.
        # To make it differentiable and efficient, we can vectorize over pairs.
        # But N=12, so 66 pairs.
        # We can compute the Minkowski difference support function.
        # h_{A-B}(u) = h_A(u) - h_B(u).
        # If max_u h_{A-B}(u) < 0, they are disjoint.
        # We want to penalize max_u h_{A-B}(u).
        # Let f(u) = h_A(u) - h_B(u).
        # We approximate max_u f(u) by evaluating f at a set of directions U and taking smooth_max.
        # What directions U? The normals to the edges of the Minkowski difference.
        # The Minkowski difference of two hexagons is a 12-gon (if rotations are generic).
        # Its edge normals are the union of the edge normals of A and B.
        # Edge normals of a hexagon are at angles 30, 90, 150, 210, 270, 330.
        # So for each pair, the critical directions are the 6 normals of A and 6 of B.
        # Total 12 directions per pair.
        
        # Base normals for unit hexagon (flat-top):
        # Vertices at 0, 60, ...
        # Edges connect 0-60, 60-120, ...
        # Edge 0-60: vector (cos60-cos0, sin60-sin0) = (-0.5, 0.866). Normal: (0.866, 0.5) -> angle 30.
        # So normals are at 30 + 60*k.
        normal_angles = np.radians(30 + 60 * np.arange(6))
        base_normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, ti = positions[i, 0], positions[i, 1], thetas[i]
                xj, yj, tj = positions[j, 0], positions[j, 1], thetas[j]
                
                # Directions to check: normals of A and normals of B
                # Normals of A are base_normals rotated by ti
                # Normals of B are base_normals rotated by tj
                ct_i, st_i = np.cos(ti), np.sin(ti)
                ct_j, st_j = np.cos(tj), np.sin(tj)
                
                # Rotate base normals for A
                n_i_x = ct_i * base_normals[:, 0] - st_i * base_normals[:, 1]
                n_i_y = st_i * base_normals[:, 0] + ct_i * base_normals[:, 1]
                n_i = np.column_stack((n_i_x, n_i_y))
                
                # Rotate base normals for B
                n_j_x = ct_j * base_normals[:, 0] - st_j * base_normals[:, 1]
                n_j_y = st_j * base_normals[:, 0] + ct_j * base_normals[:, 1]
                n_j = np.column_stack((n_j_x, n_j_y))
                
                # Combine directions
                # We can just check the 12 directions.
                # Actually, the max of h_A - h_B over the union of normals is the exact max.
                all_normals = np.vstack([n_i, n_j])
                
                # Compute h_A(u) - h_B(u) for all u in all_normals
                # h_A(u) = (xi, yi) dot u + smooth_max(V @ (R(-ti) u))
                # h_B(u) = (xj, yj) dot u + smooth_max(V @ (R(-tj) u))
                
                # Vectorized computation
                # u is 12x2
                # R(-ti) u:
                u_i_x = ct_i * all_normals[:, 0] + st_i * all_normals[:, 1]
                u_i_y = -st_i * all_normals[:, 0] + ct_i * all_normals[:, 1]
                proj_i = V @ np.column_stack((u_i_x, u_i_y)).T # 6x12
                h_i_unit = smooth_max(proj_i, axis=0) # 12
                h_i = xi * all_normals[:, 0] + yi * all_normals[:, 1] + h_i_unit
                
                u_j_x = ct_j * all_normals[:, 0] + st_j * all_normals[:, 1]
                u_j_y = -st_j * all_normals[:, 0] + ct_j * all_normals[:, 1]
                proj_j = V @ np.column_stack((u_j_x, u_j_y)).T # 6x12
                h_j_unit = smooth_max(proj_j, axis=0) # 12
                h_j = xj * all_normals[:, 0] + yj * all_normals[:, 1] + h_j_unit
                
                diff = h_i - h_j
                max_diff = smooth_max(diff, beta=100.0)
                
                # Penalty if max_diff > 0
                # Use smooth hinge: max(0, x) approximated by 0.5 * (x + sqrt(x^2 + eps))
                # Or just softplus.
                # If max_diff > 0, they overlap.
                # We want to push them apart.
                # Penalty = max(0, max_diff)
                if max_diff > 0:
                    pen += max_diff
        return pen

    def objective(params):
        """
        Minimize outer hexagon side length + collision penalty.
        params: 3N vector: [x1, y1, th1, ..., xN, yN, thN]
        """
        positions = params[:2*N].reshape(N, 2)
        thetas = params[2*N:]
        
        # Calculate required outer side length
        # The outer hexagon is centered at origin, side length S, rotation 0 (fixed for now, or optimize?).
        # Let's fix outer rotation to 0 to simplify, as the problem allows rotating inner hexagons.
        # Wait, the problem says "outer hexagon center (x, y), side length and rotation".
        # We can optimize the outer rotation too. But usually, aligning with the densest direction is best.
        # Let's assume outer is centered at (0,0) and rotation 0 for now, and just minimize S.
        # Actually, we can shift the whole configuration to center it, but since we minimize S,
        # the center will naturally be at the centroid of the support.
        
        # To contain a hexagon i, we need h_outer(u) >= h_inner_i(u) for all u.
        # h_outer(u) = S * h_unit_hex(u) (if centered at 0, rot 0).
        # So S >= h_inner_i(u) / h_unit_hex(u) for all i, u.
        # S = max_{i, u} (h_inner_i(u) / h_unit_hex(u)).
        # Since h_unit_hex(u) is the support function of the unit hexagon.
        
        # We need to compute this max.
        # The critical u are the normals of the outer hexagon and the normals of the inner hexagons.
        # Outer normals: 30, 90, ...
        # Inner normals: rotated by theta_i.
        
        # For simplicity and speed, let's sample the outer normals and inner normals.
        # Actually, the max of h_A(u)/h_B(u) occurs when u is a normal to A or B.
        
        s_max = 0.0
        # Outer normals
        outer_normals = base_normals # 30, 90, ...
        
        # For each inner hexagon, we need to check its normals and the outer normals.
        for i in range(N):
            xi, yi = positions[i, 0], positions[i, 1]
            ti = thetas[i]
            
            # Inner normals
            ct, st = np.cos(ti), np.sin(ti)
            n_i_x = ct * base_normals[:, 0] - st * base_normals[:, 1]
            n_i_y = st * base_normals[:, 0] + ct * base_normals[:, 1]
            n_i = np.column_stack((n_i_x, n_i_y))
            
            # Combine outer and inner normals
            all_n = np.vstack([outer_normals, n_i])
            
            # Compute h_inner(u)
            # h_inner(u) = (xi, yi) dot u + smooth_max(V @ (R(-ti) u))
            u_x = ct * all_n[:, 0] + st * all_n[:, 1]
            u_y = -st * all_n[:, 0] + ct * all_n[:, 1]
            proj = V @ np.column_stack((u_x, u_y)).T
            h_unit = smooth_max(proj, axis=0)
            h_in = xi * all_n[:, 0] + yi * all_n[:, 1] + h_unit
            
            # Compute h_outer(u) / S = h_unit_hex(u)
            # h_unit_hex(u) = smooth_max(V @ u)
            proj_out = V @ all_n.T
            h_out_unit = smooth_max(proj_out, axis=0)
            
            # Ratio
            ratios = h_in / h_out_unit
            s_max = max(s_max, np.max(ratios))
            
        # Add collision penalty
        pen = collision_penalty(np.hstack([positions.flatten(), thetas]))
        
        return s_max + 100.0 * pen

    # Initial guess
    # Hexagonal packing
    s3 = np.sqrt(3)
    init_params = np.zeros(3 * N)
    for i in range(N):
        # Simple grid or ring
        if i == 0:
            init_params[3*i:3*i+2] = [0, 0]
        else:
            # Ring 1: 6 hexes
            if i < 7:
                a = np.radians(60 * (i-1))
                init_params[3*i:3*i+2] = [s3 * np.cos(a), 1.5 * np.sin(a)]
            else:
                # Ring 2
                k = i - 7
                a = np.radians(60 * k + 30)
                r = 3.0
                init_params[3*i:3*i+2] = [r * np.cos(a), r * np.sin(a)]
        init_params[3*i+2] = 0.0

    best_params = init_params.copy()
    best_val = objective(best_params)
    
    # Optimization
    try:
        res = minimize(
            objective, 
            init_params, 
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50}
        )
        if res.fun < best_val:
            best_params = res.x
            best_val = res.fun
    except Exception:
        pass

    # Extract results
    positions = best_params[:2*N].reshape(N, 2)
    thetas = best_params[2*N:]
    
    # Calculate final side length (without penalty)
    final_side = 0.0
    outer_normals = base_normals
    for i in range(N):
        xi, yi = positions[i, 0], positions[i, 1]
        ti = thetas[i]
        ct, st = np.cos(ti), np.sin(ti)
        n_i_x = ct * base_normals[:, 0] - st * base_normals[:, 1]
        n_i_y = st * base_normals[:, 0] + ct * base_normals[:, 1]
        n_i = np.column_stack((n_i_x, n_i_y))
        all_n = np.vstack([outer_normals, n_i])
        
        u_x = ct * all_n[:, 0] + st * all_n[:, 1]
        u_y = -st * all_n[:, 0] + ct * all_n[:, 1]
        proj = V @ np.column_stack((u_x, u_y)).T
        h_unit = smooth_max(proj, axis=0)
        h_in = xi * all_n[:, 0] + yi * all_n[:, 1] + h_unit
        
        proj_out = V @ all_n.T
        h_out_unit = smooth_max(proj_out, axis=0)
        
        ratios = h_in / h_out_unit
        final_side = max(final_side, np.max(ratios))

    # Prepare output
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(np.degrees(thetas[i]))])
    
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
