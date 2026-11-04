<<<<<<< SEARCH
    # --- Phase 2: Smooth Surrogate Optimization with Log-Sum-Exp ---
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 8.0:
        def smooth_objective(params, beta=50.0):
            """
            Smooth objective using log-sum-exp for the max over normals and
            quadratic collision penalty.
            """
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                pos[i, 0] = params[3*i]
                pos[i, 1] = params[3*i+1]
                ang[i] = params[3*i+2]
            
            angles_rad = np.radians(ang)
            all_verts = get_all_vertices(pos, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center

            # Use a moderate set of scan angles for smoothness
            phi_vals = np.radians(np.arange(0, 60, 2))  # 30 angles
            normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            
            # proj shape: (72, 30, 6)
            proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
            abs_proj = np.abs(proj)
            
            # Soft-max over vertices (axis 0) for each normal
            # We want max_j |proj_j|. Using log-sum-exp:
            # max_j x_j ≈ (1/beta) * log(sum(exp(beta * x_j)))
            # Here x_j = abs_proj[j, i, k]
            # Shape of abs_proj is (72, 30, 6). We want soft-max over axis 0.
            # For stability, subtract max first.
            max_abs = np.max(abs_proj, axis=0, keepdims=True)
            exp_term = np.exp(beta * (abs_proj - max_abs))
            soft_max = max_abs + (1.0 / beta) * np.log(np.sum(exp_term, axis=0))
            
            # soft_max shape: (30, 6)
            # Side length for each phi (index i) is (2/sqrt3) * max_k soft_max[i, k]
            # Now we need max over k (axis 1). Apply soft-max again.
            max_soft = np.max(soft_max, axis=1, keepdims=True)
            exp_term2 = np.exp(beta * (soft_max - max_soft))
            soft_max_phi = max_soft + (1.0/beta) * np.log(np.sum(exp_term2, axis=1))
            
            # soft_max_phi shape: (30,)
            # The objective is min_phi soft_max_phi[phi]
            # Since we are minimizing, and this is a surrogate, we can just take the min
            # or even the soft-min. But standard L-BFGS minimizes.
            # If we take min, it's non-smooth.
            # Let's use soft-min: min_i x_i ≈ - (1/beta) * log(sum(exp(-beta * x_i)))
            max_neg = np.max(-soft_max_phi)
            exp_term3 = np.exp(-beta * (soft_max_phi - max_neg)) # Careful with signs
            # Actually: min_i x_i = - max_i (-x_i)
            # soft_min = - (1/beta) log sum exp (beta * (-x_i))
            neg_x = -soft_max_phi
            max_neg_x = np.max(neg_x)
            exp_neg = np.exp(beta * (neg_x - max_neg_x))
            soft_min_val = -(max_neg_x + (1.0/beta) * np.log(np.sum(exp_neg)))
            
            side = (2.0 / sqrt3) * soft_min_val
            
            # Collision Penalty
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    min_dist = 2.0 * sqrt3
                    if dist < min_dist:
                        overlap = min_dist - dist
                        penalty += 50.0 * overlap * overlap
            
            return side + penalty

        def smooth_gradient(params, beta=50.0):
            eps = 1e-5
            grad = np.zeros_like(params)
            f0 = smooth_objective(params, beta)
            for k in range(len(params)):
                params_p = params.copy()
                params_p[k] += eps
                f_p = smooth_objective(params_p, beta)
                params[k] = params_p[k]
                grad[k] = (f_p - f0) / eps
                params[k] = params_p[k]
            return grad

        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = best_positions[i, 0]
            init_params[3*i+1] = best_positions[i, 1]
            init_params[3*i+2] = best_angles[i]
            
        try:
            # Try multiple betas
            for beta in [20.0, 50.0, 100.0]:
                if time.time() - t0 > DEADLINE - 10:
                    break
                res = minimize(smooth_objective, init_params, args=(beta,), method='L-BFGS-B',
                               jac=lambda p, b=beta: smooth_gradient(p, b),
                               options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res.fun < best_side + 0.5 and res.fun < 100:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res.x[3*i]
                        new_pos[i, 1] = res.x[3*i+1]
                        new_ang[i] = res.x[3*i+2]
                    
                    if is_valid_configuration(new_pos, new_ang):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
                            # Update init_params for next beta to start closer
                            init_params = res.x.copy()
        except Exception:
            pass
=======
    # --- Phase 2: Refined Smooth Surrogate Optimization ---
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 8.0:
        # Use a denser angle scan for the surrogate to reduce staircase effect
        # 60 angles (1 degree steps) is a good balance between smoothness and cost
        phi_vals_surrogate = np.radians(np.arange(0, 60, 1))
        
        def smooth_objective(params, beta=50.0):
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                pos[i, 0] = params[3*i]
                pos[i, 1] = params[3*i+1]
                ang[i] = params[3*i+2]
            
            angles_rad = np.radians(ang)
            all_verts = get_all_vertices(pos, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center

            # 60 angles
            normal_angles = phi_vals_surrogate[:, None] + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            
            # proj shape: (72, 60, 6)
            proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
            abs_proj = np.abs(proj)
            
            # Soft-max over vertices (axis 0)
            max_abs = np.max(abs_proj, axis=0, keepdims=True)
            exp_term = np.exp(beta * (abs_proj - max_abs))
            soft_max_verts = max_abs + (1.0 / beta) * np.log(np.sum(exp_term, axis=0))
            
            # Soft-max over normals (axis 1) for each phi
            # soft_max_verts shape: (60, 6)
            max_soft = np.max(soft_max_verts, axis=1, keepdims=True)
            exp_term2 = np.exp(beta * (soft_max_verts - max_soft))
            soft_max_phi = max_soft + (1.0/beta) * np.log(np.sum(exp_term2, axis=1))
            
            # Soft-min over phi (axis 0)
            neg_x = -soft_max_phi
            max_neg_x = np.max(neg_x)
            exp_neg = np.exp(beta * (neg_x - max_neg_x))
            soft_min_val = -(max_neg_x + (1.0/beta) * np.log(np.sum(exp_neg)))
            
            side = (2.0 / sqrt3) * soft_min_val
            
            # Collision Penalty: Use a sharper penalty
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    min_dist = 2.0 * sqrt3
                    if dist < min_dist:
                        overlap = min_dist - dist
                        # Exponential penalty to create a stronger barrier
                        penalty += 100.0 * (math.exp(20.0 * overlap) - 1.0)
            
            return side + penalty

        def smooth_gradient(params, beta=50.0):
            eps = 1e-5
            grad = np.zeros_like(params)
            f0 = smooth_objective(params, beta)
            for k in range(len(params)):
                params_p = params.copy()
                params_p[k] += eps
                f_p = smooth_objective(params_p, beta)
                params[k] = params_p[k]
                grad[k] = (f_p - f0) / eps
                params[k] = params_p[k]
            return grad

        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = best_positions[i, 0]
            init_params[3*i+1] = best_positions[i, 1]
            init_params[3*i+2] = best_angles[i]
            
        try:
            # Try multiple betas
            for beta in [30.0, 60.0, 120.0]:
                if time.time() - t0 > DEADLINE - 10:
                    break
                res = minimize(smooth_objective, init_params, args=(beta,), method='L-BFGS-B',
                               jac=lambda p, b=beta: smooth_gradient(p, b),
                               options={'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res.fun < best_side + 1.0 and res.fun < 100:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res.x[3*i]
                        new_pos[i, 1] = res.x[3*i+1]
                        new_ang[i] = res.x[3*i+2]
                    
                    if is_valid_configuration(new_pos, new_ang):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
                            init_params = res.x.copy()
        except Exception:
            pass
>>>>>>> REPLACE

