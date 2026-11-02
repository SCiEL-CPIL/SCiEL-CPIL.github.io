<<<<<<< SEARCH
    # Fourth pass: Final polish if time remains
    if best_config is not None and not check_time() and (deadline - time.time()) > 15.0:
        w, h, r, xs, ys = best_config
        w_new, h_new, r_new, xs_new, ys_new = refine_contact(w, h, r, xs, ys, is_polish=True)
        
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, h_new, r_new, xs_new, ys_new)
=======
    # Fourth pass: Final polish if time remains
    if best_config is not None and not check_time() and (deadline - time.time()) > 15.0:
        w, h, r, xs, ys = best_config
        w_new, h_new, r_new, xs_new, ys_new = refine_contact(w, h, r, xs, ys, is_polish=True)
        
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, h_new, r_new, xs_new, ys_new)

    # Fifth pass: Ultra-tight polish with very high beta
    if best_config is not None and not check_time() and (deadline - time.time()) > 10.0:
        w, h, r, xs, ys = best_config
        
        # Build contact graph for tight polish
        contact_pairs = get_contact_graph(xs, ys, r, w, h, tolerance=0.01)
        
        if len(contact_pairs) > 0:
            idx_i2 = np.array([p[0] for p in contact_pairs])
            idx_j2 = np.array([p[1] for p in contact_pairs])
            
            beta_ultra = 500.0
            penalty_weight_ultra = 100.0
            contact_weight_ultra = 50.0
            
            def softplus_u(x, beta_val=500.0):
                return np.where(x > 1.0/beta_val, x, np.log1p(np.exp(beta_val * x)) / beta_val)
            
            def d_softplus_u(x, beta_val=500.0):
                z = beta_val * x
                sig = np.where(z > 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
                return sig * beta_val
            
            v0 = np.zeros(2 + 2*N)
            v0[0] = w
            v0[1] = r
            v0[2:2+N] = xs
            v0[2+N:2+2*N] = ys
            
            def objective_u(v):
                w_val = v[0]
                r_val = v[1]
                h_val = 2.0 - w_val
                
                if w_val < 0.1 or w_val > 1.9 or r_val < 0.01 or r_val > 1.0:
                    return 1e6
                
                xs_v = v[2:2+N]
                ys_v = v[2+N:2+2*N]
                
                penalty = 0.0
                
                viol_x_left = r_val - xs_v
                viol_x_right = xs_v - (w_val - r_val)
                viol_y_bottom = r_val - ys_v
                viol_y_top = ys_v - (h_val - r_val)
                
                penalty += np.sum(softplus_u(viol_x_left, beta_ultra))
                penalty += np.sum(softplus_u(viol_x_right, beta_ultra))
                penalty += np.sum(softplus_u(viol_y_bottom, beta_ultra))
                penalty += np.sum(softplus_u(viol_y_top, beta_ultra))
                
                dx = xs_v[idx_i2] - xs_v[idx_j2]
                dy = ys_v[idx_i2] - ys_v[idx_j2]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                viol = 2.0 * r_val - dist
                penalty += np.sum(softplus_u(viol, beta_ultra))
                
                near_mask = dist < 2.0 * r_val * 1.03
                if np.any(near_mask):
                    diff = dist[near_mask] - 2.0 * r_val
                    penalty += contact_weight_ultra * np.sum(softplus_u(np.abs(diff) - 0.005, 200.0))
                
                return -r_val + penalty_weight_ultra * penalty
            
            def grad_objective_u(v):
                w_val = v[0]
                r_val = v[1]
                h_val = 2.0 - w_val
                xs_v = v[2:2+N]
                ys_v = v[2+N:2+2*N]
                
                grad = np.zeros_like(v)
                grad[1] = -1.0
                
                viol_x_left = r_val - xs_v
                viol_x_right = xs_v - (w_val - r_val)
                viol_y_bottom = r_val - ys_v
                viol_y_top = ys_v - (h_val - r_val)
                
                ds_xl = d_softplus_u(viol_x_left, beta_ultra)
                ds_xr = d_softplus_u(viol_x_right, beta_ultra)
                ds_yb = d_softplus_u(viol_y_bottom, beta_ultra)
                ds_yt = d_softplus_u(viol_y_top, beta_ultra)
                
                grad[2:2+N] -= penalty_weight_ultra * ds_xl
                grad[2:2+N] += penalty_weight_ultra * ds_xr
                grad[2+N:2+2*N] -= penalty_weight_ultra * ds_yb
                grad[2+N:2+2*N] += penalty_weight_ultra * ds_yt
                grad[1] += penalty_weight_ultra * (np.sum(ds_xl) + np.sum(ds_yb))
                grad[0] -= penalty_weight_ultra * np.sum(ds_xr)
                
                dx = xs_v[idx_i2] - xs_v[idx_j2]
                dy = ys_v[idx_i2] - ys_v[idx_j2]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                viol = 2.0 * r_val - dist
                ds = d_softplus_u(viol, beta_ultra)
                
                grad_xi = penalty_weight_ultra * ds * (dx / dist)
                grad_xj = -penalty_weight_ultra * ds * (dx / dist)
                grad_yi = penalty_weight_ultra * ds * (dy / dist)
                grad_yj = -penalty_weight_ultra * ds * (dy / dist)
                
                np.add.at(grad[2:2+N], idx_i2, -grad_xi)
                np.add.at(grad[2:2+N], idx_j2, grad_xj)
                np.add.at(grad[2+N:2+2*N], idx_i2, -grad_yi)
                np.add.at(grad[2+N:2+2*N], idx_j2, grad_yj)
                grad[1] += penalty_weight_ultra * np.sum(ds * 2.0)
                
                near_mask = dist < 2.0 * r_val * 1.03
                if np.any(near_mask):
                    idx_i_n = idx_i2[near_mask]
                    idx_j_n = idx_j2[near_mask]
                    dx_n = dx[near_mask]
                    dy_n = dy[near_mask]
                    dist_n = dist[near_mask]
                    diff = dist_n - 2.0 * r_val
                    abs_diff = np.abs(diff)
                    viol_cs = abs_diff - 0.005
                    ds_cs = d_softplus_u(viol_cs, 200.0)
                    sign_diff = np.sign(diff)
                    
                    grad_xi_cs = contact_weight_ultra * ds_cs * sign_diff * (dx_n / dist_n)
                    grad_xj_cs = -contact_weight_ultra * ds_cs * sign_diff * (dx_n / dist_n)
                    grad_yi_cs = contact_weight_ultra * ds_cs * sign_diff * (dy_n / dist_n)
                    grad_yj_cs = -contact_weight_ultra * ds_cs * sign_diff * (dy_n / dist_n)
                    
                    np.add.at(grad[2:2+N], idx_i_n, -grad_xi_cs)
                    np.add.at(grad[2:2+N], idx_j_n, grad_xj_cs)
                    np.add.at(grad[2+N:2+2*N], idx_i_n, -grad_yi_cs)
                    np.add.at(grad[2+N:2+2*N], idx_j_n, grad_yj_cs)
                    grad[1] -= contact_weight_ultra * np.sum(ds_cs * sign_diff * 2.0)
                
                return grad
            
            bounds = [(0.2, 1.8), (0.01, 1.0)] + [(0.0, 2.0)] * (2*N)
            
            try:
                options_u = {'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16, 'disp': False, 'maxls': 100}
                res_u = minimize(objective_u, v0, method='L-BFGS-B', jac=grad_objective_u, bounds=bounds, options=options_u)
                
                if res_u.fun < -r + 1e-8:
                    w_res = res_u.x[0]
                    r_res = res_u.x[1]
                    h_res = 2.0 - w_res
                    xs_res = res_u.x[2:2+N]
                    ys_res = res_u.x[2+N:2+2*N]
                    
                    valid = True
                    if w_res <= 0 or h_res <= 0 or r_res <= 0:
                        valid = False
                    else:
                        for i in range(N):
                            if xs_res[i] < r_res - 1e-8 or xs_res[i] > w_res - r_res + 1e-8:
                                valid = False
                                break
                            if ys_res[i] < r_res - 1e-8 or ys_res[i] > h_res - r_res + 1e-8:
                                valid = False
                                break
                        if valid:
                            dx = xs_res[idx_i2] - xs_res[idx_j2]
                            dy = ys_res[idx_i2] - ys_res[idx_j2]
                            dist_sq = dx*dx + dy*dy
                            if np.any(dist_sq < 4*(r_res**2) - 1e-8):
                                valid = False
                    
                    if valid and r_res > best_r:
                        best_r = r_res
                        best_config = (w_res, h_res, r_res, xs_res, ys_res)
            except Exception:
                pass
>>>>>>> REPLACE

