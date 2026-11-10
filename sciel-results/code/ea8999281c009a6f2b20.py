<<<<<<< SEARCH
    # 7. Final polish on the best candidate
    if best_c is not None and time.time() < deadline - 2:
        polished = sphere_polish(best_c, max_iter=500, lr=0.02)
        val_p, _ = objective_and_grad(polished)
        if val_p < best_val:
            best_val = val_p
            best_c = polished.copy()
=======
    # 7. Final polish on the best candidate
    if best_c is not None and time.time() < deadline - 2:
        polished = sphere_polish(best_c, max_iter=500, lr=0.02)
        val_p, _ = objective_and_grad(polished)
        if val_p < best_val:
            best_val = val_p
            best_c = polished.copy()
    
    # 7b. High-precision Newton-style polish: solve for tangent direction
    # that minimizes R using local quadratic approximation on the sphere
    if best_c is not None and time.time() < deadline - 1:
        u = best_c.copy()
        norm_u = np.linalg.norm(u)
        if norm_u > 1e-12:
            u = u / norm_u
            for newton_iter in range(30):
                if time.time() > deadline - 1:
                    break
                f_vals = A @ u
                f2 = f_vals ** 2
                f4 = f2 ** 2
                norm2_sq = d_phi * np.sum(w_flat * f2)
                norm4_4 = d_phi * np.sum(w_flat * f4)
                if norm2_sq < 1e-12:
                    break
                N2 = np.sqrt(norm2_sq)
                N4 = norm4_4 ** 0.25
                R_curr = N4 / N2
                
                # Gradient on sphere
                sum_wf_A = d_phi * np.dot(B_T, f_vals)
                sum_wf3_A = d_phi * np.dot(B_T, f_vals ** 3)
                grad_u = (N4 ** 3) * sum_wf3_A - (N4 / N2 ** 3) * sum_wf_A
                u_dot_grad = np.dot(u, grad_u)
                grad_tangent = grad_u - u * u_dot_grad
                grad_norm = np.linalg.norm(grad_tangent)
                if grad_norm < 1e-14:
                    break
                
                # Approximate Hessian on tangent space via finite differences
                # Use a few directions to build a small Hessian
                n_tangent = n_m - 1
                # Orphan basis for tangent space
                # Use QR to get tangent basis
                from scipy.linalg import qr
                Q, R_mat = qr(np.eye(n_m) - np.outer(u, u), mode='economic')
                # Q is n_m x n_tangent
                # Compute Hessian in tangent coordinates
                H_tangent = np.zeros((n_tangent, n_tangent))
                eps_h = 1e-5
                g_center = grad_tangent.copy()
                for i in range(n_tangent):
                    if time.time() > deadline - 1:
                        break
                    e_i = np.zeros(n_m)
                    e_i += Q[:, i] * eps_h
                    u_plus = u + e_i
                    u_plus /= np.linalg.norm(u_plus)
                    f_plus = A @ u_plus
                    f2_plus = f_plus ** 2
                    f4_plus = f2_plus ** 2
                    n2_plus = d_phi * np.sum(w_flat * f2_plus)
                    n4_plus = d_phi * np.sum(w_flat * f4_plus)
                    if n2_plus < 1e-12:
                        continue
                    N2p = np.sqrt(n2_plus)
                    N4p = n4_plus ** 0.25
                    swf_A_p = d_phi * np.dot(B_T, f_plus)
                    swf3_A_p = d_phi * np.dot(B_T, f_plus ** 3)
                    grad_plus = (N4p ** 3) * swf3_A_p - (N4p / N2p ** 3) * swf_A_p
                    u_dot_g_plus = np.dot(u_plus, grad_plus)
                    grad_plus_t = grad_plus - u_plus * u_dot_g_plus
                    
                    H_tangent[i, :] = (grad_plus_t - g_center) / eps_h
                    # Symmetrize row
                    H_tangent[:, i] = H_tangent[i, :].copy()
                H_tangent = (H_tangent + H_tangent.T) / 2.0
                
                # Solve H_tangent * d = -grad_tangent (projected)
                # Map grad_tangent to tangent coordinates
                grad_tangent_coords = Q.T @ grad_tangent
                try:
                    d_tangent = np.linalg.solve(H_tangent + 1e-10 * np.eye(n_tangent), -grad_tangent_coords)
                except np.linalg.LinAlgError:
                    break
                
                # Newton step
                u_new = u + Q @ d_tangent
                u_new_norm = np.linalg.norm(u_new)
                if u_new_norm < 1e-12:
                    break
                u_new = u_new / u_new_norm
                
                f_new = A @ u_new
                f2_new = f_new ** 2
                f4_new = f2_new ** 2
                n2_new = d_phi * np.sum(w_flat * f2_new)
                n4_new = d_phi * np.sum(w_flat * f4_new)
                if n2_new < 1e-12:
                    break
                R_new = (n4_new ** 0.25) / np.sqrt(n2_new)
                
                if R_new < R_curr:
                    u = u_new
                    best_c = u.copy()
                    best_val = R_new
                else:
                    # Damped Newton
                    for damp in [0.5, 0.25, 0.1]:
                        u_damped = u + damp * (Q @ d_tangent)
                        u_damped /= np.linalg.norm(u_damped)
                        f_d = A @ u_damped
                        f2_d = f_d ** 2
                        f4_d = f2_d ** 2
                        n2_d = d_phi * np.sum(w_flat * f2_d)
                        n4_d = d_phi * np.sum(w_flat * f4_d)
                        if n2_d < 1e-12:
                            continue
                        R_d = (n4_d ** 0.25) / np.sqrt(n2_d)
                        if R_d < R_curr:
                            u = u_damped
                            best_c = u.copy()
                            best_val = R_d
                            break
                    else:
                        break
>>>>>>> REPLACE
