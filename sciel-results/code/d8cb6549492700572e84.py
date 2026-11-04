<<<<<<< SEARCH
            try:
                nm_res = minimize(nm_objective, nm_init, method='Nelder-Mead',
                                  options={'xatol': 1e-12, 'fatol': 1e-14,
                                           'maxiter': 5000, 'adaptive': True})
                if nm_res.fun < best_side:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = nm_res.x[3*i]
                        new_pos[i, 1] = nm_res.x[3*i+1]
                        new_pos[i, 2] = nm_res.x[3*i+2]
                    if is_valid_configuration(new_pos, new_ang):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
            except Exception:
                pass
=======
            try:
                nm_res = minimize(nm_objective, nm_init, method='Nelder-Mead',
                                  options={'xatol': 1e-13, 'fatol': 1e-15,
                                           'maxiter': 8000, 'adaptive': True})
                if nm_res.fun < best_side:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = nm_res.x[3*i]
                        new_pos[i, 1] = nm_res.x[3*i+1]
                        new_ang[i] = nm_res.x[3*i+2]
                    if is_valid_configuration(new_pos, new_ang):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos
                            best_angles = new_ang
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
            except Exception:
                pass
>>>>>>> REPLACE

