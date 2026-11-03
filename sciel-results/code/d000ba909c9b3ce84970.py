<<<<<<< SEARCH
        # 3. Final high-precision coordinate descent
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 2.0:
            final_cd_time = min(remaining_time * 0.5, 10.0)
            final_side, final_pos, final_ang = polish_coordinate_descent(best_positions, best_angles, final_cd_time)
            if final_side < best_side:
                best_side = final_side
                best_positions = final_pos
                best_angles = final_ang
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        # 4. Ultra-fine final polish with very small steps
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 1.5:
            def ultra_fine_polish(pos, ang, time_limit):
                current_side, _, _, _ = compute_min_outer_side(pos, ang)
                best_side_uf = current_side
                best_pos_uf = pos.copy()
                best_ang_uf = ang.copy()
                
                t_start = time.time()
                step = 0.005
                min_step = 0.0001
                
                while time.time() - t_start < time_limit and step > min_step:
                    improved = False
                    for idx in range(N):
                        for dim in range(3):
                            for direction in [1, -1]:
                                new_pos = best_pos_uf.copy()
                                new_ang = best_ang_uf.copy()
                                
                                if dim == 0:
                                    new_pos[idx, 0] += direction * step
                                elif dim == 1:
                                    new_pos[idx, 1] += direction * step
                                else:
                                    new_ang[idx] += direction * step
                                
                                angles_rad_new = np.radians(new_ang)
                                valid = True
                                h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                                for j in range(N):
                                    if j == idx: continue
                                    h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                    if not hexagons_disjoint_fast(h_moved, h_other):
                                        valid = False
                                        break
                                
                                if not valid:
                                    continue
                                
                                new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)
                                
                                if new_side < best_side_uf - 1e-12:
                                    best_pos_uf = new_pos
                                    best_ang_uf = new_ang
                                    best_side_uf = new_side
                                    improved = True
                                    break
                            
                            if time.time() - t_start > time_limit:
                                return best_side_uf, best_pos_uf, best_ang_uf
                        else:
                            continue
                        break
                    else:
                        step *= 0.7
                        continue
                    break
                
                return best_side_uf, best_pos_uf, best_ang_uf
            
            uf_time = min(remaining_time * 0.4, 6.0)
            uf_side, uf_pos, uf_ang = ultra_fine_polish(best_positions, best_angles, uf_time)
            if uf_side < best_side:
                best_side = uf_side
                best_positions = uf_pos
                best_angles = uf_ang
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
=======
        # 3. Final high-precision coordinate descent
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 2.0:
            final_cd_time = min(remaining_time * 0.5, 10.0)
            final_side, final_pos, final_ang = polish_coordinate_descent(best_positions, best_angles, final_cd_time)
            if final_side < best_side:
                best_side = final_side
                best_positions = final_pos
                best_angles = final_ang
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
        
        # 4. Joint Nelder-Mead on all 36 parameters (positions + angles)
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 3.0:
            def joint_objective(params):
                pos = np.zeros((N, 2))
                ang = np.zeros(N)
                for i in range(N):
                    pos[i, 0] = params[3*i]
                    pos[i, 1] = params[3*i+1]
                    ang[i] = params[3*i+2]
                angles_rad = np.radians(ang)
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                        h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                        if not hexagons_disjoint_fast(h1, h2):
                            return 1e6
                s, _, _, _ = compute_min_outer_side(pos, ang)
                return s

            joint_init = np.zeros(3*N)
            for i in range(N):
                joint_init[3*i] = best_positions[i, 0]
                joint_init[3*i+1] = best_positions[i, 1]
                joint_init[3*i+2] = best_angles[i]
            try:
                res = minimize(joint_objective, joint_init, method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < 1e5:
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
            except Exception:
                pass

        # 5. Ultra-fine final polish with very small steps
        remaining_time = DEADLINE - (time.time() - t0)
        if remaining_time > 1.5:
            def ultra_fine_polish(pos, ang, time_limit):
                current_side, _, _, _ = compute_min_outer_side(pos, ang)
                best_side_uf = current_side
                best_pos_uf = pos.copy()
                best_ang_uf = ang.copy()
                
                t_start = time.time()
                step = 0.005
                min_step = 0.0001
                
                while time.time() - t_start < time_limit and step > min_step:
                    improved = False
                    for idx in range(N):
                        for dim in range(3):
                            for direction in [1, -1]:
                                new_pos = best_pos_uf.copy()
                                new_ang = best_ang_uf.copy()
                                
                                if dim == 0:
                                    new_pos[idx, 0] += direction * step
                                elif dim == 1:
                                    new_pos[idx, 1] += direction * step
                                else:
                                    new_ang[idx] += direction * step
                                
                                angles_rad_new = np.radians(new_ang)
                                valid = True
                                h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                                for j in range(N):
                                    if j == idx: continue
                                    h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                                    if not hexagons_disjoint_fast(h_moved, h_other):
                                        valid = False
                                        break
                                
                                if not valid:
                                    continue
                                
                                new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)
                                
                                if new_side < best_side_uf - 1e-12:
                                    best_pos_uf = new_pos
                                    best_ang_uf = new_ang
                                    best_side_uf = new_side
                                    improved = True
                                    break
                            
                            if time.time() - t_start > time_limit:
                                return best_side_uf, best_pos_uf, best_ang_uf
                        else:
                            continue
                        break
                    else:
                        step *= 0.7
                        continue
                    break
                
                return best_side_uf, best_pos_uf, best_ang_uf
            
            uf_time = min(remaining_time * 0.4, 6.0)
            uf_side, uf_pos, uf_ang = ultra_fine_polish(best_positions, best_angles, uf_time)
            if uf_side < best_side:
                best_side = uf_side
                best_positions = uf_pos
                best_angles = uf_ang
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
>>>>>>> REPLACE
