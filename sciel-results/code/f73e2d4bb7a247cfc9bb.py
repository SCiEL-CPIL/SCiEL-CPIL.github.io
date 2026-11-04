<<<<<<< SEARCH
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy
=======
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy
    
    # Also try a few random perturbations of the best config
    for trial in range(3):
        if time.time() - t0 > DEADLINE * 0.1:
            break
        perturbed_pos = best_positions + np.random.normal(0, 0.5, (N, 2))
        perturbed_ang = best_angles + np.random.normal(0, 10.0, N)
        side_t, phi_t, cx_t, cy_t = compute_min_outer_side_fast(perturbed_pos, perturbed_ang)
        if is_valid_configuration(perturbed_pos, perturbed_ang) and side_t < best_side:
            best_side = side_t
            best_positions = perturbed_pos.copy()
            best_angles = perturbed_ang.copy()
            best_phi = phi_t
            best_cx = cx_t
            best_cy = cy_t
>>>>>>> REPLACE

