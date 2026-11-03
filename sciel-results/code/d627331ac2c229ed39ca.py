<<<<<<< SEARCH
    # Strategy: Run SA for 60% of the time, then Polish for the rest
    sa_time_budget = DEADLINE * 0.60
    
    # Single robust SA run with adaptive parameters
    current_positions = base_positions.copy()
    current_angles = base_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy

    T = 0.5
    T_min = 0.0001
    cooling = 0.99995
    step_size = 0.3
    step_decay = 0.99999
    
    step_count = 0
    max_steps = 200000
    accept_count = 0
    total_count = 0
=======
    # Strategy: Run SA for 70% of the time, then Polish for the rest
    # Use fast objective in SA to allow more iterations
    sa_time_budget = DEADLINE * 0.70
    
    # Single robust SA run with adaptive parameters
    current_positions = base_positions.copy()
    current_angles = base_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    current_fast_side = fast_outer_side_estimate(current_positions, current_angles)
    
    if current_side < best_side:
        best_side = current_side
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_phi = current_phi
        best_cx = current_cx
        best_cy = current_cy

    T = 0.5
    T_min = 0.0001
    cooling = 0.99995
    step_size = 0.3
    step_decay = 0.99999
    
    step_count = 0
    max_steps = 500000
    accept_count = 0
    total_count = 0
>>>>>>> REPLACE

