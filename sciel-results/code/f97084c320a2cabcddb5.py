<<<<<<< SEARCH
    # Precompute cos/sin for all hexagons once per iteration for speed
    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
        total_count += 1
        
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 2.0)
        
        new_positions[idx, 0] += dx
        new_positions[idx, 1] += dy
        new_angles[idx] += da
        
        # Vectorized validity check: only check moved hexagon against others
        angles_rad_new = np.radians(new_angles)
        c_moved = math.cos(angles_rad_new[idx])
        s_moved = math.sin(angles_rad_new[idx])
        v1x = new_positions[idx, 0] + _ux * c_moved - _uy * s_moved
        v1y = new_positions[idx, 1] + _ux * s_moved + _uy * c_moved
        en1x = _enx * c_moved - _eny * s_moved
        en1y = _enx * s_moved + _eny * s_moved  # BUG FIX: was + _eny * s_moved
        
        valid = True
        for j in range(N):
            if j == idx:
                continue
            c_j = math.cos(angles_rad_new[j])
            s_j = math.sin(angles_rad_new[j])
            v2x = new_positions[j, 0] + _ux * c_j - _uy * s_j
            v2y = new_positions[j, 1] + _ux * s_j + _uy * c_j
            en2x = _enx * c_j - _eny * s_j
            en2y = _enx * s_j + _eny * c_j
            
            # Check all 12 separating axes
            for i in range(6):
                nx, ny = en1x[i], en1y[i]
                p1 = v1x * nx + v1y * ny
                p2 = v2x * nx + v2y * ny
                if p1.max() < p2.min() or p2.max() < p1.min():
                    break
            else:
                for i in range(6):
                    nx, ny = en2x[i], en2y[i]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    if p1.max() < p2.min() or p2.max() < p1.min():
                        break
                else:
                    valid = False
                    break
        
        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.002: step_size = 0.002
            continue
=======
    while time.time() - t0 < sa_time_budget and step_count < max_steps:
        step_count += 1
        total_count += 1
        
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 2.0)
        
        new_positions[idx, 0] += dx
        new_positions[idx, 1] += dy
        new_angles[idx] += da
        
        angles_rad_new = np.radians(new_angles)
        valid = True
        h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
        for j in range(N):
            if j == idx:
                continue
            h_other = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
            if not hexagons_disjoint_fast(h_moved, h_other):
                valid = False
                break
        
        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.002: step_size = 0.002
            continue
>>>>>>> REPLACE

