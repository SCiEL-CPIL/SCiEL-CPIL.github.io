    current_pts = best_pts.copy()
    current_score = best_score
    polish_step = 0.01
    no_improve_count = 0
    
    while time.time() < deadline:
        if no_improve_count > 5000:
             # Restart polish with a fresh random kick? Or just break?
             # Let's break if we are very stuck and time is tight.
             # Or reset step size.
             polish_step = 0.005
             no_improve_count = 0
             if time.time() > deadline - 5.0: break

        i = np.random.randint(N)
        delta = np.random.uniform(-polish_step, polish_step, 3)
        old_pos = current_pts[i].copy()
        current_pts[i] += delta
        current_pts[i] = np.clip(current_pts[i], -0.5, 0.5)
        
        new_score = get_score(current_pts)
        
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
            no_improve_count = 0
            polish_step = max(0.0001, polish_step * 0.999)
        else:
            # Revert
            current_pts[i] = old_pos
            no_improve_count += 1
            # Decay step size if we are failing
            if no_improve_count % 100 == 0:
                polish_step *= 0.9
                if polish_step < 0.0001:
                    polish_step = 0.001 # Reset to small non-zero
