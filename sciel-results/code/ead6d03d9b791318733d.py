"""Root finder for 3 fixed two-dimensional systems using robust Broyden with Powell refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using robust Broyden's method
    with backtracking, and Powell's method as a derivative-free fallback.
    """
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin before 120s limit

    def time_ok():
        return time.time() < deadline

    x0 = np.array(x0, dtype=float)
    call_count = 0
    MAX_CALLS = 490  # Stay well under 500

    def F_counted(x):
        nonlocal call_count
        call_count += 1
        return F(x)

    best_x = x0.copy()
    try:
        best_norm = np.linalg.norm(F_counted(x0))
    except:
        best_norm = np.inf

    if best_norm <= 1e-8:
        return x0

    def update_best(x):
        nonlocal best_x, best_norm
        if not np.all(np.isfinite(x)):
            return
        try:
            fx = np.asarray(F_counted(x), dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return
            n = np.linalg.norm(fx)
            if np.isfinite(n) and n < best_norm:
                best_norm = n
                best_x = x.copy()
        except:
            pass

    def broyden(x_init, J_init=None, max_iter=25):
        """
        Broyden's method with backtracking line search.
        Returns (success, J_final) where J_final is the last Jacobian estimate.
        """
        if not time_ok() or call_count >= MAX_CALLS - 10:
            return False, None

        x = np.array(x_init, dtype=float)
        h = 1e-5

        try:
            Fx = np.asarray(F_counted(x), dtype=float)
            if not np.all(np.isfinite(Fx)):
                return False, None
            n = np.linalg.norm(Fx)
            update_best(x)
            if n <= 1e-8:
                return True, None
        except:
            return False, None

        # Initialize Jacobian
        if J_init is not None:
            J = J_init.copy()
        else:
            try:
                Fx1 = np.asarray(F_counted(x + [h, 0]), dtype=float)
                Fx2 = np.asarray(F_counted(x + [0, h]), dtype=float)
                if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
                    return False, None
                J = np.column_stack([(Fx1 - Fx) / h, (Fx2 - Fx) / h])
                # Regularize if ill-conditioned
                try:
                    if np.linalg.cond(J) > 1e10:
                        J = J + np.eye(2) * 1e-6
                except:
                    pass
            except:
                return False, None

        for _ in range(max_iter):
            if not time_ok() or call_count >= MAX_CALLS - 5:
                return False, J

            try:
                try:
                    s = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    s = np.linalg.lstsq(J, -Fx, rcond=None)[0]

                if not np.all(np.isfinite(s)):
                    return False, J

                alpha = 1.0
                success = False
                x_old = x.copy()
                Fx_old = Fx.copy()

                for _ in range(12):
                    x_trial = x + alpha * s
                    if not np.all(np.isfinite(x_trial)):
                        alpha *= 0.5
                        continue
                    Fx_trial = np.asarray(F_counted(x_trial), dtype=float)
                    if not np.all(np.isfinite(Fx_trial)):
                        alpha *= 0.5
                        continue
                    n_trial = np.linalg.norm(Fx_trial)

                    if n_trial < n:
                        x = x_trial
                        Fx = Fx_trial
                        n = n_trial
                        success = True
                        update_best(x)
                        break
                    alpha *= 0.5

                if not success:
                    return False, J

                if n <= 1e-8:
                    return True, J

                # Broyden rank-1 update
                s_actual = x - x_old
                y = Fx - Fx_old
                denom = np.dot(s_actual, s_actual)

                if denom > 1e-14:
                    J = J + np.outer(y - J @ s_actual, s_actual) / denom
                else:
                    return False, J

            except:
                return False, J

        return False, J

    def powell(x_init, max_iter=30):
        """
        Powell's method for minimizing Phi(x) = 0.5*||F(x)||^2.
        This is a derivative-free method that alternates between
        directional searches along a set of directions.
        """
        if not time_ok() or call_count >= MAX_CALLS - 30:
            return False

        x0 = np.array(x_init, dtype=float)
        
        try:
            f0 = 0.5 * np.dot(np.asarray(F_counted(x0), dtype=float), np.asarray(F_counted(x0), dtype=float))
        except:
            return False
        update_best(x0)
        if f0 <= 1e-16:
            return True

        # Initialize directions: standard basis vectors
        directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        
        # Current point and its function value
        x_curr = x0.copy()
        f_curr = f0
        
        for iteration in range(max_iter):
            if not time_ok() or call_count >= MAX_CALLS - 10:
                break
            
            x_old = x_curr.copy()
            f_old = f_curr
            max_delta = 0.0
            max_dir_idx = 0
            
            # Search along each direction
            for i, d in enumerate(directions):
                if not time_ok() or call_count >= MAX_CALLS - 10:
                    break
                
                # Find best point along direction d from x_curr
                best_x_d = x_curr.copy()
                best_f_d = f_curr
                best_step = 0.0
                
                # Try several step sizes
                for step in [0.1, 0.01, -0.1, -0.01, 0.05, -0.05, 0.2, -0.2]:
                    x_trial = x_curr + step * d
                    if not np.all(np.isfinite(x_trial)):
                        continue
                    
                    try:
                        Fx_trial = np.asarray(F_counted(x_trial), dtype=float)
                        if not np.all(np.isfinite(Fx_trial)):
                            continue
                        f_trial = 0.5 * np.dot(Fx_trial, Fx_trial)
                        if f_trial < best_f_d:
                            best_f_d = f_trial
                            best_x_d = x_trial
                            best_step = step
                    except:
                        continue
                
                # Update current point
                if best_f_d < f_curr:
                    x_curr = best_x_d
                    f_curr = best_f_d
                    delta = f_old - f_curr
                    if delta > max_delta:
                        max_delta = delta
                        max_dir_idx = i
                update_best(x_curr)
            
            if f_curr <= 1e-16:
                return True
            
            # If no significant improvement, stop
            if max_delta < 1e-12:
                break
            
            # Replace the direction with the largest improvement
            # with the new direction from x_old to x_curr
            new_dir = x_curr - x_old
            if np.linalg.norm(new_dir) > 1e-14:
                directions[max_dir_idx] = new_dir / np.linalg.norm(new_dir)
            
            # Search along the new combined direction
            if not time_ok() or call_count >= MAX_CALLS - 10:
                break
            
            x_temp = x_curr.copy()
            f_temp = f_curr
            
            # Try steps along the new direction
            for step in [0.1, 0.01, -0.1, -0.01]:
                x_trial = x_temp + step * directions[max_dir_idx]
                if not np.all(np.isfinite(x_trial)):
                    continue
                try:
                    Fx_trial = np.asarray(F_counted(x_trial), dtype=float)
                    if not np.all(np.isfinite(Fx_trial)):
                        continue
                    f_trial = 0.5 * np.dot(Fx_trial, Fx_trial)
                    if f_trial < f_temp:
                        f_temp = f_trial
                        x_temp = x_trial
                except:
                    continue
            
            if f_temp < f_curr:
                x_curr = x_temp
                f_curr = f_temp
                update_best(x_curr)
                if f_curr <= 1e-16:
                    return True
        
        return f_curr <= 1e-16

    # Main algorithm: try Broyden from multiple starting points
    for restart in range(4):
        if not time_ok() or call_count >= MAX_CALLS - 20:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            angle = restart * np.pi / 2
            radius = 0.5 * (np.linalg.norm(x0) + 1.0) * restart
            x_start = x0 + radius * np.array([np.cos(angle), np.sin(angle)])
        
        success, J_final = broyden(x_start, J_init=None, max_iter=20)
        
        if success:
            return best_x
        
        # If Broyden failed, try Powell from the best point found
        if not time_ok() or call_count >= MAX_CALLS - 30:
            break
        
        if powell(best_x, max_iter=20):
            return best_x
    
    # Final attempt: try Powell from x0 if we haven't already
    if not time_ok() or call_count >= MAX_CALLS - 30:
        pass
    else:
        if powell(x0, max_iter=20):
            return best_x
    
    return best_x
# EVOLVE-BLOCK-END
