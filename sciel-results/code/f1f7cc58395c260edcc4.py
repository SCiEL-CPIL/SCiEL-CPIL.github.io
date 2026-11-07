# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with adaptive line search and time guard.
    """
    x = np.array(x0, dtype=float).reshape(2)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_up():
        return time.time() > deadline
    
    def norm(v):
        return np.linalg.norm(v)
    
    # Initial evaluation
    Fx = np.array(F(x), dtype=float)
    calls = 1
    best_x = x.copy()
    best_norm = norm(Fx)
    
    if best_norm <= 1e-8:
        return best_x
    
    # Initialize Jacobian estimate using finite differences
    h = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_pert = np.array(F(x + e), dtype=float)
        calls += 1
        J[:, j] = (Fx_pert - Fx) / h
    
    # Broyden's method
    max_iter = 200
    for it in range(max_iter):
        if time_up():
            break
        
        # Check convergence
        if best_norm <= 1e-8:
            break
        
        # Solve J dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break
        
        # Adaptive line search
        alpha = 1.0
        max_ls = 20
        for ls in range(max_ls):
            if time_up():
                break
            
            x_new = x + alpha * dx
            Fx_new = np.array(F(x_new), dtype=float)
            calls += 1
            
            if not np.all(np.isfinite(Fx_new)):
                alpha *= 0.5
                continue
            
            Fx_new_norm = norm(Fx_new)
            
            if Fx_new_norm <= 1e-8:
                x = x_new
                Fx = Fx_new
                best_x = x.copy()
                best_norm = Fx_new_norm
                break
            
            # Armijo-like condition: ensure sufficient decrease
            if Fx_new_norm < best_norm:
                x = x_new
                Fx = Fx_new
                best_x = x.copy()
                best_norm = Fx_new_norm
                break
            elif ls == max_ls - 1:
                # Accept the best point found in line search
                x = x_new
                Fx = Fx_new
                if Fx_new_norm < best_norm:
                    best_x = x.copy()
                    best_norm = Fx_new_norm
                break
            else:
                alpha *= 0.5
        
        if best_norm <= 1e-8:
            break
        
        # Update Jacobian using Broyden's formula
        if calls < 490 and not time_up():
            s = x - (x - alpha * dx)  # This is just alpha*dx, but let's be careful
            # Actually, s should be the step taken
            # Let's recompute: the step was from old_x to new_x
            # We need to track old_x properly
            
            # Let me restructure to track old_x
            pass
    
    # The above has issues with tracking. Let me rewrite more carefully.
    
    # Restart with cleaner implementation
    x = np.array(x0, dtype=float).reshape(2)
    Fx = np.array(F(x), dtype=float)
    calls = 1
    best_x = x.copy()
    best_norm = norm(Fx)
    
    if best_norm <= 1e-8:
        return best_x
    
    # Initialize Jacobian
    h = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_pert = np.array(F(x + e), dtype=float)
        calls += 1
        J[:, j] = (Fx_pert - Fx) / h
    
    max_iter = 300
    for it in range(max_iter):
        if time_up() or calls >= 495:
            break
        
        if best_norm <= 1e-8:
            break
        
        # Solve for step
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break
        
        # If step is too large, scale it down
        dx_norm = norm(dx)
        if dx_norm > 10.0:
            dx = dx / dx_norm * 10.0
        
        # Line search
        alpha = 1.0
        found = False
        for ls in range(30):
            if time_up() or calls >= 495:
                break
            
            x_trial = x + alpha * dx
            Fx_trial = np.array(F(x_trial), dtype=float)
            calls += 1
            
            if not np.all(np.isfinite(Fx_trial)):
                alpha *= 0.5
                continue
            
            Fx_trial_norm = norm(Fx_trial)
            
            # Accept if we made progress or if this is a good point
            if Fx_trial_norm < best_norm * 0.95 or Fx_trial_norm <= 1e-8:
                x = x_trial
                Fx = Fx_trial
                if Fx_trial_norm < best_norm:
                    best_x = x.copy()
                    best_norm = Fx_trial_norm
                found = True
                break
            elif ls == 29:
                # Take the trial step anyway if we haven't found a better one
                if Fx_trial_norm < best_norm:
                    x = x_trial
                    Fx = Fx_trial
                    best_x = x.copy()
                    best_norm = Fx_trial_norm
                found = True
                break
            else:
                alpha *= 0.5
        
        if not found:
            break
        
        if best_norm <= 1e-8:
            break
        
        # Broyden update
        if calls < 490 and not time_up():
            s = x - (x - alpha * dx)  # This is wrong, let's track properly
            # We need the actual step taken. Let me fix this.
            pass
    
    # Let me do a clean rewrite with proper tracking
    x = np.array(x0, dtype=float).reshape(2)
    Fx = np.array(F(x), dtype=float)
    calls = 1
    best_x = x.copy()
    best_norm = norm(Fx)
    
    if best_norm <= 1e-8:
        return best_x
    
    # Initialize Jacobian
    h = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_pert = np.array(F(x + e), dtype=float)
        calls += 1
        J[:, j] = (Fx_pert - Fx) / h
    
    max_iter = 300
    for it in range(max_iter):
        if time_up() or calls >= 495:
            break
        
        if best_norm <= 1e-8:
            break
        
        # Solve for step
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break
        
        # Scale step if too large
        dx_norm = norm(dx)
        if dx_norm > 10.0:
            dx = dx / dx_norm * 10.0
        
        # Line search with proper tracking
        alpha = 1.0
        x_old = x.copy()
        Fx_old = Fx.copy()
        accepted = False
        
        for ls in range(30):
            if time_up() or calls >= 495:
                break
            
            x_trial = x_old + alpha * dx
            Fx_trial = np.array(F(x_trial), dtype=float)
            calls += 1
            
            if not np.all(np.isfinite(Fx_trial)):
                alpha *= 0.5
                continue
            
            Fx_trial_norm = norm(Fx_trial)
            
            if Fx_trial_norm <= 1e-8 or Fx_trial_norm < best_norm * 0.95:
                x = x_trial
                Fx = Fx_trial
                if Fx_trial_norm < best_norm:
                    best_x = x.copy()
                    best_norm = Fx_trial_norm
                accepted = True
                break
            elif ls == 29:
                if Fx_trial_norm < best_norm:
                    x = x_trial
                    Fx = Fx_trial
                    best_x = x.copy()
                    best_norm = Fx_trial_norm
                accepted = True
                break
            else:
                alpha *= 0.5
        
        if not accepted:
            break
        
        if best_norm <= 1e-8:
            break
        
        # Broyden update
        if calls < 490 and not time_up():
            s = x - x_old
            y = Fx - Fx_old
            s_norm = norm(s)
            y_norm = norm(y)
            
            if s_norm > 1e-15 and y_norm > 1e-15:
                # Broyden's good rank-1 update
                J = J + np.outer(y - J @ s, s) / (s @ s)
    
    return best_x
# EVOLVE-BLOCK-END
