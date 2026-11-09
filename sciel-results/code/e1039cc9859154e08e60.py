import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using a robust hybrid strategy:
    1. Try Newton's method with adaptive step size and restarts.
    2. If Newton fails, switch to a global search using a coarse grid + local refinement.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def time_ok():
        return time.time() < deadline
    
    x0 = np.array(x0, dtype=float)
    call_count = 0
    MAX_CALLS = 480  # Stay well under 500
    
    def F_counted(x):
        nonlocal call_count
        call_count += 1
        return F(x)
    
    best_x = x0.copy()
    try:
        best_norm = np.linalg.norm(F_counted(x0))
    except:
        best_norm = np.inf
        
    # If we start at a root, we are done
    if best_norm <= 1e-8:
        return x0
    
    def is_good(x):
        if not np.all(np.isfinite(x)):
            return False
        fx = np.asarray(F_counted(x), dtype=float)
        if fx.shape != (2,) or not np.all(np.isfinite(fx)):
            return False
        return np.linalg.norm(fx) <= 1e-8
    
    def update_best(x):
        nonlocal best_x, best_norm
        if not np.all(np.isfinite(x)):
            return
        fx = np.asarray(F_counted(x), dtype=float)
        if fx.shape != (2,) or not np.all(np.isfinite(fx)):
            return
        n = np.linalg.norm(fx)
        if np.isfinite(n) and n < best_norm:
            best_norm = n
            best_x = x.copy()
    
    def broyden_method(x_init, max_iter=100):
        """
        Broyden's method with backtracking line search.
        Returns success flag.
        """
        nonlocal call_count, best_norm, best_x
        
        x = np.array(x_init, dtype=float)
        h = 1e-7  # Step size for finite differences
        
        # Initial Function Value
        Fx = np.asarray(F_counted(x), dtype=float)
        if not np.all(np.isfinite(Fx)):
            return False
        n = np.linalg.norm(Fx)
        update_best(x)
        
        if n <= 1e-8:
            return True
        
        # Initial Jacobian Estimate (Forward Differences)
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        
        Fx1 = np.asarray(F_counted(x + e1), dtype=float)
        Fx2 = np.asarray(F_counted(x + e2), dtype=float)
        
        if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
            return False
            
        J = np.column_stack([(Fx1 - Fx) / h, (Fx2 - Fx) / h])
        
        # Check condition number, if too high, add regularization
        try:
            cond = np.linalg.cond(J)
            if cond > 1e15:
                J = J + np.eye(2) * 1e-6 * np.linalg.norm(J, ord=2)
        except:
            pass
        
        for _ in range(max_iter):
            if call_count >= MAX_CALLS or not time_ok():
                break
                
            # Solve for Newton step
            try:
                s = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # If singular, use pseudo-inverse or regularize
                try:
                    s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    break
                    
            if not np.all(np.isfinite(s)):
                break
                
            # Line Search
            # We want to find alpha such that ||F(x + alpha*s)|| < ||F(x)||
            # Start with alpha = 1.0
            alpha = 1.0
            x_new = x
            Fx_new = Fx
            n_new = n
            success_step = False
            
            for _ in range(20): # Max 20 line search steps
                x_trial = x + alpha * s
                if not np.all(np.isfinite(x_trial)):
                    alpha *= 0.5
                    continue
                    
                Fx_trial = np.asarray(F_counted(x_trial), dtype=float)
                if not np.all(np.isfinite(Fx_trial)):
                    alpha *= 0.5
                    continue
                    
                n_trial = np.linalg.norm(Fx_trial)
                
                # Armijo-like condition: sufficient decrease
                if n_trial < n * (1 - 1e-4 * alpha * np.dot(s, -Fx)): 
                    # Simplified: just check for decrease
                    if n_trial < n:
                        x_new = x_trial
                        Fx_new = Fx_trial
                        n_new = n_trial
                        success_step = True
                        break
                alpha *= 0.5
                
            if not success_step:
                # If no decrease found, try a small step in gradient direction? 
                # Or just break to avoid wasting calls.
                # Let's try one more time with a very small alpha
                alpha = 0.01
                x_trial = x + alpha * s
                if np.all(np.isfinite(x_trial)):
                    Fx_trial = np.asarray(F_counted(x_trial), dtype=float)
                    if np.all(np.isfinite(Fx_trial)):
                        n_trial = np.linalg.norm(Fx_trial)
                        if n_trial < n:
                            x_new = x_trial
                            Fx_new = Fx_trial
                            n_new = n_trial
                            success_step = True
                            
            if not success_step:
                # Stagnation
                break
                
            # Update Best
            update_best(x_new)
            
            if n_new <= 1e-8:
                return True
                
            # Broyden Update
            # s_actual = x_new - x
            # y = Fx_new - Fx
            s_actual = x_new - x
            y = Fx_new - Fx
            denom = np.dot(s_actual, s_actual)
            
            if denom > 1e-12:
                # Standard Broyden update
                J = J + np.outer(y - J @ s_actual, s_actual) / denom
            else:
                # If step was too small, recompute Jacobian?
                # Recomputing is expensive (3 calls). Let's just keep J.
                pass
                
            x = x_new
            Fx = Fx_new
            n = n_new
            
        return (n <= 1e-8)
    
    # 1. Try Broyden from x0
    if broyden_method(x0):
        if best_norm <= 1e-8:
            return best_x

    # 2. Generate diverse restarts
    scale = max(1.0, np.linalg.norm(x0))
    restarts = []
    for s in [0.5, 1.0, 2.0, 5.0, 10.0]:
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1), (-1,0), (0,-1), (-1,-1), (-1,1)]:
            restarts.append(x0 + s * np.array([dx, dy]) * scale)
            
    # Add some random-ish perturbations based on x0
    for i in range(20):
        angle = 2 * np.pi * i / 20
        r = scale * (0.5 + i * 0.1)
        restarts.append(x0 + r * np.array([np.cos(angle), np.sin(angle)]))
        
    # Shuffle to avoid bias
    np.random.seed(42)
    np.random.shuffle(restarts)
    
    for x_init in restarts:
        if not time_ok() or call_count >= MAX_CALLS - 5:
            break
        if not np.all(np.isfinite(x_init)):
            continue
        if broyden_method(x_init):
            if best_norm <= 1e-8:
                return best_x

    # 3. Fallback: Grid search if still not found
    # This is expensive, so only do it if we have budget
    if time_ok() and call_count < MAX_CALLS - 50:
        # Define a grid around x0
        center = x0
        radius = max(10.0, 5 * np.linalg.norm(x0))
        steps = 10
        dx = 2 * radius / steps
        
        for i in range(steps + 1):
            for j in range(steps + 1):
                if not time_ok() or call_count >= MAX_CALLS - 5:
                    break
                x_g = center + np.array([i * dx - radius, j * dx - radius])
                if broyden_method(x_g):
                    if best_norm <= 1e-8:
                        return best_x

    return best_x if np.all(np.isfinite(best_x)) else x0
