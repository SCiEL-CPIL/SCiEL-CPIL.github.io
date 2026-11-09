"""Root finder for 3 fixed two-dimensional systems using Broyden's method with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy: Use Broyden's method (1 F call/iter after setup) with multiple restarts.
    Broyden updates Jacobian approximation, so each iteration costs 1 F(x) call.
    For 2D, initial Jacobian needs 3 calls (F(x0), F(x0+e1), F(x0+e2)).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
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
    best_norm = np.inf
    
    def update_best(x):
        nonlocal best_x, best_norm
        try:
            x = np.asarray(x, dtype=float)
            if x.shape != (2,) or not np.all(np.isfinite(x)):
                return
            fx = F_counted(x)
            fx = np.asarray(fx, dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return
            n = np.linalg.norm(fx)
            if np.isfinite(n) and n < best_norm:
                best_norm = n
                best_x = x.copy()
        except:
            pass
    
    def broyden(x_init, max_iter=40):
        """Broyden's method for 2D systems with adaptive damping. Returns (x, success)."""
        nonlocal call_count, best_norm, best_x
        if call_count >= MAX_CALLS or not time_ok():
            return x_init, False
        
        x = np.array(x_init, dtype=float)
        h = 1e-7
        
        # Initial Jacobian estimate: 3 calls
        Fx0 = np.asarray(F_counted(x), dtype=float)
        if not np.all(np.isfinite(Fx0)):
            return x, False
        n0 = np.linalg.norm(Fx0)
        if n0 <= 1e-10:
            update_best(x)
            return x, True
        
        # Estimate Jacobian
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        Fx1 = np.asarray(F_counted(x + e1), dtype=float)
        Fx2 = np.asarray(F_counted(x + e2), dtype=float)
        
        if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
            return x, False
        
        J = np.column_stack([(Fx1 - Fx0) / h, (Fx2 - Fx0) / h])
        
        # Check if Jacobian is singular
        detJ = np.linalg.det(J)
        if abs(detJ) < 1e-15:
            J += np.eye(2) * 1e-8
        
        for _ in range(max_iter):
            if call_count >= MAX_CALLS or not time_ok():
                break
            
            # Broyden step: solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx0)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -Fx0, rcond=None)[0]
            
            # Adaptive damping: start with full step, halve if needed
            step_norm = np.linalg.norm(dx)
            if step_norm > 10.0:
                dx = dx * (10.0 / step_norm)
                step_norm = 10.0
            
            # Try full step first, then halve up to 4 times
            success_step = False
            for damp in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                x_new = x + damp * dx
                if not np.all(np.isfinite(x_new)):
                    continue
                
                Fx_new = np.asarray(F_counted(x_new), dtype=float)
                if not np.all(np.isfinite(Fx_new)):
                    continue
                
                n_new = np.linalg.norm(Fx_new)
                update_best(x_new)
                
                if n_new <= 1e-10:
                    return x_new, True
                
                # Accept if we made progress (even small)
                if n_new < n0 * 1.0:
                    success_step = True
                    # Broyden update
                    s = x_new - x
                    y = Fx_new - Fx0
                    denom = np.dot(s, s)
                    if denom > 1e-30:
                        J = J + np.outer(y - J @ s, s) / denom
                    
                    x = x_new
                    Fx0 = Fx_new
                    n0 = n_new
                    break
            
            if not success_step:
                # No progress even with heavy damping; try a different direction
                # Perturb the Jacobian slightly and continue
                J += np.eye(2) * 1e-10
                break
        
        return x, n0 <= 1e-10
    
    def polish(x_init, max_iter=15):
        """Final high-precision polish pass using Newton with finite-diff Jacobian."""
        nonlocal call_count, best_norm, best_x
        if call_count >= MAX_CALLS or not time_ok():
            return x_init, False
        
        x = np.array(x_init, dtype=float)
        
        # Get current F(x)
        Fx = np.asarray(F_counted(x), dtype=float)
        if not np.all(np.isfinite(Fx)):
            return x, False
        n = np.linalg.norm(Fx)
        if n <= 1e-10:
            return x, True
        
        for _ in range(max_iter):
            if call_count >= MAX_CALLS or not time_ok():
                break
            
            # Adaptive step size for finite differences
            h = max(1e-8, 1e-4 * n ** 0.5)
            
            # Compute finite-difference Jacobian (3 calls)
            e1 = np.array([h, 0.0])
            e2 = np.array([0.0, h])
            Fx1 = np.asarray(F_counted(x + e1), dtype=float)
            Fx2 = np.asarray(F_counted(x + e2), dtype=float)
            
            if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
                break
            
            J = np.column_stack([(Fx1 - Fx) / h, (Fx2 - Fx) / h])
            
            detJ = np.linalg.det(J)
            if abs(detJ) < 1e-15:
                J += np.eye(2) * 1e-8
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            
            step_norm = np.linalg.norm(dx)
            if step_norm > 2.0:
                dx = dx * (2.0 / step_norm)
            
            # Adaptive damping for polish
            success_step = False
            for damp in [1.0, 0.5, 0.25, 0.125]:
                x_new = x + damp * dx
                if not np.all(np.isfinite(x_new)):
                    continue
                
                Fx_new = np.asarray(F_counted(x_new), dtype=float)
                if not np.all(np.isfinite(Fx_new)):
                    continue
                
                n_new = np.linalg.norm(Fx_new)
                update_best(x_new)
                
                if n_new <= 1e-10:
                    return x_new, True
                
                if n_new < n * 0.9:
                    success_step = True
                    x = x_new
                    Fx = Fx_new
                    n = n_new
                    break
            
            if not success_step:
                break
        
        return x, n <= 1e-10
    
    # First, evaluate x0
    update_best(x0)
    if best_norm <= 1e-8:
        return best_x
    
    # Try Broyden from x0
    x_result, success = broyden(x0)
    if success and best_norm <= 1e-8:
        # Polish for higher precision
        polish(best_x, max_iter=8)
        if best_norm <= 1e-8:
            return best_x
    
    # Generate alternative starting points
    scale = max(1.0, np.linalg.norm(x0))
    candidates = []
    
    # Perturbations at various scales and directions
    for s in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1), (-1,0), (0,-1), (-1,-1), (-1,1)]:
            candidates.append(x0 + s * np.array([dx, dy]))
    
    # Scaled versions
    for factor in [0.1, 0.5, 2.0, 5.0]:
        candidates.append(x0 * factor)
    
    # Origin and axis points
    candidates.append(np.array([0.0, 0.0]))
    candidates.append(np.array([1.0, 0.0]))
    candidates.append(np.array([0.0, 1.0]))
    candidates.append(np.array([1.0, 1.0]))
    candidates.append(np.array([-1.0, 0.0]))
    candidates.append(np.array([0.0, -1.0]))
    candidates.append(np.array([-1.0, -1.0]))
    
    # Try Broyden from each candidate
    for x_init in candidates:
        if call_count >= MAX_CALLS or not time_ok():
            break
        if not np.all(np.isfinite(x_init)):
            continue
        x_result, success = broyden(x_init)
        if success and best_norm <= 1e-8:
            # Polish for higher precision
            polish(best_x, max_iter=8)
            if best_norm <= 1e-8:
                return best_x
    
    # If still not good enough, try from best_x
    if best_norm > 1e-8 and call_count < MAX_CALLS and time_ok():
        x_result, success = broyden(best_x)
        if success and best_norm <= 1e-8:
            polish(best_x, max_iter=8)
            if best_norm <= 1e-8:
                return best_x
    
    # Final polish attempt if we have budget
    if best_norm > 1e-8 and call_count < MAX_CALLS and time_ok():
        polish(best_x, max_iter=10)
    
    return best_x if np.all(np.isfinite(best_x)) else x0

# EVOLVE-BLOCK-END
