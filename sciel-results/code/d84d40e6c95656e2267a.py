"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    # Strategy: Derivative-free Powell's method for 2D potential minimization
    # 1. Define Phi(x) = 0.5 * ||F(x)||^2. We want to find x such that Phi(x) ~ 0.
    # 2. Powell's method is efficient in 2D, requiring only 1 function evaluation per step
    #    after the initial direction setup. It avoids Jacobian estimation costs.
    # 3. We run Powell's method from multiple diverse starting points to ensure global convergence.
    # 4. Once a very small Phi is found (||F|| < 1e-4), we switch to a damped Newton method
    #    using central differences for the Jacobian to polish the root to 1e-8 precision quickly.
    # 5. This hybrid approach balances robustness (Powell) and speed/precision (Newton).
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    call_count = [0]
    max_calls = 480  # Leave margin for safety
    
    def F_counted(x):
        call_count[0] += 1
        if call_count[0] > max_calls:
            raise Exception("Call budget exceeded")
        return F(x)
    
    def Phi(x):
        try:
            Fx = F_counted(x)
            if not np.all(np.isfinite(Fx)):
                return np.inf
            return 0.5 * np.dot(Fx, Fx)
        except Exception:
            return np.inf
            
    def jac_central(x, h=1e-5):
        # Central difference Jacobian
        J = np.zeros((2, 2))
        for j in range(2):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += h
            x_minus[j] -= h
            F_plus = F_counted(x_plus)
            F_minus = F_counted(x_minus)
            J[:, j] = (F_plus - F_minus) / (2 * h)
        return J

    def newton_polish(x_start, max_iter=20):
        # Damped Newton method for final polish
        x = np.array(x_start, dtype=float)
        for _ in range(max_iter):
            if call_count[0] > max_calls - 10:
                break
            if time.time() > deadline:
                break
            
            try:
                Fx = F_counted(x)
                if not np.all(np.isfinite(Fx)):
                    break
                norm_F = np.linalg.norm(Fx)
                if norm_F <= 1e-8:
                    return x, norm_F
            except Exception:
                break
            
            try:
                J = jac_central(x)
                if not np.all(np.isfinite(J)):
                    break
                # Solve J * dx = -Fx
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    # Fallback to least squares if singular
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                    
                if not np.all(np.isfinite(dx)):
                    break
                
                # Damping factor
                step_size = 1.0
                # Backtracking line search
                for _ in range(20):
                    x_trial = x + step_size * dx
                    try:
                        F_trial = F_counted(x_trial)
                        if not np.all(np.isfinite(F_trial)):
                            step_size *= 0.5
                            continue
                        norm_trial = np.linalg.norm(F_trial)
                        if norm_trial < norm_F:
                            x = x_trial
                            break
                        else:
                            step_size *= 0.5
                    except Exception:
                        step_size *= 0.5
                
                if norm_F <= 1e-8:
                    return x, norm_F
                    
            except Exception:
                break
                
        # Final check
        try:
            Fx = F_counted(x)
            if np.all(np.isfinite(Fx)):
                return x, np.linalg.norm(Fx)
        except Exception:
            pass
        return x, np.inf

    def powell_2d(x0, max_iter=50):
        # Powell's method for 2D
        x = np.array(x0, dtype=float)
        if not np.all(np.isfinite(x)):
            return x, np.inf
        
        # Initial directions
        d1 = np.array([1.0, 0.0])
        d2 = np.array([0.0, 1.0])
        
        f0 = Phi(x)
        if not np.isfinite(f0):
            return x, np.inf
            
        best_x = x.copy()
        best_f = f0
        
        for _ in range(max_iter):
            if call_count[0] > max_calls - 10:
                break
            if time.time() > deadline:
                break
                
            x1 = x.copy()
            f1 = f0
            
            # Minimize along d1
            try:
                x1, f1 = line_search(x1, d1, f1)
            except Exception:
                break
                
            if f1 < best_f:
                best_f = f1
                best_x = x1.copy()
                
            # Minimize along d2
            try:
                x2, f2 = line_search(x1, d2, f1)
            except Exception:
                break
                
            if f2 < best_f:
                best_f = f2
                best_x = x2.copy()
                
            # Check for convergence
            if np.linalg.norm(x2 - x) < 1e-10:
                break
                
            # Update direction
            # If significant improvement, update one of the directions
            if f0 - f2 > 2 * (f0 - f1) + 1e-12:
                # Improvement along d2 was larger
                # Replace d2 with (x2 - x)
                d2 = x2 - x
                if np.linalg.norm(d2) > 1e-12:
                    d2 = d2 / np.linalg.norm(d2)
            elif f0 - f2 > 2 * (f0 - f1) + 1e-12:
                # This condition is redundant in standard Powell, 
                # usually we check if f0 - f2 > 2*(f0 - f1) and f0 - f2 > 2*(f1 - f2)
                pass
                
            # Standard Powell update:
            # If f0 - f2 > 2 * (f0 - f1) + 1e-12, replace d1 with d2 (conceptually)
            # Actually, standard Powell replaces the direction with the largest improvement.
            # Let's implement the standard "replace worst direction" logic for 2D.
            
            # Let's restart the direction update logic properly.
            # We have x0, x1 (after d1), x2 (after d2).
            # If f2 < f0 - 2*(f0 - f1), it means the improvement in the new direction is significant.
            # For 2D, we can simply maintain the two directions.
            # If the path from x0 to x2 is better than the sum of steps, we update.
            
            # Simpler robust Powell for 2D:
            # Just keep the two directions orthogonal or update one if convergence is slow.
            # Given 2D, we can just use the two axes and rotate if needed.
            # However, the standard Powell method is quite effective.
            
            # Let's stick to the simple version:
            # If f2 < f0 - 2*(f0 - f1) - 1e-12:
            #    d1 = d2
            #    d2 = x2 - x0
            #    if norm(d2) > 0: d2 = d2/norm(d2)
            #    x0 = x2
            #    f0 = f2
            # else:
            #    x0 = x2
            #    f0 = f2
            
            # Let's implement this specific update rule.
            if f2 < f0 - 2 * (f0 - f1) - 1e-12:
                d1 = d2
                d2 = x2 - x0
                if np.linalg.norm(d2) > 1e-12:
                    d2 = d2 / np.linalg.norm(d2)
                else:
                    d2 = np.array([0.0, 1.0]) # Fallback
            else:
                # No direction update, just move
                pass
                
            x0 = x2
            f0 = f2
            
            if f0 < 1e-12:
                break
                
        return best_x, best_f

    def line_search(x, d, f_init):
        # Simple line search along direction d
        # Try a few step sizes
        step = 1.0
        best_x = x
        best_f = f_init
        
        # Adaptive step size based on previous success? 
        # For now, fixed set of steps is robust enough for 2D.
        for s in [1.0, 0.5, 0.25, 0.1, 0.05, 2.0, 4.0]:
            x_trial = x + s * d
            f_trial = Phi(x_trial)
            if np.isfinite(f_trial) and f_trial < best_f:
                best_f = f_trial
                best_x = x_trial
                
        return best_x, best_f

    x0 = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        x0 = np.array([0.0, 0.0])
        
    # Generate diverse starting points
    candidates = []
    candidates.append(x0.copy())
    candidates.append(np.array([0.0, 0.0]))
    
    scale = max(1.0, np.linalg.norm(x0))
    
    for s in [0.5, 2.0, 0.1, 5.0]:
        candidates.append(x0 * s)
        
    rng = np.random.default_rng(42)
    for r in [scale * 0.1, scale * 0.5, scale * 1.0, scale * 2.0, scale * 5.0]:
        for _ in range(3):
            u = rng.normal(size=2)
            u = u / np.linalg.norm(u)
            candidates.append(x0 + r * u)
            
    for r in [0.1, 1.0, 10.0]:
        for _ in range(2):
            u = rng.normal(size=2)
            u = u / np.linalg.norm(u)
            candidates.append(r * u)

    x_best = x0.copy()
    best_norm = np.inf
    
    # Initial evaluation
    try:
        F_best = F(x_best)
        if np.all(np.isfinite(F_best)):
            best_norm = np.linalg.norm(F_best)
    except Exception:
        pass
        
    for c in candidates:
        if time.time() > deadline:
            break
        if call_count[0] > max_calls - 50: # Leave room for Newton polish
            break
            
        try:
            # Run Powell's method
            x_powell, f_powell = powell_2d(c)
            
            if not np.all(np.isfinite(x_powell)):
                continue
                
            norm_powell = np.sqrt(2 * f_powell) if np.isfinite(f_powell) else np.inf
            
            if norm_powell < best_norm:
                best_norm = norm_powell
                x_best = x_powell.copy()
                
            # If Powell got us close, try Newton polish
            if norm_powell < 1e-4:
                x_newton, norm_newton = newton_polish(x_powell)
                if norm_newton < best_norm:
                    best_norm = norm_newton
                    x_best = x_newton.copy()
                    
                if norm_newton <= 1e-8:
                    return x_newton
                    
        except Exception:
            continue
            
    # Final check
    try:
        F_final = F(x_best)
        if np.all(np.isfinite(F_final)) and np.linalg.norm(F_final) <= 1e-8:
            return x_best
    except Exception:
        pass
        
    return x_best
# EVOLVE-BLOCK-END
