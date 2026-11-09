"""Root finder for 3 fixed two-dimensional systems using Broyden's method with call optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Broyden's method (1 F call/iter after setup) with multiple restarts.
    2. Initial Jacobian via central differences (3 calls) for accuracy.
    3. Polish phase with Newton (central differences for precision).
    4. Track calls carefully to stay under 500.
    5. Time-guarded with safety margin.
    6. Adaptive step control, line search, and stagnation detection.
    7. Jacobian reuse across nearby restarts.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def time_ok():
        return time.time() < deadline
    
    x0 = np.array(x0, dtype=float)
    call_count = 0
    MAX_CALLS = 490
    
    def F_counted(x):
        nonlocal call_count
        call_count += 1
        return F(x)
    
    best_x = x0.copy()
    best_norm = np.inf
    cached_J = None  # Reuse Jacobian across restarts
    
    def update_best(x, fx=None):
        nonlocal best_x, best_norm
        try:
            x = np.asarray(x, dtype=float)
            if x.shape != (2,) or not np.all(np.isfinite(x)):
                return
            if fx is None:
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
    
    def compute_jacobian(x, fx, use_central=True):
        """Compute Jacobian at x. Uses central differences if use_central and budget allows."""
        nonlocal call_count
        h = 1e-6 * max(1.0, np.linalg.norm(x))
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        
        if use_central and call_count + 4 <= MAX_CALLS:
            Fx_p1 = np.asarray(F_counted(x + e1), dtype=float)
            Fx_m1 = np.asarray(F_counted(x - e1), dtype=float)
            Fx_p2 = np.asarray(F_counted(x + e2), dtype=float)
            Fx_m2 = np.asarray(F_counted(x - e2), dtype=float)
            if not all(np.all(np.isfinite(f)) for f in [Fx_p1, Fx_m1, Fx_p2, Fx_m2]):
                return None
            J = np.column_stack([(Fx_p1 - Fx_m1) / (2*h), (Fx_p2 - Fx_m2) / (2*h)])
        else:
            Fx_p1 = np.asarray(F_counted(x + e1), dtype=float)
            Fx_p2 = np.asarray(F_counted(x + e2), dtype=float)
            if not all(np.all(np.isfinite(f)) for f in [Fx_p1, Fx_p2]):
                return None
            J = np.column_stack([(Fx_p1 - fx) / h, (Fx_p2 - fx) / h])
        
        detJ = np.linalg.det(J)
        if abs(detJ) < 1e-10:
            J += np.eye(2) * 1e-6
        return J
    
    def broyden(x_init, J_init=None, max_iter=40):
        """Broyden's method for 2D systems. Returns (x, fx, success, J)."""
        nonlocal call_count, best_norm, best_x, cached_J
        if call_count >= MAX_CALLS or not time_ok():
            return x_init, None, False, J_init
        
        x = np.array(x_init, dtype=float)
        
        # Initial F(x)
        Fx0 = np.asarray(F_counted(x), dtype=float)
        if not np.all(np.isfinite(Fx0)):
            return x, None, False, J_init
        n0 = np.linalg.norm(Fx0)
        if n0 <= 1e-8:
            update_best(x, Fx0)
            return x, Fx0, True, J_init
        
        # Get or compute Jacobian
        if J_init is not None and np.all(np.isfinite(J_init)):
            J = J_init.copy()
        else:
            J = compute_jacobian(x, Fx0, use_central=True)
            if J is None:
                return x, Fx0, False, J_init
            cached_J = J
        
        prev_norm = n0
        stagnation_count = 0
        line_search_count = 0
        
        for _ in range(max_iter):
            if call_count >= MAX_CALLS or not time_ok():
                break
            
            # Solve J * dx = -Fx0
            try:
                dx = np.linalg.solve(J, -Fx0)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx0, rcond=None)[0]
                except:
                    break
            
            # Damping with line search
            step_norm = np.linalg.norm(dx)
            if step_norm > 10.0:
                dx = dx * (10.0 / step_norm)
            elif step_norm < 1e-15:
                break
            
            # Line search: try full step, then halve
            alpha = 1.0
            x_new = None
            Fx_new = None
            n_new = np.inf
            
            for ls_iter in range(5):
                x_candidate = x + alpha * dx
                if not np.all(np.isfinite(x_candidate)):
                    alpha *= 0.5
                    continue
                
                Fx_candidate = np.asarray(F_counted(x_candidate), dtype=float)
                if not np.all(np.isfinite(Fx_candidate)):
                    alpha *= 0.5
                    continue
                
                n_candidate = np.linalg.norm(Fx_candidate)
                
                # Accept if it reduces the norm
                if n_candidate < prev_norm * 0.95 or n_candidate <= 1e-8:
                    x_new = x_candidate
                    Fx_new = Fx_candidate
                    n_new = n_candidate
                    break
                
                alpha *= 0.5
            
            if x_new is None:
                # All line search steps failed; try smallest step
                x_candidate = x + 0.5 * alpha * dx
                if np.all(np.isfinite(x_candidate)):
                    Fx_candidate = np.asarray(F_counted(x_candidate), dtype=float)
                    if np.all(np.isfinite(Fx_candidate)):
                        n_candidate = np.linalg.norm(Fx_candidate)
                        if n_candidate < prev_norm:
                            x_new = x_candidate
                            Fx_new = Fx_candidate
                            n_new = n_candidate
                        else:
                            break
                else:
                    break
            
            update_best(x_new, Fx_new)
            
            if n_new <= 1e-8:
                return x_new, Fx_new, True, J
            
            # Stagnation detection
            if n_new >= prev_norm * 0.98:
                stagnation_count += 1
                if stagnation_count >= 4:
                    break
            else:
                stagnation_count = 0
            prev_norm = n_new
            
            # Broyden update
            s = x_new - x
            y = Fx_new - Fx0
            denom = np.dot(s, s)
            if denom > 1e-30:
                J = J + np.outer(y - J @ s, s) / denom
                cached_J = J
            
            x = x_new
            Fx0 = Fx_new
            n0 = n_new
        
        return x, Fx0, n0 <= 1e-8, J
    
    def polish(x_init, fx_init=None, max_iter=12):
        """High-precision Newton polish with central differences."""
        nonlocal call_count, best_norm, best_x
        if call_count >= MAX_CALLS - 20 or not time_ok():
            return x_init, fx_init, False
        
        x = np.array(x_init, dtype=float)
        
        if fx_init is None:
            Fx = np.asarray(F_counted(x), dtype=float)
        else:
            Fx = np.asarray(fx_init, dtype=float)
        
        n = np.linalg.norm(Fx)
        if not np.isfinite(n):
            return x, Fx, False
        
        for _ in range(max_iter):
            if call_count >= MAX_CALLS - 5 or not time_ok():
                break
            if n <= 1e-8:
                return x, Fx, True
            
            # Central difference Jacobian for precision
            h = max(1e-8, 1e-6 * max(1.0, np.linalg.norm(x)) * max(1.0, n))
            e1 = np.array([h, 0.0])
            e2 = np.array([0.0, h])
            
            if call_count + 4 <= MAX_CALLS:
                Fx_p1 = np.asarray(F_counted(x + e1), dtype=float)
                Fx_m1 = np.asarray(F_counted(x - e1), dtype=float)
                Fx_p2 = np.asarray(F_counted(x + e2), dtype=float)
                Fx_m2 = np.asarray(F_counted(x - e2), dtype=float)
                
                if not all(np.all(np.isfinite(f)) for f in [Fx_p1, Fx_m1, Fx_p2, Fx_m2]):
                    break
                
                J = np.column_stack([(Fx_p1 - Fx_m1) / (2*h), (Fx_p2 - Fx_m2) / (2*h)])
            else:
                Fx_p1 = np.asarray(F_counted(x + e1), dtype=float)
                Fx_p2 = np.asarray(F_counted(x + e2), dtype=float)
                
                if not all(np.all(np.isfinite(f)) for f in [Fx_p1, Fx_p2]):
                    break
                
                J = np.column_stack([(Fx_p1 - Fx) / h, (Fx_p2 - Fx) / h])
            
            detJ = np.linalg.det(J)
            if abs(detJ) < 1e-10:
                J += np.eye(2) * 1e-6
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    break
            
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                break
            
            Fx_new = np.asarray(F_counted(x_new), dtype=float)
            if not np.all(np.isfinite(Fx_new)):
                break
            
            n_new = np.linalg.norm(Fx_new)
            update_best(x_new, Fx_new)
            
            if n_new < n:
                x = x_new
                Fx = Fx_new
                n = n_new
            else:
                # Try smaller step
                for factor in [0.5, 0.25, 0.1]:
                    dx_small = dx * factor
                    x_small = x + dx_small
                    if not np.all(np.isfinite(x_small)):
                        continue
                    Fx_small = np.asarray(F_counted(x_small), dtype=float)
                    if not np.all(np.isfinite(Fx_small)):
                        continue
                    n_small = np.linalg.norm(Fx_small)
                    if n_small < n:
                        x = x_small
                        Fx = Fx_small
                        n = n_small
                        break
                else:
                    break
        
        return x, Fx, n <= 1e-8
    
    # Phase 1: Broyden from x0
    x_result, fx_result, success, J_final = broyden(x0, J_init=None)
    if success and best_norm <= 1e-8:
        return best_x
    
    # Phase 2: If close, try polish
    if best_norm < 1e-4 and best_norm > 1e-8 and call_count < MAX_CALLS - 20 and time_ok():
        x_pol, fx_pol, pol_success = polish(best_x, None)
        if pol_success and best_norm <= 1e-8:
            return best_x
    
    # Phase 3: Generate alternative starting points
    scale = max(1.0, np.linalg.norm(x0))
    candidates = []
    
    # Angular perturbations around best point
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        for r in [0.5, 1.0, 2.0, 0.25, 4.0]:
            candidates.append(best_x + r * scale * np.array([np.cos(angle), np.sin(angle)]))
    
    # Perturbations around x0
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        for r in [0.5, 1.0, 2.0]:
            candidates.append(x0 + r * scale * np.array([np.cos(angle), np.sin(angle)]))
    
    # Scaled versions
    for factor in [0.5, 2.0, 0.1, 5.0]:
        candidates.append(x0 * factor)
    
    # Standard points
    candidates.append(np.array([0.0, 0.0]))
    candidates.append(np.array([1.0, 0.0]))
    candidates.append(np.array([0.0, 1.0]))
    candidates.append(np.array([1.0, 1.0]))
    candidates.append(np.array([-1.0, 0.0]))
    candidates.append(np.array([0.0, -1.0]))
    candidates.append(np.array([-1.0, -1.0]))
    candidates.append(np.array([-1.0, 1.0]))
    
    # Sort candidates by distance to best_x (prioritize nearby)
    if best_norm < np.inf:
        candidates.sort(key=lambda c: np.linalg.norm(c - best_x))
    
    # Try Broyden from each candidate, reusing cached Jacobian when close
    for x_init in candidates:
        if call_count >= MAX_CALLS or not time_ok():
            break
        if not np.all(np.isfinite(x_init)):
            continue
        
        # Reuse cached Jacobian if x_init is close to where it was computed
        J_for_init = None
        if cached_J is not None and np.all(np.isfinite(cached_J)):
            dist_to_best = np.linalg.norm(x_init - best_x)
            if dist_to_best < 0.5 * scale:
                J_for_init = cached_J
        
        x_result, fx_result, success, J_final = broyden(x_init, J_init=J_for_init)
        if success and best_norm <= 1e-8:
            return best_x
        
        # If close, try polish
        if best_norm < 1e-4 and best_norm > 1e-8 and call_count < MAX_CALLS - 20 and time_ok():
            x_pol, fx_pol, pol_success = polish(best_x, None)
            if pol_success and best_norm <= 1e-8:
                return best_x
    
    # Phase 4: Final polish attempts from best point
    for _ in range(2):
        if best_norm > 1e-8 and best_norm < 1.0 and call_count < MAX_CALLS - 20 and time_ok():
            x_pol, fx_pol, pol_success = polish(best_x, None)
            if pol_success and best_norm <= 1e-8:
                return best_x
        else:
            break
    
    return best_x if np.all(np.isfinite(best_x)) else x0

# EVOLVE-BLOCK-END
