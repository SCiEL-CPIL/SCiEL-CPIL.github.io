"""Root finder using Globalized Trust-Region Newton with multi-start fallback."""
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    time_limit = 140.0
    
    x0_arr = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0_arr)):
        x0_arr = np.zeros(2)
    
    call_counter = [0]
    max_calls = 495
    
    def safe_eval(x):
        if call_counter[0] >= max_calls:
            return None
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                try:
                    Fx = Fx.reshape(2)
                except:
                    return None
            if not np.all(np.isfinite(Fx)):
                return None
            call_counter[0] += 1
            return Fx
        except Exception:
            return None
    
    def compute_jacobian(x, Fx):
        """Compute Jacobian using central differences (4 calls)."""
        J = np.zeros((2, 2))
        h = 1e-7
        for j in range(2):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += h
            x_minus[j] -= h
            
            Fx_plus = safe_eval(x_plus)
            Fx_minus = safe_eval(x_minus)
            
            if Fx_plus is None or Fx_minus is None:
                # Fall back to forward difference if central fails
                x_plus = x.copy()
                x_plus[j] += h
                Fx_plus = safe_eval(x_plus)
                if Fx_plus is None:
                    return None
                J[:, j] = (Fx_plus - Fx) / h
            else:
                J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
        return J
    
    def trust_region_newton(x_start, max_iters=80):
        """Globalized trust-region Newton method."""
        nonlocal call_counter
        
        x = x_start.copy()
        Fx = safe_eval(x)
        if Fx is None:
            return None, float('inf')
        
        norm_F = np.linalg.norm(Fx)
        if norm_F <= 1e-8:
            return x, norm_F
        
        # Initial Jacobian
        J = compute_jacobian(x, Fx)
        if J is None:
            return x, norm_F
        
        delta = 1.0  # trust region radius
        min_delta = 1e-12
        max_delta = 100.0
        best_x = x.copy()
        best_norm = norm_F
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            if call_counter[0] >= max_calls:
                break
            
            # Solve J * s = -Fx with trust region constraint
            try:
                s = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            
            s_norm = np.linalg.norm(s)
            
            # Scale step to trust region if needed
            if s_norm > delta:
                s = s * (delta / s_norm)
            
            x_trial = x + s
            Fx_trial = safe_eval(x_trial)
            
            if Fx_trial is None:
                delta *= 0.5
                if delta < min_delta:
                    break
                continue
            
            norm_trial = np.linalg.norm(Fx_trial)
            
            # Compute actual reduction
            pred_reduction = 0.5 * np.dot(s, Fx) - 0.25 * np.dot(s, J @ s)
            actual_reduction = 0.5 * (norm_F**2 - norm_trial**2)
            
            if pred_reduction > 1e-15:
                rho = actual_reduction / pred_reduction
            else:
                rho = 1.0 if actual_reduction > 0 else 0.0
            
            if rho > 0.75:
                # Accept step, expand trust region
                x = x_trial
                Fx = Fx_trial
                norm_F = norm_trial
                delta = min(max_delta, delta * 2.0)
                
                # Update Jacobian
                J_new = compute_jacobian(x, Fx)
                if J_new is not None:
                    J = J_new
                
                if norm_F < best_norm:
                    best_norm = norm_F
                    best_x = x.copy()
                
                if norm_F <= 1e-8:
                    return best_x, best_norm
            elif rho > 0.25:
                # Accept step, keep trust region
                x = x_trial
                Fx = Fx_trial
                norm_F = norm_trial
                delta = delta  # unchanged
                
                J_new = compute_jacobian(x, Fx)
                if J_new is not None:
                    J = J_new
                
                if norm_F < best_norm:
                    best_norm = norm_F
                    best_x = x.copy()
                
                if norm_F <= 1e-8:
                    return best_x, best_norm
            else:
                # Reject step, shrink trust region
                delta *= 0.5
                if delta < min_delta:
                    break
        
        return best_x, best_norm
    
    # Multi-start points
    starts = [x0_arr.copy()]
    
    # Add some perturbed starts
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            perturbation = np.array([scale * np.cos(angle), scale * np.sin(angle)])
            starts.append(x0_arr + perturbation)
    
    # Also try origin and some standard points
    starts.append(np.zeros(2))
    starts.append(np.ones(2))
    starts.append(np.array([1.0, 0.0]))
    starts.append(np.array([0.0, 1.0]))
    
    best_overall_x = x0_arr.copy()
    best_overall_norm = float('inf')
    
    for start in starts:
        if time.time() - start_time > time_limit:
            break
        if call_counter[0] >= max_calls:
            break
        
        # Skip if start is too far or non-finite
        if not np.all(np.isfinite(start)):
            continue
        if np.linalg.norm(start) > 100:
            continue
        
        x_result, norm_result = trust_region_newton(start)
        
        if x_result is not None and norm_result < best_overall_norm:
            best_overall_norm = norm_result
            best_overall_x = x_result.copy()
        
        if best_overall_norm <= 1e-8:
            break
    
    # Final polish if close but not quite there
    if best_overall_norm <= 1e-4 and best_overall_norm > 1e-8:
        # Try a few more Newton steps from the best point
        x = best_overall_x.copy()
        Fx = safe_eval(x)
        if Fx is not None:
            J = compute_jacobian(x, Fx)
            if J is not None:
                for _ in range(5):
                    if time.time() - start_time > time_limit:
                        break
                    if call_counter[0] >= max_calls:
                        break
                    
                    try:
                        s = np.linalg.solve(J, -Fx)
                    except np.linalg.LinAlgError:
                        break
                    
                    if not np.all(np.isfinite(s)):
                        break
                    
                    x_new = x + s
                    Fx_new = safe_eval(x_new)
                    if Fx_new is None:
                        break
                    
                    norm_new = np.linalg.norm(Fx_new)
                    if norm_new < np.linalg.norm(Fx):
                        x = x_new
                        Fx = Fx_new
                        J_new = compute_jacobian(x, Fx)
                        if J_new is not None:
                            J = J_new
                        if norm_new < best_overall_norm:
                            best_overall_norm = norm_new
                            best_overall_x = x.copy()
                        if norm_new <= 1e-8:
                            break
    
    return best_overall_x
