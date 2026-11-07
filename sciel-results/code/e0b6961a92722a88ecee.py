import numpy as np
import time


def find_root_system(F, x0):
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = float('inf')
    
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    calls = 0
    max_calls = 490  # Safety margin below 500

    def safe_F(x):
        nonlocal calls
        if time.time() > deadline:
            raise TimeoutError
        if calls >= max_calls:
            raise RuntimeError("Call limit exceeded")
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return None
            calls += 1
            return Fx
        except Exception:
            return None

    def is_valid(x):
        return np.all(np.isfinite(x)) and x.shape == (2,)

    # Initial evaluation
    Fx = safe_F(x)
    if Fx is None:
        return best_x
    norm = np.linalg.norm(Fx)
    best_norm = norm
    best_x = x.copy()
    if norm <= 1e-8:
        return best_x

    # Helper to compute Jacobian via finite differences
    def compute_jacobian(x, Fx, h=1e-6):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_j = safe_F(x + e)
            if Fx_j is None:
                return None
            J[:, j] = (Fx_j - Fx) / h
        return J

    def try_newton(x_start, Fx_start, max_iter=100):
        nonlocal best_x, best_norm, calls
        x = x_start.copy()
        Fx = Fx_start.copy()
        norm = np.linalg.norm(Fx)
        
        if norm <= 1e-8:
            return x, norm, True
        
        # Compute initial Jacobian
        J = compute_jacobian(x, Fx)
        if J is None:
            return x, norm, False
        
        for _ in range(max_iter):
            if time.time() > deadline or calls >= max_calls - 5:
                break
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except Exception:
                    break
            
            if not np.all(np.isfinite(dx)):
                break
            
            # Backtracking line search
            alpha = 1.0
            Fx_old_norm = norm
            step_accepted = False
            
            for _ in range(15):
                x_try = x + alpha * dx
                if not is_valid(x_try):
                    alpha *= 0.5
                    continue
                Fx_try = safe_F(x_try)
                if Fx_try is None:
                    alpha *= 0.5
                    continue
                
                norm_try = np.linalg.norm(Fx_try)
                
                if norm_try < best_norm:
                    best_norm = norm_try
                    best_x = x_try.copy()
                
                if norm_try < Fx_old_norm * 0.9:  # Sufficient decrease
                    # Update Jacobian (Broyden)
                    s = alpha * dx
                    y = Fx_try - Fx
                    J_s = J @ s
                    diff = y - J_s
                    denom = s @ s
                    if denom > 1e-15:
                        J = J + np.outer(diff, s) / denom
                    else:
                        J = compute_jacobian(x_try, Fx_try)
                        if J is None:
                            break
                    
                    x = x_try
                    Fx = Fx_try
                    norm = norm_try
                    step_accepted = True
                    break
                else:
                    alpha *= 0.5
            
            if not step_accepted:
                break
            
            if norm <= 1e-8:
                return x, norm, True
        
        return x, norm, norm <= 1e-8

    def high_precision_polish(x_start, Fx_start, max_iter=30):
        """Final polish using exact Newton with tight tolerance."""
        nonlocal best_x, best_norm, calls
        x = x_start.copy()
        Fx = Fx_start.copy()
        norm = np.linalg.norm(Fx)
        
        if norm <= 1e-8:
            return x, norm, True
        
        for _ in range(max_iter):
            if time.time() > deadline - 5 or calls >= max_calls - 10:
                break
            
            # Use tighter finite difference step for high precision
            h = max(1e-8, 1e-6 * np.sqrt(np.linalg.norm(x) + 1.0))
            J = compute_jacobian(x, Fx, h=h)
            if J is None:
                break
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except Exception:
                    break
            
            if not np.all(np.isfinite(dx)):
                break
            
            # Simple step, no backtracking needed near convergence
            x_try = x + dx
            if not is_valid(x_try):
                break
            
            Fx_try = safe_F(x_try)
            if Fx_try is None:
                break
            
            norm_try = np.linalg.norm(Fx_try)
            
            if norm_try < best_norm:
                best_norm = norm_try
                best_x = x_try.copy()
            
            if norm_try < norm * 0.5:
                x = x_try
                Fx = Fx_try
                norm = norm_try
            
            if norm <= 1e-8:
                return x, norm, True
        
        return x, norm, norm <= 1e-8

    # Try Newton from initial point
    x_result, norm_result, success = try_newton(x, Fx, max_iter=150)
    if success:
        return x_result
    
    # If close to convergence, try high-precision polish
    if norm_result <= 1e-6 and calls < max_calls - 20:
        x_polish, norm_polish, polish_success = high_precision_polish(x_result, F(x_result) if calls < max_calls - 1 else Fx, max_iter=30)
        if polish_success:
            return x_polish
        if norm_polish < norm_result:
            x_result = x_polish
            norm_result = norm_polish
    
    # Multi-start strategy: try different initial points
    # Generate candidate points based on the initial guess and function behavior
    candidates = []
    
    # Points along coordinate axes from x0
    for scale in [0.5, 2.0, 5.0, 10.0]:
        for sign in [-1, 1]:
            for axis in range(2):
                x_cand = x0.copy()
                x_cand[axis] += sign * scale * (abs(x0[axis]) + 1.0)
                candidates.append(x_cand)
    
    # Points based on gradient direction (approximate)
    if norm > 0:
        grad_dir = -Fx / norm
        for scale in [1.0, 2.0, 5.0]:
            x_cand = x + scale * grad_dir
            candidates.append(x_cand)
    
    # Random points in a neighborhood
    rng = np.random.RandomState(42)
    for _ in range(10):
        x_cand = x + rng.randn(2) * (np.linalg.norm(x) + 1.0)
        candidates.append(x_cand)
    
    # Try each candidate with limited Newton iterations
    for x_cand in candidates:
        if time.time() > deadline - 10 or calls >= max_calls - 10:
            break
        
        if not is_valid(x_cand):
            continue
        
        Fx_cand = safe_F(x_cand)
        if Fx_cand is None:
            continue
        
        norm_cand = np.linalg.norm(Fx_cand)
        if norm_cand < best_norm:
            best_norm = norm_cand
            best_x = x_cand.copy()
        
        if norm_cand <= 1e-8:
            return x_cand
        
        # Try Newton from this candidate with fewer iterations
        x_result, norm_result, success = try_newton(x_cand, Fx_cand, max_iter=50)
        if success:
            return x_result
        
        # If close, try polish
        if norm_result <= 1e-6 and calls < max_calls - 20:
            Fx_current = safe_F(x_result)
            if Fx_current is not None:
                x_polish, norm_polish, polish_success = high_precision_polish(x_result, Fx_current, max_iter=30)
                if polish_success:
                    return x_polish
                if norm_polish < norm_result:
                    x_result = x_polish
                    norm_result = norm_polish
    
    # Final polish attempt on best_x if we're close
    if best_norm <= 1e-6 and calls < max_calls - 20:
        Fx_best = safe_F(best_x)
        if Fx_best is not None:
            x_final, norm_final, final_success = high_precision_polish(best_x, Fx_best, max_iter=30)
            if final_success:
                return x_final
            if norm_final < best_norm:
                best_x = x_final
                best_norm = norm_final
    
    return best_x
