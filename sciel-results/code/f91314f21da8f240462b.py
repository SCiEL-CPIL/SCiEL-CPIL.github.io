"""Root finder for 3 fixed 2-D systems using robust Broyden's method with restarts."""
import time
import numpy as np

def find_root_system(F, x0):
    deadline = time.time() + 145.0
    tol = 1e-8
    x = np.array(x0, dtype=float)
    
    def time_ok():
        return time.time() < deadline

    def safe_eval_F(v):
        try:
            result = np.asarray(F(v), dtype=float)
            if result.shape != (2,) or not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            return None

    def compute_J(x, Fx):
        h = 1e-6
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = safe_eval_F(x + e)
            if Fx_plus is None:
                return None
            J[:, j] = (Fx_plus - Fx) / h
        return J

    # Initial evaluation
    Fx = safe_eval_F(x)
    if Fx is None:
        return x
    if np.linalg.norm(Fx) <= tol:
        return x

    J = compute_J(x, Fx)
    if J is None:
        return x

    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)
    
    max_iters = 480
    for it in range(max_iters):
        if not time_ok():
            break
        
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.solve(J + 1e-6 * np.eye(2), -Fx)
            except np.linalg.LinAlgError:
                break
        
        # Check for stagnation: if dx is very small, we might be stuck
        if np.linalg.norm(dx) < 1e-12:
            break

        x_new = x + dx
        Fx_new = safe_eval_F(x_new)
        if Fx_new is None:
            break
        
        norm_new = np.linalg.norm(Fx_new)
        
        if norm_new <= tol:
            return x_new
        
        if norm_new < best_norm:
            s = x_new - x
            y = Fx_new - Fx
            Js = J @ s
            J += np.outer(y - Js, s) / (s @ s)
            
            x = x_new
            Fx = Fx_new
            best_x = x.copy()
            best_norm = norm_new
        else:
            alpha = 0.5
            found = False
            for _ in range(5):
                if not time_ok():
                    break
                x_trial = x + alpha * dx
                Fx_trial = safe_eval_F(x_trial)
                if Fx_trial is None:
                    break
                norm_trial = np.linalg.norm(Fx_trial)
                if norm_trial < best_norm:
                    s = x_trial - x
                    y = Fx_trial - Fx
                    Js = J @ s
                    J += np.outer(y - Js, s) / (s @ s)
                    
                    x = x_trial
                    Fx = Fx_trial
                    best_x = x.copy()
                    best_norm = norm_trial
                    if norm_trial <= tol:
                        return x
                    found = True
                    break
                alpha *= 0.5
            
            if not found:
                # Stagnation detected: try restarting with a new Jacobian
                if it < 400:
                    J_new = compute_J(x, Fx)
                    if J_new is not None:
                        J = J_new
                    else:
                        break
                else:
                    break

    return best_x
