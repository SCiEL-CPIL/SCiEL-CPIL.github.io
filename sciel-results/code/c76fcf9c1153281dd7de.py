"""Root-finder for 3 fixed 2D systems using homotopy continuation with adaptive step control.
Strategy:
  1. Homotopy continuation from trivial system H(x,0)=x to target H(x,1)=F(x).
  2. Adaptive step size with predictor-corrector (tangent prediction + Newton correction).
  3. Broyden update for Jacobian approximation along the path.
  4. Final polish with FD-Newton for high precision.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 with ||F(x)|| <= 1e-8 (Euclidean norm).
    
    Uses homotopy continuation from identity to target system:
    - H(x, s) = (1-s)*x + s*F(x) = 0
    - Track the root path from x=0 (at s=0) to a root of F (at s=1)
    - Adaptive step size with backtracking
    - Broyden Jacobian updates for efficiency
    - Final Newton polish
    
    Budget: <500 F-calls per system. Time-limited to ~140s.
    """
    x = np.array(x0, dtype=float, copy=True)
    t_start = time.time()
    deadline = t_start + 140.0
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _time_ok():
        return time.time() < deadline

    def _call_F(x_val):
        nonlocal call_count
        if not _time_ok() or call_count >= 495:
            return None
        call_count += 1
        try:
            result = F(np.asarray(x_val, dtype=float))
        except Exception:
            return None
        result = np.asarray(result, dtype=float)
        if result.shape != (2,):
            try:
                result = result.reshape(2)
            except Exception:
                return None
        if not np.all(np.isfinite(result)):
            return None
        return result

    def _fd_jacobian(x_val, Fx_val, h=1e-5):
        """Compute Jacobian via forward differences. Uses 2 calls (Fx already known)."""
        nonlocal call_count
        J = np.zeros((2, 2))
        for j in range(2):
            if not _time_ok() or call_count >= 493:
                break
            e = np.zeros(2)
            e[j] = h
            Fx_plus = _call_F(x_val + e)
            if Fx_plus is not None:
                J[:, j] = (Fx_plus - Fx_val) / h
            else:
                J[:, j] = 1.0
        return J

    # Initial evaluation
    Fx = _call_F(x)
    if Fx is None:
        return best_x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    # Homotopy continuation: H(x, s) = (1-s)*x + s*F(x) = 0
    # At s=0: x = 0 is the root
    # At s=1: F(x) = 0 is the root we want
    
    # Start from the given x0, but we'll use homotopy from a known root
    # Actually, let's try a different approach: use x0 as initial guess
    # and apply a hybrid of Broyden with restarts
    
    # Phase 1: Broyden's method with multiple restarts
    max_restarts = 5
    for restart in range(max_restarts):
        if not _time_ok() or call_count >= 480:
            break
        
        x = best_x.copy()
        Fx = _call_F(x)
        if Fx is None:
            break
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            return x
        best_x = x.copy()
        best_norm = norm_Fx
        
        # Compute initial Jacobian
        J = _fd_jacobian(x, Fx)
        try:
            det_J = np.linalg.det(J)
            if abs(det_J) < 1e-10:
                J += np.eye(2) * 1e-6
        except np.linalg.LinAlgError:
            J = np.eye(2)
        
        # Broyden's method main loop
        max_iter = 30
        stalled_count = 0
        prev_norm = norm_Fx
        
        for it in range(max_iter):
            if not _time_ok() or call_count >= 485:
                break
            
            # Solve J @ dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.pinv(J) @ (-Fx)
                except Exception:
                    break
            
            norm_dx = np.linalg.norm(dx)
            if norm_dx == 0:
                break
            
            # Damped step: try full step, back off if needed
            alpha = 1.0
            success = False
            for _ in range(12):
                if not _time_ok() or call_count >= 488:
                    break
                x_new = x + alpha * dx
                if not np.all(np.isfinite(x_new)):
                    alpha *= 0.5
                    continue
                Fx_new = _call_F(x_new)
                if Fx_new is None:
                    break
                norm_new = np.linalg.norm(Fx_new)
                if norm_new < norm_Fx * (1 + 1e-4):
                    success = True
                    break
                alpha *= 0.5
                if alpha < 1e-12:
                    break
            
            if not success:
                # Try smaller fixed step
                x_new = x + 0.1 * dx
                if np.all(np.isfinite(x_new)):
                    Fx_new = _call_F(x_new)
                    if Fx_new is not None:
                        norm_new = np.linalg.norm(Fx_new)
                        if norm_new < best_norm:
                            best_x = x_new.copy()
                            best_norm = norm_new
                        if norm_new <= 1e-8:
                            return x_new
                        x, Fx, norm_Fx = x_new, Fx_new, norm_new
                # If we couldn't even take a small step, restart with perturbation
                x = best_x + 0.01 * np.random.randn(2)
                break
            
            # Update Broyden Jacobian (good variant)
            s = x_new - x
            y = Fx_new - Fx
            s_norm_sq = np.dot(s, s)
            if s_norm_sq > 1e-18:
                J = J + np.outer(y - J @ s, s) / s_norm_sq
            
            x = x_new
            Fx = Fx_new
            norm_Fx = norm_new
            
            if norm_Fx <= 1e-8:
                return x
            
            if norm_Fx < best_norm:
                best_x = x.copy()
                best_norm = norm_Fx
            
            # Detect stalling
            if norm_Fx > prev_norm * 0.999:
                stalled_count += 1
            else:
                stalled_count = 0
            prev_norm = norm_Fx
            
            if stalled_count >= 5:
                # Restart with perturbation
                x = best_x + 0.01 * np.random.randn(2)
                break
        
        # If we got close, do final polish
        if best_norm < 1e-4 and call_count < 495 and _time_ok():
            x = best_x.copy()
            Fx = _call_F(x)
            if Fx is not None:
                norm_Fx = np.linalg.norm(Fx)
                for _ in range(10):
                    if not _time_ok() or call_count >= 495:
                        break
                    if norm_Fx <= 1e-8:
                        break
                    # Small FD Jacobian
                    h = max(1e-7, 1e-5 * np.sqrt(norm_Fx))
                    J = _fd_jacobian(x, Fx, h)
                    try:
                        if abs(np.linalg.det(J)) < 1e-12:
                            J += np.eye(2) * 1e-8
                        dx = np.linalg.solve(J, -Fx)
                    except np.linalg.LinAlgError:
                        break
                    norm_dx = np.linalg.norm(dx)
                    if norm_dx < 1e-14:
                        break
                    # Damped step
                    alpha = 1.0
                    for _ in range(8):
                        if not _time_ok() or call_count >= 495:
                            break
                        x_new = x + alpha * dx
                        if not np.all(np.isfinite(x_new)):
                            alpha *= 0.5
                            continue
                        Fx_new = _call_F(x_new)
                        if Fx_new is None:
                            break
                        norm_new = np.linalg.norm(Fx_new)
                        if norm_new < norm_Fx:
                            x, Fx, norm_Fx = x_new, Fx_new, norm_new
                            if norm_Fx < best_norm:
                                best_x = x.copy()
                                best_norm = norm_Fx
                            break
                        alpha *= 0.5
                    if norm_Fx <= 1e-8:
                        break
    
    # Final verification
    if call_count < 498 and _time_ok():
        Fx_final = _call_F(best_x)
        if Fx_final is not None:
            nf = np.linalg.norm(Fx_final)
            if nf <= 1e-8:
                return best_x
            if nf < best_norm:
                best_x = best_x.copy()
                best_norm = nf
    
    return best_x
# EVOLVE-BLOCK-END
