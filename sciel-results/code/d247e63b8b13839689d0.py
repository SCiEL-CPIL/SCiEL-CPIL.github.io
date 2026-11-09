"""Root-finder for 3 fixed 2D systems using a hybrid approach:
  1. Broyden's method for fast local convergence when close to a root.
  2. SciPy's L-BFGS-B on ||F||^2 as a robust derivative-free-like fallback.
  3. Final polish with FD-Newton steps for high precision.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 with ||F(x)|| <= 1e-8 (Euclidean norm).
    
    Strategy:
    - Try Broyden's method first for fast convergence.
    - If that fails or stalls, use scipy L-BFGS-B on g(x) = 0.5*||F(x)||^2.
    - Final polish with Newton steps.
    
    Budget: <500 F-calls per system. Time-limited to ~130s.
    """
    x = np.array(x0, dtype=float, copy=True)
    t_start = time.time()
    deadline = t_start + 130.0
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

    # Phase 1: Broyden's method
    # Compute initial Jacobian
    J = _fd_jacobian(x, Fx)
    try:
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-10:
            J += np.eye(2) * 1e-6
    except np.linalg.LinAlgError:
        J = np.eye(2)

    max_iter = 30
    stalled_count = 0
    prev_norm = norm_Fx
    broyden_success = False

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
        if norm_dx < 1e-15:
            break

        # Damped step
        alpha = 1.0
        success = False
        for _ in range(10):
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
            # Try a small fixed step
            x_new = x + 0.05 * dx
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

        if stalled_count >= 4:
            break

    # Check if Broyden succeeded
    if norm_Fx <= 1e-8:
        return x

    # Phase 2: SciPy L-BFGS-B fallback on g(x) = 0.5 * ||F(x)||^2
    if best_norm > 1e-6 and _time_ok() and call_count < 470:
        def objective(x_val):
            """Objective: 0.5 * ||F(x)||^2. Counts F calls."""
            nonlocal call_count
            if not _time_ok() or call_count >= 490:
                return 1e10, None
            Fx_val = _call_F(x_val)
            if Fx_val is None:
                return 1e10, None
            val = 0.5 * np.dot(Fx_val, Fx_val)
            # Numerical gradient via finite differences (2 extra calls per component)
            if call_count >= 488:
                return val, None
            h = 1e-5
            grad = np.zeros(2)
            for j in range(2):
                if not _time_ok() or call_count >= 490:
                    return val, None
                e = np.zeros(2)
                e[j] = h
                Fx_plus = _call_F(x_val + e)
                if Fx_plus is None:
                    return val, None
                val_plus = 0.5 * np.dot(Fx_plus, Fx_plus)
                grad[j] = (val_plus - val) / h
            return val, grad

        # Try L-BFGS-B
        try:
            result = minimize(objective, best_x.copy(), method='L-BFGS-B',
                             options={'maxiter': 30, 'ftol': 1e-14, 'gtol': 1e-10,
                                      'maxfun': 200})
            if result.x is not None and np.all(np.isfinite(result.x)):
                Fx_res = _call_F(result.x)
                if Fx_res is not None:
                    norm_res = np.linalg.norm(Fx_res)
                    if norm_res < best_norm:
                        best_x = result.x.copy()
                        best_norm = norm_res
                    if norm_res <= 1e-8:
                        return result.x
        except Exception:
            pass

    # Phase 3: Final polish with FD-Newton if close
    if best_norm < 1e-4 and call_count < 495 and _time_ok():
        x = best_x.copy()
        Fx = _call_F(x)
        if Fx is not None:
            norm_Fx = np.linalg.norm(Fx)
            for _ in range(8):
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
                for _ in range(6):
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
                best_norm = nf

    return best_x
# EVOLVE-BLOCK-END
