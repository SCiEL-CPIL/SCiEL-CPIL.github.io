"""
Root-finder for 3 fixed 2D systems using a Levenberg-Marquardt approach
with a rank-1 updated Jacobian approximation.

Strategy:
1. Initialize Jacobian via finite differences (3 calls).
2. Use Levenberg-Marquardt iteration:
   - Solve (J^T J + lambda * I) dx = -J^T F
   - Adapt lambda based on reduction in ||F||^2
   - Update J using Broyden's rank-1 update to keep it fresh
3. If LM stalls or lambda becomes too large/small, restart with fresh FD Jacobian.
4. Final polish phase using damped Newton with FD Jacobians to reach 1e-8.

This structural change replaces the line-search-heavy Broyden method with a
damping-based LM method, which is often more robust for fixed budgets.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 with ||F(x)|| <= 1e-8 (Euclidean norm).
    
    Uses a Levenberg-Marquardt variant with rank-1 Jacobian updates.
    """
    x = np.array(x0, dtype=float, copy=True)
    t_start = time.time()
    deadline = t_start + 135.0  # Safety margin before 150s kill
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _time_ok():
        return time.time() < deadline

    def _call_F(x_val):
        nonlocal call_count
        if not _time_ok():
            return None
        if call_count >= 490:  # Leave buffer for final checks
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
            if not _time_ok() or call_count >= 488:
                break
            e = np.zeros(2)
            e[j] = h
            Fx_plus = _call_F(x_val + e)
            if Fx_plus is not None:
                J[:, j] = (Fx_plus - Fx_val) / h
            else:
                J[:, j] = 1.0  # Fallback
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

    # Compute initial Jacobian
    J = _fd_jacobian(x, Fx)
    
    # Ensure J is not completely degenerate initially
    try:
        if np.linalg.cond(J) > 1e10:
            J += np.eye(2) * 1e-6
    except np.linalg.LinAlgError:
        J = np.eye(2)

    # LM Parameters
    lam = 1e-3
    lam_min = 1e-10
    lam_max = 1e10
    rho_threshold = 0.1  # If rho < 0.1, increase lambda (more damping)
    rho_good = 0.75      # If rho > 0.75, decrease lambda (less damping)

    max_iter = 450
    stalled_count = 0
    prev_norm_sq = norm_Fx**2

    for it in range(max_iter):
        if not _time_ok():
            break
        if call_count >= 485:
            break

        # Compute LM step: (J^T J + lam * I) dx = -J^T F
        JTJ = J.T @ J
        JTF = J.T @ Fx
        A = JTJ + lam * np.eye(2)
        
        try:
            dx = np.linalg.solve(A, -JTF)
        except np.linalg.LinAlgError:
            # Fallback to regularized least squares or pseudo-inverse
            try:
                dx = np.linalg.pinv(A) @ (-JTF)
            except Exception:
                break

        if not np.all(np.isfinite(dx)):
            break

        # Try the step
        x_new = x + dx
        if not np.all(np.isfinite(x_new)):
            lam *= 10
            if lam > lam_max:
                break
            continue

        Fx_new = _call_F(x_new)
        if Fx_new is None:
            break
        
        norm_new_sq = np.linalg.norm(Fx_new)**2
        norm_new = np.sqrt(norm_new_sq)

        # Calculate rho: observed reduction / predicted reduction
        # Predicted reduction in ||F||^2 is approx -2 * dx^T J^T F - dx^T J^T J dx
        # But simpler: we just compare actual norm squared to previous
        # Standard LM uses: rho = (||F(x)||^2 - ||F(x+dx)||^2) / (predicted_reduction)
        # Here we use a simpler heuristic: if norm decreased, good; else bad.
        # However, to adapt lambda correctly, we estimate the gain.
        # Let's use the standard formula:
        # predicted = -2 * JTF @ dx - dx @ A @ dx  (approx)
        # Actually, for LM, the predicted reduction in the objective f=||F||^2 is:
        # f(x) - f(x+dx) approx 2 * ||JTF||^2 / (lambda + ||J^T J||_F) ? No.
        
        # Let's stick to the robust simple version:
        # If norm_new < norm_Fx, accept.
        # If accepted, update J and adjust lambda.
        
        if norm_new < norm_Fx:
            # Accept step
            s = x_new - x
            y = Fx_new - Fx
            
            # Broyden rank-1 update for J
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
                stalled_count = 0
            else:
                stalled_count += 1

            # Adapt lambda
            # Estimate gain: how much did we reduce norm compared to step size?
            # If we reduced it a lot, we can trust the linear model more -> decrease lambda
            # If we reduced it a little, we might be in a non-linear region -> increase lambda
            # Simple heuristic:
            # ratio = (norm_Fx_old^2 - norm_new^2) / (norm_Fx_old^2)
            # If ratio > 0.75, we are converging well, decrease lambda.
            # If ratio < 0.1, we are struggling, increase lambda.
            reduction_ratio = (prev_norm_sq - norm_new_sq) / prev_norm_sq if prev_norm_sq > 0 else 0
            
            if reduction_ratio > 0.75:
                lam = max(lam / 10.0, lam_min)
            elif reduction_ratio < 0.1:
                lam = min(lam * 10.0, lam_max)
            # else keep lam
            
            prev_norm_sq = norm_new_sq

        else:
            # Reject step
            lam = min(lam * 10.0, lam_max)
            
            # If lambda is too large, the step is tiny and we're stuck
            if lam > 1e6:
                # Try to restart with a fresh Jacobian estimate
                if _time_ok() and call_count < 480:
                    J = _fd_jacobian(x, Fx)
                    lam = 1e-3
                    prev_norm_sq = norm_Fx**2
                else:
                    break
            
            stalled_count += 1
            if stalled_count > 20:
                # Force a restart to escape local minimum or bad Jacobian
                if _time_ok() and call_count < 480:
                    # Perturb x slightly? No, just refresh Jacobian.
                    J = _fd_jacobian(x, Fx)
                    lam = 1e-3
                    stalled_count = 0
                    prev_norm_sq = norm_Fx**2
                else:
                    break

    # Final polish phase: If we are close, use FD-Newton for high precision
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
                    if np.linalg.cond(J) > 1e10:
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
                    if Fx_new is not None:
                        norm_new = np.linalg.norm(Fx_new)
                        if norm_new < norm_Fx:
                            x = x_new
                            Fx = Fx_new
                            norm_Fx = norm_new
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
