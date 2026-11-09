"""Root-finder for 3 fixed 2D systems: Broyden core + damped-step recovery + FD-Newton polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 with ||F(x)|| <= 1e-8 (Euclidean norm).

    Strategy (single coherent pipeline, kept within the Broyden family):
      1. Broyden's method (good-Broyden) with a forward-difference initial
         Jacobian (3 calls total to start) and rank-1 updates (1 call/iter).
      2. Damped-step line search: if a full step fails to reduce the residual,
         try damped steps before giving up (recovery instead of break).
      3. FD-Newton polish pass: when Broyden stalls near a root, switch to a
         high-precision Newton with a freshly computed Jacobian and tighter
         tolerance to finish off the last digits.

    Budget: < 500 calls. Explicit deadline guard with safety margin.
    """
    x = np.array(x0, dtype=float, copy=True)
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _call_F(x_val):
        nonlocal call_count
        if time.time() > deadline:
            return None
        if call_count >= 495:
            return None
        call_count += 1
        try:
            result = F(np.asarray(x_val, dtype=float))
        except Exception:
            return None
        try:
            result = np.asarray(result, dtype=float)
        except Exception:
            return None
        if result.shape != (2,):
            try:
                result = result.reshape(2)
            except Exception:
                return None
        if not np.all(np.isfinite(result)):
            return None
        return result

    def _fd_jacobian(x, Fx):
        """Forward-difference Jacobian (2 extra calls, reuses Fx)."""
        J = np.zeros((2, 2))
        h = 1e-5
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = _call_F(x + e)
            if Fx_plus is None:
                return None
            J[:, j] = (Fx_plus - Fx) / h
        return J

    def _is_finite_2d(v):
        return np.all(np.isfinite(v))

    # ---- Initial evaluation ----
    Fx = _call_F(x)
    if Fx is None:
        return best_x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    # ---- Initial Jacobian via forward differences (2 extra calls) ----
    J = _fd_jacobian(x, Fx)
    if J is None:
        return best_x
    if abs(np.linalg.det(J)) < 1e-12:
        J = J + np.eye(2) * 1e-6

    # ---- Broyden's method with damped-step recovery ----
    stalled = 0
    for it in range(480):
        if time.time() > deadline:
            break
        if call_count >= 490:
            break

        # Solve J @ dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.pinv(J) @ (-Fx)
            except Exception:
                break

        # Step control: limit step size
        norm_dx = np.linalg.norm(dx)
        if norm_dx > 5.0:
            dx = dx / norm_dx * 5.0
        if not _is_finite_2d(dx):
            break

        # Try full step first
        accepted = False
        for damp in (1.0, 0.5, 0.25, 0.1, 0.05):
            x_new = x + damp * dx
            if not _is_finite_2d(x_new):
                continue
            Fx_new = _call_F(x_new)
            if Fx_new is None:
                break
            norm_Fx_new = np.linalg.norm(Fx_new)
            if not _is_finite_2d(Fx_new):
                continue
            if norm_Fx_new < best_norm:
                # Accept: Broyden rank-1 update
                s = x_new - x
                y = Fx_new - Fx
                s_norm_sq = np.dot(s, s)
                if s_norm_sq > 1e-16:
                    J = J + np.outer(y - J @ s, s) / s_norm_sq
                x = x_new
                Fx = Fx_new
                best_x = x_new.copy()
                best_norm = norm_Fx_new
                stalled = 0
                accepted = True
                if norm_Fx_new <= 1e-8:
                    return x_new
                break
        if not accepted:
            # Full and damped steps all failed to improve.
            # Recompute Jacobian from scratch once to escape a stale J.
            J2 = _fd_jacobian(x, Fx)
            if J2 is None:
                break
            if abs(np.linalg.det(J2)) < 1e-12:
                J2 = J2 + np.eye(2) * 1e-6
            J = J2
            stalled += 1
            if stalled > 4:
                break

    # ---- FD-Newton polish pass (high-precision finisher) ----
    # Start from the best point found; use exact local Newton with a fresh
    # Jacobian and a tighter tolerance to push ||F|| down to <= 1e-8.
    x = best_x.copy()
    Fx = _call_F(x)
    if Fx is not None:
        best_norm = np.linalg.norm(Fx)
        if best_norm <= 1e-8:
            return x
    polish_stall = 0
    for _ in range(60):
        if time.time() > deadline:
            break
        if call_count >= 495:
            break
        Fx = _call_F(x)
        if Fx is None:
            break
        norm_Fx = np.linalg.norm(Fx)
        if not _is_finite_2d(Fx):
            break
        if norm_Fx <= 1e-8:
            return x
        if norm_Fx < best_norm:
            best_x = x.copy()
            best_norm = norm_Fx
        J = _fd_jacobian(x, Fx)
        if J is None:
            break
        if abs(np.linalg.det(J)) < 1e-12:
            J = J + np.eye(2) * 1e-6
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.pinv(J) @ (-Fx)
            except Exception:
                break
        if not _is_finite_2d(dx):
            break
        # Damped Newton with tighter acceptance
        accepted = False
        for damp in (1.0, 0.5, 0.25, 0.1):
            x_new = x + damp * dx
            if not _is_finite_2d(x_new):
                continue
            Fx_new = _call_F(x_new)
            if Fx_new is None:
                break
            if not _is_finite_2d(Fx_new):
                continue
            norm_new = np.linalg.norm(Fx_new)
            if norm_new < norm_Fx:
                x = x_new
                Fx = Fx_new
                if norm_new < best_norm:
                    best_x = x_new.copy()
                    best_norm = norm_new
                accepted = True
                if norm_new <= 1e-8:
                    return x_new
                break
        if not accepted:
            polish_stall += 1
            if polish_stall > 3:
                break
        else:
            polish_stall = 0

    return best_x
# EVOLVE-BLOCK-END
