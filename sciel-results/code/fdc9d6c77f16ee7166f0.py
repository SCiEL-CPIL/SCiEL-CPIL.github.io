# EVOLVE-BLOCK-START
import numpy as np
import time


def _safe_F(F, x):
    """Call F safely, returning a finite array of length 2 or None on failure."""
    try:
        y = F(x)
        y = np.asarray(y, dtype=float)
        if y.shape != (2,) or not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None


def _broyden_update(J, x_new, x_old, F_new, F_old):
    """Rank-1 Broyden update."""
    s = x_new - x_old
    y = F_new - F_old
    denom = np.dot(y, s)
    if abs(denom) > 1e-14:
        J = J + np.outer(y - J @ s, s) / denom
    return J


def _newton_broyden(F, x0, max_iter=30, tol=1e-12):
    """Newton's method with Broyden update and line search."""
    x = np.array(x0, dtype=float)
    Fx = _safe_F(F, x)
    if Fx is None:
        return x, 1
    calls = 1
    
    if np.linalg.norm(Fx) <= tol:
        return x, calls
    
    # Initial Jacobian via finite differences
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = 1e-7
        Fx_e = _safe_F(F, x + e)
        calls += 1
        if Fx_e is None:
            J[:, j] = np.array([1.0, 0.0]) if j == 0 else np.array([0.0, 1.0])
        else:
            J[:, j] = (Fx_e - Fx) / 1e-7
    
    for _ in range(max_iter):
        # Solve J dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Regularize
            try:
                dx = np.linalg.solve(J + 1e-10 * np.eye(2), -Fx)
            except np.linalg.LinAlgError:
                break
        
        # Line search
        alpha = 1.0
        Fx_norm = np.linalg.norm(Fx)
        for _ in range(20):
            x_new = x + alpha * dx
            Fx_new = _safe_F(F, x_new)
            calls += 1
            if Fx_new is None:
                alpha *= 0.5
                continue
            Fx_new_norm = np.linalg.norm(Fx_new)
            if Fx_new_norm <= tol:
                return x_new, calls
            if Fx_new_norm < Fx_norm:
                x = x_new
                Fx = Fx_new
                J = _broyden_update(J, x_new, x - alpha * dx, Fx_new, Fx - (Fx_new - (Fx_new - (Fx - (Fx_new - Fx)))))
                # Simpler update:
                J = _broyden_update(J, x, x - alpha * dx, Fx, Fx - alpha * (Fx - Fx))
                # Let's redo the Broyden update properly
                # We need F_old and F_new, x_old and x_new
                # x_old = x - alpha * dx, F_old = Fx_old
                # x_new = x, F_new = Fx
                # But we already updated x and Fx above
                # So we need to be careful
                break
            alpha *= 0.5
        else:
            # No improvement, try smaller step or break
            break
        
        # Proper Broyden update
        x_old = x - alpha * dx
        Fx_old = _safe_F(F, x_old)
        if Fx_old is not None:
            J = _broyden_update(J, x, x_old, Fx, Fx_old)
            calls += 1
        else:
            # Keep old J
            pass
    
    return x, calls


def find_root_system(F, x0):
    """
    Root finder for 2D systems using a hybrid approach:
    1. Try Newton-Broyden from x0
    2. If that fails, try multiple restarts with grid search
    3. Use deadline guard to avoid timeout
    """
    deadline = time.time() + 110  # Leave 10s safety margin for 120s limit
    x0 = np.array(x0, dtype=float)
    
    # First, try Newton-Broyden from x0
    x, calls = _newton_broyden(F, x0)
    Fx = _safe_F(F, x)
    if Fx is not None and np.linalg.norm(Fx) <= 1e-8:
        return x
    calls += 1
    
    if time.time() > deadline:
        return x
    
    # Try multiple restarts
    # Generate candidate starting points
    candidates = [x0.copy()]
    
    # Grid around x0
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                candidates.append(x0 + np.array([dx * scale, dy * scale]))
    
    # Random candidates
    np.random.seed(42)
    for _ in range(10):
        candidates.append(x0 + np.random.randn(2) * 5.0)
    
    best_x = x
    best_norm = np.inf
    if Fx is not None:
        best_norm = np.linalg.norm(Fx)
    
    for x_start in candidates:
        if time.time() > deadline:
            break
        if not np.all(np.isfinite(x_start)):
            continue
        
        x_new, calls_new = _newton_broyden(F, x_start, max_iter=15)
        calls += calls_new
        Fx_new = _safe_F(F, x_new)
        calls += 1
        if Fx_new is not None:
            norm_new = np.linalg.norm(Fx_new)
            if norm_new < best_norm:
                best_norm = norm_new
                best_x = x_new
                if norm_new <= 1e-8:
                    return best_x
    
    return best_x
# EVOLVE-BLOCK-END
