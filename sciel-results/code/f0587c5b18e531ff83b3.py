"""Global Grid-Search Screening + Safeguarded Broyden Root Finder for 3 fixed 2D systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_DEADLINE_MARGIN = 5  # seconds to spare before 150s limit

def _elapsed():
    return time.time() - _START_TIME if _START_TIME is not None else 0.0

def _time_left():
    return 150.0 - _elapsed() - _DEADLINE_MARGIN

def _safe_norm(x, F):
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            return np.inf
        return float(np.linalg.norm(Fx))
    except Exception:
        return np.inf

def _broyden(F, x0, tol=1e-12, max_iter=50):
    """
    Safeguarded Broyden's method (good direction) with backtracking line search.
    """
    x = np.array(x0, dtype=float)
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            return x
    except Exception:
        return x

    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= tol:
        return x

    # Initialize Jacobian approximation
    # Use finite differences for the first step to get a decent J0
    h = 1e-5
    J = np.zeros((2, 2))
    valid_J = True
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        try:
            Fx_p = F(x + e)
            Fx_p = np.asarray(Fx_p, dtype=float)
            if Fx_p.shape != (2,) or not np.all(np.isfinite(Fx_p)):
                valid_J = False
                break
            J[:, j] = (Fx_p - Fx) / h
        except Exception:
            valid_J = False
            break

    if not valid_J:
        # Fallback to identity if FD fails
        J = np.eye(2)

    for k in range(max_iter):
        if _time_left() < 0.5:
            break

        try:
            # Solve J * s = -Fx
            try:
                s = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Pseudo-inverse fallback
                s = np.linalg.pinv(J) @ (-Fx)

            if not np.all(np.isfinite(s)):
                break

            # Backtracking line search
            alpha = 1.0
            improved = False
            for _ in range(20):
                x_new = x + alpha * s
                if not np.all(np.isfinite(x_new)):
                    alpha *= 0.5
                    continue

                try:
                    Fx_new = F(x_new)
                    Fx_new = np.asarray(Fx_new, dtype=float)
                    if Fx_new.shape != (2,) or not np.all(np.isfinite(Fx_new)):
                        alpha *= 0.5
                        continue
                except Exception:
                    alpha *= 0.5
                    continue

                norm_Fx_new = np.linalg.norm(Fx_new)

                if norm_Fx_new < norm_Fx:
                    improved = True
                    break

                alpha *= 0.5

            if not improved:
                break

            # Update x, Fx
            x_old = x
            x = x_new
            Fx_old = Fx
            Fx = Fx_new
            norm_Fx = norm_Fx_new

            if norm_Fx <= tol:
                break

            # Update Jacobian (Broyden good direction)
            # s = x - x_old
            # y = F(x) - F(x_old)
            y = Fx - Fx_old
            s_vec = x - x_old

            try:
                # J = J + (y - J*s) * s^T / (s^T * s)
                Js = J @ s_vec
                diff = y - Js
                denom = np.dot(s_vec, s_vec)
                if denom > 1e-12:
                    J += np.outer(diff, s_vec) / denom
            except Exception:
                pass

        except Exception:
            break

    return x

def find_root_system(F, x0):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    x0 = np.asarray(x0, dtype=float).flatten()
    if x0.shape != (2,):
        x0 = np.array([0.0, 0.0])

    best_x = None
    best_norm = np.inf

    def _update_best(x_candidate):
        nonlocal best_x, best_norm
        if x_candidate is None:
            return
        try:
            x_candidate = np.asarray(x_candidate, dtype=float).flatten()
            if x_candidate.shape != (2,) or not np.all(np.isfinite(x_candidate)):
                return
            n = _safe_norm(x_candidate, F)
            if n < best_norm:
                best_norm = n
                best_x = x_candidate
        except Exception:
            pass

    # 1. Try Broyden from x0
    x_broyden = _broyden(F, x0)
    _update_best(x_broyden)
    if best_norm <= 1e-8:
        return best_x

    # 2. Global Grid Search Screening
    # We define a grid of starting points around x0 and the origin.
    # This helps escape local basins if x0 is bad.
    points_to_try = []

    # Add x0 and origin
    points_to_try.append(x0)
    points_to_try.append(np.array([0.0, 0.0]))

    # Generate grid points
    # Scale based on x0 magnitude or default
    scale = max(1.0, np.linalg.norm(x0))
    # Radii to explore
    radii = [scale * 0.5, scale, scale * 2, scale * 5]
    
    for r in radii:
        # 8 points around the circle + center
        for theta in np.linspace(0, 2 * np.pi, 9, endpoint=False):
            offset = np.array([r * np.cos(theta), r * np.sin(theta)])
            points_to_try.append(x0 + offset)
            points_to_try.append(offset) # Also try from origin offset

    # Deduplicate and limit
    unique_points = []
    seen = set()
    for p in points_to_try:
        key = tuple(np.round(p, 6))
        if key not in seen:
            seen.add(key)
            unique_points.append(p)

    # Sort by distance to x0? Or just iterate? 
    # Let's prioritize points closer to x0 first, then expand.
    # Actually, the grid generation above already expands.
    # But we should check time budget.
    
    for p in unique_points:
        if _time_left() < 1.0:
            break
        
        x_broyden = _broyden(F, p)
        _update_best(x_broyden)
        if best_norm <= 1e-8:
            return best_x

    # 3. Final high-precision polish if we have a candidate
    if best_x is not None:
        final_x = _broyden(F, best_x, tol=1e-14, max_iter=20)
        _update_best(final_x)

    if best_x is not None:
        return best_x
    return x0

# EVOLVE-BLOCK-END
