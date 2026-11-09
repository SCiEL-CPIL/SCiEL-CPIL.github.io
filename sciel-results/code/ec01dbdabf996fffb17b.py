"""Root finder for 3 fixed two-dimensional systems using a robust hybrid approach."""
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


def _nelder_mead(F, x0, max_iter=200, tol=1e-12, deadline=None):
    """Nelder-Mead simplex method for 2D root finding."""
    # Initial simplex
    x0 = np.array(x0, dtype=float)
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5    # contraction
    sigma = 0.5  # shrink

    # Create initial simplex
    simplex = [x0.copy()]
    for i in range(2):
        xi = x0.copy()
        xi[i] += 0.1 * (abs(x0[i]) + 1.0)
        simplex.append(xi)

    # Evaluate objective (squared norm of F)
    def obj(x):
        Fx = _safe_F(F, x)
        if Fx is None:
            return 1e10
        return np.dot(Fx, Fx)

    vals = [obj(x) for x in simplex]
    best_idx = np.argmin(vals)
    best_x = simplex[best_idx].copy()
    best_val = vals[best_idx]

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        if best_val <= tol:
            break

        # Sort simplex
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        best_idx = 0
        best_x = simplex[0].copy()
        best_val = vals[0]

        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = obj(x_r)

        if f_r < vals[0]:
            # Expansion
            x_e = centroid + gamma * (centroid - simplex[-1])
            f_e = obj(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                vals[-1] = f_e
            else:
                simplex[-1] = x_r
                vals[-1] = f_r
        elif f_r < vals[-2]:
            simplex[-1] = x_r
            vals[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = obj(x_c)
            if f_c < vals[-1]:
                simplex[-1] = x_c
                vals[-1] = f_c
            else:
                # Shrink
                for i in range(1, 3):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    vals[i] = obj(simplex[i])

        # Update best
        current_best_idx = np.argmin(vals)
        if vals[current_best_idx] < best_val:
            best_val = vals[current_best_idx]
            best_x = simplex[current_best_idx].copy()

    return best_x


def _newton_broyden(F, x0, max_iter=50, tol=1e-12, deadline=None):
    """Newton's method with Broyden's update for Jacobian."""
    x = np.array(x0, dtype=float)
    Fx = _safe_F(F, x)
    if Fx is None:
        return x
    if np.linalg.norm(Fx) <= tol:
        return x

    # Initial Jacobian via finite differences
    J = np.zeros((2, 2))
    h = 1e-7
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_e = _safe_F(F, x + e)
        if Fx_e is None:
            Fx_e = Fx
        J[:, j] = (Fx_e - Fx) / h

    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        if np.linalg.norm(Fx) <= tol:
            break

        try:
            step = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            break

        # Damping: find lambda in (0,1] such that ||F(x - lambda*step)|| < ||F(x)||
        lam = 1.0
        Fx_norm = np.linalg.norm(Fx)
        for _ in range(20):
            x_new = x - lam * step
            Fx_new = _safe_F(F, x_new)
            if Fx_new is not None and np.linalg.norm(Fx_new) < Fx_norm:
                break
            lam *= 0.5
        else:
            # If no damping worked, try a small step
            x_new = x - lam * step
            Fx_new = _safe_F(F, x_new)
            if Fx_new is None:
                break

        # Broyden update
        J = _broyden_update(J, x_new, x, Fx_new, Fx)
        x = x_new
        Fx = Fx_new

    return x


def find_root_system(F, x0):
    """Find root of F: R^2 -> R^2 using hybrid Newton + Nelder-Mead."""
    deadline = time.time() + 110.0  # Safety margin before 120s limit

    x0 = np.array(x0, dtype=float)

    # First try Newton-Broyden
    x_newton = _newton_broyden(F, x0, max_iter=50, tol=1e-12, deadline=deadline)
    Fx_newton = _safe_F(F, x_newton)
    if Fx_newton is not None and np.linalg.norm(Fx_newton) <= 1e-8:
        return x_newton

    # If Newton didn't converge, try Nelder-Mead from x0
    x_nm = _nelder_mead(F, x0, max_iter=300, tol=1e-12, deadline=deadline)
    Fx_nm = _safe_F(F, x_nm)
    if Fx_nm is not None and np.linalg.norm(Fx_nm) <= 1e-8:
        return x_nm

    # Try Newton-Broyden from Nelder-Mead result
    x_hybrid = _newton_broyden(F, x_nm, max_iter=50, tol=1e-12, deadline=deadline)
    Fx_hybrid = _safe_F(F, x_hybrid)
    if Fx_hybrid is not None and np.linalg.norm(Fx_hybrid) <= 1e-8:
        return x_hybrid

    # Return the best result found
    candidates = [x_newton, x_nm, x_hybrid]
    best_x = x_newton
    best_norm = np.inf
    for xc in candidates:
        Fxc = _safe_F(F, xc)
        if Fxc is not None:
            nrm = np.linalg.norm(Fxc)
            if nrm < best_norm:
                best_norm = nrm
                best_x = xc

    return best_x
# EVOLVE-BLOCK-END
