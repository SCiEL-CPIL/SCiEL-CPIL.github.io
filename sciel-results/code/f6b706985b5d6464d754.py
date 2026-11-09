"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 such that ||F(x)|| <= 1e-8.
    Uses a hybrid approach:
    1. Try Broyden's method with a finite-difference Jacobian update.
    2. If that fails or stagnates, fall back to a robust derivative-free
       optimization of ||F(x)||^2 using Nelder-Mead.
    3. Polish with a few Newton steps using central differences.
    """
    x = np.array(x0, dtype=float)
    deadline = time.time() + 110  # 10s safety margin

    def _norm(Fx):
        return np.linalg.norm(Fx)

    def _safe_F(x):
        try:
            fx = F(x)
            fx = np.asarray(fx, dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return None
            return fx
        except Exception:
            return None

    # Check initial point
    Fx = _safe_F(x)
    if Fx is None:
        return x
    if _norm(Fx) <= 1e-8:
        return x

    # --- Phase 1: Broyden's Good Method ---
    def _jacobian_fd(x, Fx, h=1e-7):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            fx_plus = _safe_F(x + e)
            if fx_plus is None:
                return None
            J[:, j] = (fx_plus - Fx) / h
        return J

    J = _jacobian_fd(x, Fx)
    if J is None:
        J = np.eye(2)

    for it in range(80):
        if time.time() > deadline:
            break

        Fx = _safe_F(x)
        if Fx is None:
            break
        if _norm(Fx) <= 1e-8:
            return x

        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Singular Jacobian, try pseudo-inverse
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break

        # Damped step if needed
        x_new = x + dx
        Fx_new = _safe_F(x_new)
        if Fx_new is None:
            break

        # Accept step if it reduces ||F||
        if _norm(Fx_new) < _norm(Fx):
            # Update Jacobian (Broyden's Good)
            s = x_new - x
            y = Fx_new - Fx
            J = J + np.outer(y - J @ s, s) / (s @ s) if s @ s > 1e-16 else J
            x = x_new
            Fx = Fx_new
        else:
            # Halve the step
            dx = dx * 0.5
            x_new = x + dx
            Fx_new = _safe_F(x_new)
            if Fx_new is None or _norm(Fx_new) >= _norm(Fx):
                break
            s = x_new - x
            y = Fx_new - Fx
            J = J + np.outer(y - J @ s, s) / (s @ s) if s @ s > 1e-16 else J
            x = x_new
            Fx = Fx_new

    if _norm(Fx) <= 1e-8:
        return x

    # --- Phase 2: Nelder-Mead on ||F(x)||^2 ---
    def objective(z):
        fx = _safe_F(z)
        if fx is None:
            return 1e10
        return float(np.dot(fx, fx))

    # Initialize simplex
    x_best = x.copy()
    f_best = objective(x_best)
    simplex = [x_best.copy()]
    for i in range(2):
        xi = x_best.copy()
        xi[i] += 0.1 * (abs(x_best[i]) + 1.0)
        simplex.append(xi)

    # Nelder-Mead parameters
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    for it in range(200):
        if time.time() > deadline:
            break

        # Sort simplex by objective
        objs = [objective(v) for v in simplex]
        order = np.argsort(objs)
        simplex = [simplex[i] for i in order]
        objs = [objs[i] for i in order]

        f_best = objs[0]
        x_best = simplex[0].copy()

        if np.sqrt(f_best) <= 1e-8:
            return x_best

        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)

        if fr < objs[-2]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
            else:
                simplex[-1] = xr
        elif fr < objs[-1]:
            simplex[-1] = xr
        else:
            # Contraction
            if fr < objs[-1]:
                xc = centroid + rho * (simplex[-1] - centroid)
            else:
                xc = centroid - rho * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < min(fr, objs[-1]):
                simplex[-1] = xc
            else:
                # Shrink
                for i in range(1, 3):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])

    if np.sqrt(objective(x_best)) <= 1e-8:
        return x_best

    # --- Phase 3: Polish with Newton using central differences ---
    x = x_best.copy()
    for it in range(30):
        if time.time() > deadline:
            break

        Fx = _safe_F(x)
        if Fx is None:
            break
        if _norm(Fx) <= 1e-8:
            return x

        # Central difference Jacobian
        h = 1e-6
        J = np.zeros((2, 2))
        ok = True
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            fx_p = _safe_F(x + e)
            fx_m = _safe_F(x - e)
            if fx_p is None or fx_m is None:
                ok = False
                break
            J[:, j] = (fx_p - fx_m) / (2 * h)
        if not ok:
            break

        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break

        x_new = x + dx
        Fx_new = _safe_F(x_new)
        if Fx_new is None:
            break
        if _norm(Fx_new) < _norm(Fx):
            x = x_new
            Fx = Fx_new
        else:
            break

    return x
# EVOLVE-BLOCK-END
