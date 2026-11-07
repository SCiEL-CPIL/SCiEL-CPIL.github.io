# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using a trust-region dogleg method
    with Broyden-updated Jacobian approximation, followed by a Broyden-Newton polish.
    """
    x = np.array(x0, dtype=float, copy=True)
    n = 2
    max_total_calls = 480
    call_count = [0]

    def fcounted(z):
        call_count[0] += 1
        out = F(z)
        out = np.asarray(out, dtype=float)
        return out

    # --- Initial evaluation ---
    Fx = fcounted(x)
    if not np.all(np.isfinite(Fx)):
        return x
    fval = np.dot(Fx, Fx)
    if fval <= (1e-8) ** 2:
        return x

    # --- Initial Jacobian via finite differences ---
    def finite_diff_J(z, Fz, h=1e-6):
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            Fp = fcounted(z + e)
            J[:, j] = (Fp - Fz) / h
        return J

    J = finite_diff_J(x, Fx)
    if not np.all(np.isfinite(J)):
        J = finite_diff_J(x, Fx, h=1e-4)
    if not np.all(np.isfinite(J)):
        J = np.eye(n)

    # --- Trust Region Parameters ---
    delta = 1.0
    delta_min = 1e-14
    delta_max = 1e6
    eta = 0.1

    # Broyden update helper (inverse Broyden for better conditioning)
    def broyden_update(J, Fx_old, Fx_new, s):
        y = Fx_new - Fx_old
        Js = J @ s
        diff = y - Js
        sTs = s @ s
        if sTs > 1e-30:
            J = J + np.outer(diff, s) / sTs
        return J

    # --- Phase 1: Trust Region Dogleg ---
    for iteration in range(80):
        if call_count[0] >= max_total_calls:
            break

        # Check convergence using current Fx
        fval = np.dot(Fx, Fx)
        if fval <= (1e-8) ** 2:
            return x

        # Gradient and Gauss-Newton Hessian
        g = J.T @ Fx
        H = J.T @ J

        # Solve H p = -g
        try:
            p_gn = np.linalg.solve(H, -g)
            if not np.all(np.isfinite(p_gn)):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            try:
                p_gn = np.linalg.lstsq(H, -g, rcond=None)[0]
            except Exception:
                p_gn = -g / (np.linalg.norm(g) + 1e-30)

        # Steepest Descent
        g_norm = np.linalg.norm(g)
        if g_norm > 1e-30:
            p_sd = -g / g_norm
        else:
            p_sd = np.zeros(n)

        # Dogleg Step
        gn_norm = np.linalg.norm(p_gn)
        if gn_norm <= delta:
            p = p_gn
        else:
            # Project p_gn onto p_sd direction
            if g_norm > 1e-30:
                t = np.dot(p_gn, p_sd)
                # t is the scalar projection length since p_sd is unit vector
                corner = t * p_sd
                v = p_gn - corner
                v_norm = np.linalg.norm(v)
                corner_norm = np.linalg.norm(corner)
                remaining = delta - corner_norm
                if remaining <= 0:
                    p = p_sd * delta
                else:
                    p = corner + v * (remaining / v_norm)
            else:
                p = p_gn * (delta / gn_norm)

        # Predicted Reduction
        pred = fval + np.dot(g, p) + 0.5 * np.dot(p, H @ p)
        
        # Try Step
        x_new = x + p
        Fx_new = fcounted(x_new)
        if not np.all(np.isfinite(Fx_new)):
            delta *= 0.5
            continue

        fval_new = np.dot(Fx_new, Fx_new)
        actual = fval - fval_new
        predicted = fval - pred

        if predicted <= 0:
            rho = -1.0
        else:
            rho = actual / predicted

        if rho > eta:
            # Accept
            s = p
            J = broyden_update(J, Fx, Fx_new, s)
            x = x_new
            Fx = Fx_new
            fval = fval_new
            
            if rho > 0.75:
                delta = min(2.0 * delta, delta_max)
            else:
                delta = min(1.5 * delta, delta_max)
        else:
            # Reject
            delta *= 0.5
            if delta < delta_min:
                # Escape strategy with random restart
                x = x + 1e-3 * np.random.randn(n)
                Fx = fcounted(x)
                delta = 1e-2

    # --- Phase 2: Newton Polish with Broyden Jacobian ---
    # Use the current J estimate. If it fails, fallback to finite diff.
    for _ in range(30):
        if call_count[0] >= max_total_calls:
            break
        
        fval = np.dot(Fx, Fx)
        if fval <= (1e-8) ** 2:
            return x

        # Try Newton step with current J
        try:
            dx = np.linalg.solve(J, -Fx)
            if not np.all(np.isfinite(dx)):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            # If J is singular or bad, recompute via finite diff
            J = finite_diff_J(x, Fx)
            if not np.all(np.isfinite(J)):
                break
            try:
                dx = np.linalg.solve(J, -Fx)
                if not np.all(np.isfinite(dx)):
                    break
            except np.linalg.LinAlgError:
                break

        x_new = x + dx
        Fx_new = fcounted(x_new)
        if not np.all(np.isfinite(Fx_new)):
            break
        
        fval_new = np.dot(Fx_new, Fx_new)
        
        if fval_new < fval:
            # Accept and update J
            s = dx
            J = broyden_update(J, Fx, Fx_new, s)
            x = x_new
            Fx = Fx_new
        else:
            # Damped step
            accepted = False
            for t in [0.5, 0.25, 0.1, 0.05]:
                x_trial = x + t * dx
                Fx_trial = fcounted(x_trial)
                if np.all(np.isfinite(Fx_trial)):
                    fval_trial = np.dot(Fx_trial, Fx_trial)
                    if fval_trial < fval:
                        s = t * dx
                        J = broyden_update(J, Fx, Fx_trial, s)
                        x = x_trial
                        Fx = Fx_trial
                        fval = fval_trial
                        accepted = True
                        break
            if not accepted:
                break

    return x
# EVOLVE-BLOCK-END
