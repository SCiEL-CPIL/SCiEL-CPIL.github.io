"""Root finder for 3 fixed two-dimensional systems using multi-strategy approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2.
    Uses a multi-strategy approach:
    1. Broyden's method with central-difference Jacobian initialization
    2. If Broyden stalls, try damped Newton with fresh Jacobian
    3. If still not converged, try a small grid search
    Returns x with ||F(x)|| <= 1e-8 or best effort.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    x = np.array(x0, dtype=float).copy()
    n = 2

    # Initial evaluation
    Fx = np.array(F(x), dtype=float)
    norm_Fx = float(np.linalg.norm(Fx))

    if norm_Fx <= 1e-8:
        return x

    best_x = x.copy()
    best_norm = norm_Fx

    # Strategy 1: Broyden's method with central-difference Jacobian initialization
    # Initialize Jacobian estimate via central differences (4 calls)
    h = 1e-5
    J = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        Fp = np.array(F(x + e), dtype=float)
        Fm = np.array(F(x - e), dtype=float)
        J[:, j] = (Fp - Fm) / (2 * h)

    max_iter_broyden = 50
    for it in range(max_iter_broyden):
        if time.time() > deadline:
            break

        # Try to solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break

        # Line search to ensure sufficient decrease
        alpha = 1.0
        max_ls = 12
        ls_success = False
        for _ in range(max_ls):
            x_new = x + alpha * dx
            Fx_new = np.array(F(x_new), dtype=float)
            norm_Fx_new = float(np.linalg.norm(Fx_new))

            if norm_Fx_new <= 1e-8:
                return x_new

            if norm_Fx_new < best_norm:
                best_x = x_new.copy()
                best_norm = norm_Fx_new

            # Sufficient decrease condition
            if norm_Fx_new < (1 - 1e-4 * alpha) * norm_Fx:
                # Good-Broyden update
                s = x_new - x
                y = Fx_new - Fx
                s_dot_s = float(np.dot(s, s))
                if s_dot_s > 1e-30:
                    J += np.outer(y - J @ s, s) / s_dot_s
                x = x_new
                Fx = Fx_new
                norm_Fx = norm_Fx_new
                ls_success = True
                break

            alpha *= 0.5

        if not ls_success:
            # Line search failed; break and try other strategies
            break

        if norm_Fx <= 1e-8:
            return x

    # Strategy 2: Damped Newton with fresh Jacobian, multiple restarts
    if time.time() < deadline:
        for restart in range(8):
            if time.time() > deadline:
                break

            # Recompute Jacobian with central differences
            Fx = np.array(F(x), dtype=float)
            norm_Fx = float(np.linalg.norm(Fx))
            if norm_Fx <= 1e-8:
                return x
            if norm_Fx < best_norm:
                best_x = x.copy()
                best_norm = norm_Fx

            # Adaptive step size for Jacobian estimation
            h_jac = 1e-5
            J = np.zeros((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = h_jac
                Fp = np.array(F(x + e), dtype=float)
                Fm = np.array(F(x - e), dtype=float)
                J[:, j] = (Fp - Fm) / (2 * h_jac)

            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    # Perturb x slightly and try again
                    x = x + np.random.normal(0, 0.1, n)
                    continue

            # Damped Newton step
            alpha = 1.0
            max_ls = 15
            newton_success = False
            for _ in range(max_ls):
                x_new = x + alpha * dx
                Fx_new = np.array(F(x_new), dtype=float)
                norm_Fx_new = float(np.linalg.norm(Fx_new))

                if norm_Fx_new <= 1e-8:
                    return x_new

                if norm_Fx_new < best_norm:
                    best_x = x_new.copy()
                    best_norm = norm_Fx_new

                # Armijo condition for damping
                if norm_Fx_new < norm_Fx - 1e-4 * alpha * float(np.dot(dx, J.T @ dx)):
                    x = x_new
                    Fx = Fx_new
                    norm_Fx = norm_Fx_new
                    newton_success = True
                    break

                alpha *= 0.5

            if not newton_success:
                # If Newton step failed, perturb x to escape local issues
                x = x + np.random.normal(0, 0.05, n)

    # Strategy 3: Small grid search around best_x if still not converged
    if time.time() < deadline and best_norm > 1e-8:
        Fx_best = np.array(F(best_x), dtype=float)
        best_norm = float(np.linalg.norm(Fx_best))

        if best_norm > 1e-8:
            # Try a few offsets around best_x
            offsets = [
                [0.01, 0.0], [-0.01, 0.0], [0.0, 0.01], [0.0, -0.01],
                [0.001, 0.0], [-0.001, 0.0], [0.0, 0.001], [0.0, -0.001],
                [0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1],
            ]
            for off in offsets:
                if time.time() > deadline:
                    break
                x_try = best_x + np.array(off)
                Fx_try = np.array(F(x_try), dtype=float)
                norm_try = float(np.linalg.norm(Fx_try))
                if norm_try <= 1e-8:
                    return x_try
                if norm_try < best_norm:
                    best_x = x_try
                    best_norm = norm_try

    return best_x
# EVOLVE-BLOCK-END
