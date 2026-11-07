"""Root finder for 3 fixed 2D systems using Broyden's method with safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with line search safeguards and a deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    x = np.array(x0, dtype=float).flatten()
    if x.shape != (2,):
        x = x.reshape(2)

    # Initial function evaluation
    Fx = np.asarray(F(x), dtype=float).flatten()
    if Fx.shape != (2,):
        Fx = Fx.reshape(2)

    if not np.all(np.isfinite(Fx)):
        # Try a small perturbation
        x = x + np.array([1e-3, -1e-3])
        Fx = np.asarray(F(x), dtype=float).flatten()
        if Fx.shape != (2,):
            Fx = Fx.reshape(2)
        if not np.all(np.isfinite(Fx)):
            return x

    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-12:
        return x

    # Compute initial Jacobian via finite differences
    def compute_jacobian(x, Fx):
        J = np.zeros((2, 2))
        h = 1e-7
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = np.asarray(F(x + e), dtype=float).flatten()
            if Fx_plus.shape != (2,):
                Fx_plus = Fx_plus.reshape(2)
            J[:, j] = (Fx_plus - Fx) / h
        return J

    J = compute_jacobian(x, Fx)

    # Broyden's method
    max_iter = 200
    best_x = x.copy()
    best_norm = norm_Fx

    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Check convergence
        if norm_Fx <= 1e-12:
            return x

        # Try to solve J dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, try a step in the negative gradient direction
            dx = -Fx / max(np.linalg.norm(Fx), 1e-15)

        # Line search to ensure decrease
        alpha = 1.0
        max_line_search = 20
        for ls in range(max_line_search):
            if time.time() > deadline:
                break

            x_new = x + alpha * dx
            Fx_new = np.asarray(F(x_new), dtype=float).flatten()
            if Fx_new.shape != (2,):
                Fx_new = Fx_new.reshape(2)

            if not np.all(np.isfinite(Fx_new)):
                alpha *= 0.5
                continue

            norm_Fx_new = np.linalg.norm(Fx_new)

            if norm_Fx_new < norm_Fx * (1 - 1e-4 * alpha):
                # Accept the step
                # Update Jacobian using Broyden's formula
                if np.linalg.norm(Fx_new - Fx) > 1e-15 and np.linalg.norm(dx) > 1e-15:
                    delta_F = Fx_new - Fx
                    delta_x = x_new - x
                    J = J + (delta_F[:, None] - J @ delta_x[:, None]) / np.dot(delta_x, delta_x)

                x = x_new
                Fx = Fx_new
                norm_Fx = norm_Fx_new

                if norm_Fx < best_norm:
                    best_norm = norm_Fx
                    best_x = x.copy()

                break
            else:
                alpha *= 0.5

        else:
            # Line search failed, try a smaller step
            x_new = x + 0.5 * dx
            Fx_new = np.asarray(F(x_new), dtype=float).flatten()
            if Fx_new.shape != (2,):
                Fx_new = Fx_new.reshape(2)

            if np.all(np.isfinite(Fx_new)):
                norm_Fx_new = np.linalg.norm(Fx_new)
                if norm_Fx_new < norm_Fx:
                    if np.linalg.norm(Fx_new - Fx) > 1e-15 and np.linalg.norm(dx) > 1e-15:
                        delta_F = Fx_new - Fx
                        delta_x = x_new - x
                        J = J + (delta_F[:, None] - J @ delta_x[:, None]) / np.dot(delta_x, delta_x)

                    x = x_new
                    Fx = Fx_new
                    norm_Fx = norm_Fx_new

                    if norm_Fx < best_norm:
                        best_norm = norm_Fx
                        best_x = x.copy()

    return best_x if best_norm <= 1e-8 else x
# EVOLVE-BLOCK-END
