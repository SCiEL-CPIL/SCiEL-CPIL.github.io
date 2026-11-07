"""Root finder for 3 fixed 2D systems using a robust hybrid approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with adaptive step control and fallback strategies.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.array([0.0, 0.0])

    # Count function calls to stay within budget
    call_count = 0
    max_calls = 450  # Leave buffer for safety

    def F_counted(x):
        nonlocal call_count
        call_count += 1
        result = F(x)
        result = np.asarray(result, dtype=float)
        if result.shape != (2,):
            result = result.reshape(2)
        if not np.all(np.isfinite(result)):
            return np.array([1e10, 1e10])
        return result

    # Compute initial F and Jacobian
    Fx = F_counted(x)
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x

    # Finite difference Jacobian
    def compute_jacobian(x, Fx):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = 1e-7
            Fx_e = F_counted(x + e)
            J[:, j] = (Fx_e - Fx) / 1e-7
        return J

    J = compute_jacobian(x, Fx)

    # Broyden's method with adaptive step
    for iteration in range(50):
        if time.time() > deadline:
            break
        if call_count >= max_calls:
            break

        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            return x

        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Fallback: use least squares
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break

        # Adaptive step size
        step = 1.0
        # Try the full step first
        x_new = x + step * dx
        Fx_new = F_counted(x_new)
        norm_Fx_new = np.linalg.norm(Fx_new)

        # If step made things worse, reduce step size
        if norm_Fx_new > norm_Fx:
            for _ in range(10):
                step *= 0.5
                x_new = x + step * dx
                Fx_new = F_counted(x_new)
                norm_Fx_new = np.linalg.norm(Fx_new)
                if norm_Fx_new < norm_Fx:
                    break
            if norm_Fx_new >= norm_Fx:
                # Step is not helping, try a different direction
                # Use steepest descent direction
                try:
                    grad = J.T @ Fx
                    dx = -grad / (np.linalg.norm(grad) + 1e-12)
                    x_new = x + 0.1 * dx
                    Fx_new = F_counted(x_new)
                    norm_Fx_new = np.linalg.norm(Fx_new)
                except:
                    break

        # Update x and Fx
        x = x_new
        Fx = Fx_new

        # Broyden update
        if call_count < max_calls and time.time() < deadline:
            delta_F = Fx - (Fx_prev if 'Fx_prev' in dir() else Fx)
            # Compute delta_F properly
            # We need to track previous Fx
            # Let's restructure to track this

    # Track Fx_prev properly
    # Let's redo with proper tracking
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.array([0.0, 0.0])

    call_count = 0
    max_calls = 450

    def F_counted(x):
        nonlocal call_count
        call_count += 1
        result = F(x)
        result = np.asarray(result, dtype=float)
        if result.shape != (2,):
            result = result.reshape(2)
        if not np.all(np.isfinite(result)):
            return np.array([1e10, 1e10])
        return result

    Fx = F_counted(x)
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x

    J = compute_jacobian(x, Fx)
    Fx_prev = Fx.copy()

    for iteration in range(50):
        if time.time() > deadline:
            break
        if call_count >= max_calls:
            break

        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            return x

        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break

        # Adaptive step size
        step = 1.0
        x_new = x + step * dx
        Fx_new = F_counted(x_new)
        norm_Fx_new = np.linalg.norm(Fx_new)

        if norm_Fx_new > norm_Fx:
            for _ in range(10):
                step *= 0.5
                x_new = x + step * dx
                Fx_new = F_counted(x_new)
                norm_Fx_new = np.linalg.norm(Fx_new)
                if norm_Fx_new < norm_Fx:
                    break
            if norm_Fx_new >= norm_Fx:
                try:
                    grad = J.T @ Fx
                    dx = -grad / (np.linalg.norm(grad) + 1e-12)
                    x_new = x + 0.1 * dx
                    Fx_new = F_counted(x_new)
                    norm_Fx_new = np.linalg.norm(Fx_new)
                except:
                    break

        # Broyden update
        delta_x = x_new - x
        delta_F = Fx_new - Fx
        if np.linalg.norm(delta_x) > 1e-12:
            J = J + (delta_F - J @ delta_x) @ (delta_x.T) / (delta_x @ delta_x)

        x = x_new
        Fx = Fx_new
        Fx_prev = Fx.copy()

    # Final check
    if np.linalg.norm(Fx) <= 1e-8:
        return x

    # Fallback: try a few random restarts
    best_x = x
    best_norm = np.linalg.norm(Fx)

    for _ in range(5):
        if time.time() > deadline or call_count >= max_calls:
            break
        x_try = np.array(x0, dtype=float) + np.random.normal(0, 0.1, 2)
        Fx_try = F_counted(x_try)
        norm_try = np.linalg.norm(Fx_try)
        if norm_try < best_norm:
            best_x = x_try
            best_norm = norm_try

    return best_x
# EVOLVE-BLOCK-END
