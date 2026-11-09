"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with rank-1 Jacobian updates for efficiency,
    with adaptive damping and fallback to damped Newton if needed.
    """
    x = np.array(x0, dtype=float)
    deadline = time.time() + 110  # 10s safety margin

    # Initial function evaluation
    Fx = np.asarray(F(x), dtype=float)
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-12:
        return x

    # Build initial Jacobian using forward differences
    h = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_pert = np.asarray(F(x + e), dtype=float)
        J[:, j] = (Fx_pert - Fx) / h

    # Broyden's method with adaptive damping
    best_x = x.copy()
    best_norm = norm_Fx

    for it in range(400):
        if time.time() > deadline:
            break

        # Try to solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse or damped step
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                dx = -Fx * 0.1

        # Adaptive damping based on step size
        norm_dx = np.linalg.norm(dx)
        if norm_dx == 0:
            break

        # Limit step size to prevent divergence
        max_step = 1.0
        if norm_dx > max_step:
            dx = dx * (max_step / norm_dx)
            norm_dx = max_step

        # Try full step first
        x_trial = x + dx
        Fx_trial = np.asarray(F(x_trial), dtype=float)
        norm_trial = np.linalg.norm(Fx_trial)

        if not np.all(np.isfinite(Fx_trial)):
            # Step led to invalid values, try smaller step
            for damp in [0.5, 0.1, 0.01]:
                x_trial = x + damp * dx
                Fx_trial = np.asarray(F(x_trial), dtype=float)
                norm_trial = np.linalg.norm(Fx_trial)
                if np.all(np.isfinite(Fx_trial)):
                    break
            if not np.all(np.isfinite(Fx_trial)):
                # All steps failed, break
                break

        if norm_trial < best_norm:
            # Update best
            best_x = x_trial.copy()
            best_norm = norm_trial

            if norm_trial <= 1e-12:
                return x_trial

            # Broyden rank-1 update
            s = x_trial - x
            y = Fx_trial - Fx
            # Check if update is meaningful
            if np.linalg.norm(s) > 1e-14 and np.linalg.norm(y) > 1e-14:
                J = J + np.outer(y - J @ s, s) / (s @ s)

            x = x_trial
            Fx = Fx_trial
            norm_Fx = norm_trial
        else:
            # Full step didn't improve, try damped steps
            improved = False
            for damp in [0.5, 0.25, 0.1, 0.05, 0.01]:
                x_trial = x + damp * dx
                Fx_trial = np.asarray(F(x_trial), dtype=float)
                norm_trial = np.linalg.norm(Fx_trial)

                if not np.all(np.isfinite(Fx_trial)):
                    continue

                if norm_trial < norm_Fx:
                    # Accept this step
                    s = x_trial - x
                    y = Fx_trial - Fx
                    if np.linalg.norm(s) > 1e-14 and np.linalg.norm(y) > 1e-14:
                        J = J + np.outer(y - J @ s, s) / (s @ s)

                    x = x_trial
                    Fx = Fx_trial
                    norm_Fx = norm_trial

                    if norm_Fx < best_norm:
                        best_x = x.copy()
                        best_norm = norm_Fx

                    if norm_Fx <= 1e-12:
                        return x

                    improved = True
                    break

            if not improved:
                # No improvement found, try rebuilding Jacobian
                h = 1e-7
                J_new = np.zeros((2, 2))
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h
                    Fx_pert = np.asarray(F(x + e), dtype=float)
                    J_new[:, j] = (Fx_pert - Fx) / h
                J = J_new

                # If still stuck, try a small random perturbation
                if it > 50:
                    rng = np.random.RandomState(it)
                    perturbation = rng.randn(2) * 0.1
                    x_trial = x + perturbation
                    Fx_trial = np.asarray(F(x_trial), dtype=float)
                    if np.all(np.isfinite(Fx_trial)):
                        norm_trial = np.linalg.norm(Fx_trial)
                        if norm_trial < norm_Fx:
                            x = x_trial
                            Fx = Fx_trial
                            norm_Fx = norm_trial
                            if norm_Fx < best_norm:
                                best_x = x.copy()
                                best_norm = norm_Fx
                            # Rebuild Jacobian
                            h = 1e-7
                            J_new = np.zeros((2, 2))
                            for j in range(2):
                                e = np.zeros(2)
                                e[j] = h
                                Fx_pert = np.asarray(F(x + e), dtype=float)
                                J_new[:, j] = (Fx_pert - Fx) / h
                            J = J_new
                        else:
                            # Reset to best
                            x = best_x.copy()
                            Fx = np.asarray(F(x), dtype=float)
                            norm_Fx = np.linalg.norm(Fx)
                            h = 1e-7
                            J_new = np.zeros((2, 2))
                            for j in range(2):
                                e = np.zeros(2)
                                e[j] = h
                                Fx_pert = np.asarray(F(x + e), dtype=float)
                                J_new[:, j] = (Fx_pert - Fx) / h
                            J = J_new

    # Final check
    if best_norm <= 1e-8:
        return best_x

    # Try a few more Newton steps from best_x with tighter tolerance
    x = best_x.copy()
    Fx = np.asarray(F(x), dtype=float)
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-12:
        return x

    h = 1e-8
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_pert = np.asarray(F(x + e), dtype=float)
        J[:, j] = (Fx_pert - Fx) / h

    for _ in range(20):
        if time.time() > deadline:
            break
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
        x = x + dx
        Fx = np.asarray(F(x), dtype=float)
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-12:
            return x
        h = 1e-8
        J_new = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_pert = np.asarray(F(x + e), dtype=float)
            J_new[:, j] = (Fx_pert - Fx) / h
        J = J_new

    return x
# EVOLVE-BLOCK-END
