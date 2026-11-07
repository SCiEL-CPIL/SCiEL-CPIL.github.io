"""Root finder for 3 fixed 2D systems using hybrid Nelder-Mead + Broyden's method."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def time_left():
        return deadline - time.time()

    def norm_f(x):
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(-1)
                if Fx.size != 2:
                    return np.inf
            n = np.linalg.norm(Fx)
            return n if np.isfinite(n) else np.inf
        except Exception:
            return np.inf

    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = norm_f(best_x)

    # Phase 1: Nelder-Mead on ||F(x)||^2 to find a basin
    def merit(x):
        n = norm_f(x)
        return n * n if np.isfinite(n) else 1e20

    # Initial simplex
    scale = max(0.1, np.linalg.norm(x))
    simplex = [
        x.copy(),
        x + np.array([scale, 0.0]),
        x + np.array([0.0, scale]),
    ]
    fvals = [merit(v) for v in simplex]

    # Nelder-Mead parameters
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    max_nm_iter = 200

    for _ in range(max_nm_iter):
        if time_left() < 5.0:
            break
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        if fvals[0] < best_norm * best_norm:
            best_norm = np.sqrt(fvals[0])
            best_x = simplex[0].copy()

        # Check convergence
        if fvals[0] < 1e-16:
            break
        if abs(fvals[-1] - fvals[0]) < 1e-12:
            break

        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]

        # Reflection
        xr = centroid + alpha * (centroid - worst)
        fr = merit(xr)
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - worst)
            fe = merit(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + rho * (worst - centroid)
            fc = merit(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, 3):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = merit(simplex[i])

    # Update best
    if merit(best_x) < best_norm * best_norm:
        best_norm = np.sqrt(merit(best_x))

    # Phase 2: Broyden's method with adaptive step control
    if time_left() > 3.0:
        x = best_x.copy()
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float).reshape(2)

        # Initial Jacobian via finite differences
        h = 1e-6
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - Fx) / h

        max_broyden_iter = 100
        for _ in range(max_broyden_iter):
            if time_left() < 2.0:
                break

            nF = np.linalg.norm(Fx)
            if nF < best_norm:
                best_norm = nF
                best_x = x.copy()

            if nF <= 1e-12:
                return x

            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Fallback: steepest descent step
                dx = -Fx * 0.01

            # Adaptive step control
            step_size = 1.0
            for _ in range(20):
                x_new = x + step_size * dx
                Fx_new = F(x_new)
                Fx_new = np.asarray(Fx_new, dtype=float).reshape(2)
                nF_new = np.linalg.norm(Fx_new)

                if np.isfinite(nF_new) and nF_new < nF:
                    break
                step_size *= 0.5

            if not np.isfinite(nF_new) or nF_new >= nF:
                # Try smaller step or break
                if step_size < 1e-10:
                    break
                x_new = x + step_size * dx
                Fx_new = F(x_new)
                Fx_new = np.asarray(Fx_new, dtype=float).reshape(2)
                nF_new = np.linalg.norm(Fx_new)
                if not np.isfinite(nF_new):
                    break

            # Broyden rank-1 update
            s = x_new - x
            y = Fx_new - Fx
            if np.linalg.norm(s) > 1e-14:
                J = J + np.outer(y - J @ s, s) / (s @ s)

            x = x_new
            Fx = Fx_new

            if nF_new < best_norm:
                best_norm = nF_new
                best_x = x.copy()

    # Final check
    final_norm = norm_f(best_x)
    if final_norm < best_norm:
        best_norm = final_norm

    if best_norm <= 1e-8:
        return best_x

    # If we didn't converge, return best attempt
    return best_x
# EVOLVE-BLOCK-END
