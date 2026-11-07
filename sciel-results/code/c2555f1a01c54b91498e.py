import numpy as np
import time


def find_root_system(F, x0):
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = float('inf')
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    calls = 0
    
    def safe_F(x):
        nonlocal calls
        if time.time() > deadline:
            raise TimeoutError
        if calls >= 495:
            raise RuntimeError("Call limit exceeded")
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return None
            calls += 1
            return Fx
        except Exception:
            return None

    # Initial evaluation
    Fx = safe_F(x)
    if Fx is None:
        return best_x
    norm = np.linalg.norm(Fx)
    best_norm = norm
    best_x = x.copy()
    if norm <= 1e-8:
        return best_x

    # Compute initial Jacobian using finite differences
    h = 1e-6
    J = np.zeros((2, 2))
    jacobian_valid = True
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_j = safe_F(x + e)
        if Fx_j is None:
            jacobian_valid = False
            break
        J[:, j] = (Fx_j - Fx) / h

    if not jacobian_valid:
        J = np.eye(2)

    # Broyden's method with backtracking line search
    max_iter = 200
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if calls >= 490:
            break

        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break

        if not np.all(np.isfinite(dx)):
            break

        # Backtracking line search
        alpha = 1.0
        Fx_old_norm = norm
        step_accepted = False

        for _ in range(20):
            x_try = x + alpha * dx
            Fx_try = safe_F(x_try)
            if Fx_try is None:
                alpha *= 0.5
                continue

            norm_try = np.linalg.norm(Fx_try)

            if norm_try < best_norm:
                best_norm = norm_try
                best_x = x_try.copy()

            # Accept if any decrease
            if norm_try < Fx_old_norm:
                # Update Broyden's Jacobian estimate
                s = alpha * dx
                y = Fx_try - Fx
                J_s = J @ s
                diff = y - J_s
                denom = s @ s
                if denom > 1e-15:
                    J = J + np.outer(diff, s) / denom
                else:
                    J = np.eye(2)

                x = x_try
                Fx = Fx_try
                norm = norm_try
                step_accepted = True
                break
            else:
                alpha *= 0.5

        if not step_accepted:
            break

        if norm <= 1e-8:
            return best_x

    return best_x
