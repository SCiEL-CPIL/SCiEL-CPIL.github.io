"""Root finder for 3 fixed 2D systems using Broyden's method with robust restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def time_left():
        return deadline - time.time()

    def norm_f(x):
        return np.linalg.norm(F(x))

    def compute_jacobian(x, Fx):
        """Central difference Jacobian (4 function calls)."""
        h = 1e-5
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - F(x - e)) / (2 * h)
        return J

    def compute_jacobian_forward(x, Fx):
        """Forward difference Jacobian (2 function calls, reuses Fx)."""
        h = 1e-5
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - Fx) / h
        return J

    def try_broyden(x, max_iter=40):
        """Broyden's method with line search. Returns (x, success, calls_used)."""
        calls = 0
        Fx = F(x)
        calls += 1
        fnorm = np.linalg.norm(Fx)
        if fnorm <= 1e-12:
            return x, True, calls

        # Initial Jacobian
        J = compute_jacobian_forward(x, Fx)
        calls += 2  # 2 extra calls for the two F(x+e) evaluations

        for _ in range(max_iter):
            if time_left() < 2.0:
                break
            if fnorm <= 1e-12:
                return x, True, calls

            # Solve J dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Fallback: use pseudo-inverse
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except Exception:
                    break

            # Line search with backtracking
            alpha = 1.0
            step_accepted = False
            for _ls in range(20):
                if time_left() < 2.0:
                    break
                x_new = x + alpha * dx
                Fx_new = F(x_new)
                calls += 1
                fnorm_new = np.linalg.norm(Fx_new)
                if fnorm_new < fnorm:
                    # Broyden update
                    s = alpha * dx
                    y = Fx_new - Fx
                    # Broyden update: J_new = J + (y - J s) s^T / (s^T s)
                    J = J + (np.outer(y - J @ s, s)) / (s @ s + 1e-30)
                    x = x_new
                    Fx = Fx_new
                    fnorm = fnorm_new
                    step_accepted = True
                    break
                alpha *= 0.5

            if not step_accepted:
                # If line search failed, try smaller step or break
                break

        return x, fnorm <= 1e-12, calls

    def try_newton(x, max_iter=30):
        """Full Newton with line search. Returns (x, success, calls_used)."""
        calls = 0
        Fx = F(x)
        calls += 1
        fnorm = np.linalg.norm(Fx)
        if fnorm <= 1e-12:
            return x, True, calls

        for _ in range(max_iter):
            if time_left() < 2.0:
                break
            if fnorm <= 1e-12:
                return x, True, calls

            J = compute_jacobian_forward(x, Fx)
            calls += 2

            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except Exception:
                    break

            # Line search
            alpha = 1.0
            step_accepted = False
            for _ls in range(20):
                if time_left() < 2.0:
                    break
                x_new = x + alpha * dx
                Fx_new = F(x_new)
                calls += 1
                fnorm_new = np.linalg.norm(Fx_new)
                if fnorm_new < fnorm:
                    x = x_new
                    Fx = Fx_new
                    fnorm = fnorm_new
                    step_accepted = True
                    break
                alpha *= 0.5

            if not step_accepted:
                break

        return x, fnorm <= 1e-12, calls

    # Strategy: try Broyden from x0, then Newton from x0, then restarts
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = np.inf
    total_calls = 0

    # Phase 1: Broyden from x0
    x1, ok1, calls1 = try_broyden(x, max_iter=40)
    total_calls += calls1
    n1 = np.linalg.norm(F(x1))
    total_calls += 1
    if n1 < best_norm:
        best_norm = n1
        best_x = x1.copy()
    if ok1 or n1 <= 1e-10:
        return best_x

    # Phase 2: Newton from x0
    x2, ok2, calls2 = try_newton(x, max_iter=30)
    total_calls += calls2
    n2 = np.linalg.norm(F(x2))
    total_calls += 1
    if n2 < best_norm:
        best_norm = n2
        best_x = x2.copy()
    if ok2 or n2 <= 1e-10:
        return best_x

    # Phase 3: Restart from perturbed points
    rng = np.random.RandomState(42)
    for attempt in range(10):
        if time_left() < 5.0:
            break

        # Generate a perturbed starting point
        scale = 1.0 * (attempt + 1)
        perturbation = rng.randn(2) * scale
        x_start = x + perturbation

        # Try Broyden from this point
        x3, ok3, calls3 = try_broyden(x_start, max_iter=30)
        total_calls += calls3
        n3 = np.linalg.norm(F(x3))
        total_calls += 1
        if n3 < best_norm:
            best_norm = n3
            best_x = x3.copy()
        if ok3 or n3 <= 1e-10:
            return best_x

        # If Broyden didn't work well, try Newton
        if n3 > 1e-6:
            x4, ok4, calls4 = try_newton(x_start, max_iter=20)
            total_calls += calls4
            n4 = np.linalg.norm(F(x4))
            total_calls += 1
            if n4 < best_norm:
                best_norm = n4
                best_x = x4.copy()
            if ok4 or n4 <= 1e-10:
                return best_x

    # Final check: if best is good enough, return it
    if best_norm <= 1e-8:
        return best_x

    # Last resort: try a few more Newton iterations from best_x
    if time_left() > 5.0:
        x5, ok5, calls5 = try_newton(best_x, max_iter=20)
        total_calls += calls5
        n5 = np.linalg.norm(F(x5))
        total_calls += 1
        if n5 < best_norm:
            best_norm = n5
            best_x = x5.copy()

    return best_x
# EVOLVE-BLOCK-END
