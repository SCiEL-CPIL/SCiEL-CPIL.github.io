"""ODE solver: Robertson problem with adaptive stiff integration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class _CallCounter:
    def __init__(self, f):
        self.f = f
        self.count = 0

    def __call__(self, t, y):
        self.count += 1
        return self.f(t, y)


def _robertson_jacobian(y):
    y0, y1, y2 = y
    a, b, c = 0.04, 3e7, 1e10
    return np.array([
        [-a, c * y2, c * y1],
        [a, -2 * b * y1 - c * y2, -c * y1],
        [0, 2 * b * y1, 0]
    ])


def integrate(f, t_span, y0):
    t0, t1 = t_span
    start_time = time.time()
    deadline = start_time + 280

    counter = _CallCounter(f)

    try:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(
            counter.f,
            [t0, t1],
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13,
            max_step=(t1 - t0) / 50
        )
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback: implicit Euler with analytic Jacobian
    y = np.array(y0, dtype=float)
    t = t0
    h = (t1 - t0) / 500

    for _ in range(50000):
        if t >= t1 - 1e-15:
            break
        if t + h > t1:
            h = t1 - t

        z = y.copy()
        for _ in range(15):
            fz = counter.f(t + h, z)
            J = _robertson_jacobian(z)
            A = np.eye(3) - h * J
            b_vec = y - h * fz
            try:
                step = np.linalg.solve(A, b_vec)
                z_new = z + step
            except np.linalg.LinAlgError:
                z_new = z
            if np.linalg.norm(step) <= 1e-13 * (1 + np.linalg.norm(z_new)):
                break
            z = z_new
        y = z
        t += h

        if time.time() > deadline:
            break

    return y
# EVOLVE-BLOCK-END
