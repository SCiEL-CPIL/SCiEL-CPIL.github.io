# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    x0 = np.asarray(x0, dtype=float)
    x0 = x0.reshape(2)
    start = time.time()
    DEADLINE = 110.0          # safety margin before the 120 s hard kill
    TOL = 1e-8
    best = x0.copy()
    best_n = float('inf')

    def now_ok():
        return (time.time() - start) < DEADLINE

    def call_guarded(x):
        return F(x)

    def g_of(x, t):
        return t * F(x) + (1.0 - t) * (x - x0)

    def jacobian(G, x, h=1e-5):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (G(x + e) - G(x)) / h
        return J

    def norm2(v):
        return float(v @ v)

    def update_best(x, n):
        nonlocal best, best_n
        if np.all(np.isfinite(x)) and n < best_n:
            best = x.copy()
            best_n = n

    def damped_newton(G, x, tol, max_iter=80):
        x = np.array(x, dtype=float)
        Gx = G(x)
        n = norm2(Gx)
        if n <= tol:
            return x, n
        for _ in range(max_iter):
            if not now_ok():
                return x, n
            # try to solve J d = -Gx
            J = jacobian(G, x)
            try:
                d = np.linalg.solve(J, -Gx)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(d)):
                break
            # damping: find alpha in (0,1] that reduces ||G||^2
            Gx_cur = Gx
            f_cur = n
            alpha = 1.0
            ok = False
            for _ls in range(30):
                xn = x + alpha * d
                if not np.all(np.isfinite(xn)):
                    alpha *= 0.5
                    continue
                Gn = G(xn)
                fn = norm2(Gn)
                if not np.isfinite(fn):
                    alpha *= 0.5
                    continue
                if fn < f_cur * (1.0 - 1e-4 * alpha):
                    x = xn
                    Gx = Gn
                    n = fn
                    ok = True
                    break
                alpha *= 0.5
            if not ok:
                # bisection on alpha to guarantee descent
                lo, hi = 0.0, 1.0
                found = False
                for _b in range(40):
                    mid = 0.5 * (lo + hi)
                    xn = x + mid * d
                    if not np.all(np.isfinite(xn)):
                        hi = mid
                        continue
                    Gn = G(xn)
                    fn = norm2(Gn)
                    if np.isfinite(fn) and fn < f_cur:
                        lo = mid
                        found = True
                    else:
                        hi = mid
                if found:
                    x = x + lo * d
                    Gx = G(x)
                    n = norm2(Gx)
                else:
                    break
            if n <= tol:
                return x, n
        return x, n

    # --- Homotopy continuation ---
    def run_homotopy(t_start, t_end, n_steps):
        nonlocal best, best_n
        x = x0.copy()
        # ensure a valid starting point for t_start
        G = lambda z: g_of(z, t_start)
        Gx = G(x)
        n = norm2(Gx)
        update_best(x, n)
        ts = np.linspace(t_start, t_end, n_steps)
        for t in ts:
            if not now_ok():
                return x
            G = (lambda tt: (lambda z: tt * F(z) + (1.0 - tt) * (z - x0)))(t)
            x, n = damped_newton(G, x, TOL)
            update_best(x, n)
        return x

    run_homotopy(0.0, 1.0, 6)
    if best_n <= TOL:
        return best

    # --- Multi-start fallback with damped Newton on F directly ---
    offsets = [
        np.array([0.5, 0.5]), np.array([-0.5, 0.5]),
        np.array([0.5, -0.5]), np.array([-0.5, -0.5]),
        np.array([1.0, 0.0]), np.array([0.0, 1.0]),
        np.array([-1.0, 0.0]), np.array([0.0, -1.0]),
    ]
    for off in offsets:
        if not now_ok():
            break
        xs = x0 + off
        G = F
        xs, n = damped_newton(G, xs, TOL)
        update_best(xs, n)
        if best_n <= TOL:
            return best

    return best
# EVOLVE-BLOCK-END
