import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start = time.time()
    deadline = start + 105.0  # 120s limit, 15s margin

    def overlaps(g):
        f = np.empty(N)
        one_minus_g = 1.0 - g
        for k in range(N):
            f[k] = np.dot(g[:N - k], one_minus_g[k:])
        return f

    def raw_value(g):
        return float(np.max(overlaps(g)) * (2.0 / N))

    def project(g):
        target = N / 2.0
        lo, hi = -1.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            gp = np.clip(g + mid, 0.0, 1.0)
            if np.sum(gp) > target:
                lo = mid
            else:
                hi = mid
        gp = np.clip(g + (lo + hi) / 2.0, 0.0, 1.0)
        # Fine-tune free variables to hit sum exactly
        for _ in range(10):
            diff = target - np.sum(gp)
            if abs(diff) < 1e-12:
                break
            free = (gp > 1e-12) & (gp < 1.0 - 1e-12)
            if not np.any(free):
                break
            gp[free] += diff / np.sum(free)
            gp = np.clip(gp, 0.0, 1.0)
        return gp

    def soft_max(f, tau):
        m = np.max(f)
        return m + tau * np.log(np.sum(np.exp((f - m) / tau)))

    best_g = np.full(N, 0.5)
    best_val = raw_value(best_g)

    x = np.arange(N)
    inits = [
        np.full(N, 0.5),
        0.5 + 0.3 * np.cos(2 * np.pi * x / N),
        0.5 + 0.4 * np.cos(2 * np.pi * x / N),
        0.5 + 0.3 * np.cos(2 * np.pi * 2 * x / N),
        0.5 + 0.2 * np.cos(2 * np.pi * x / N) + 0.2 * np.sin(2 * np.pi * x / N),
        0.5 + 0.25 * np.cos(2 * np.pi * x / N) + 0.15 * np.cos(2 * np.pi * 3 * x / N),
    ]

    taus = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02]
    penalty = 100.0

    for init in inits:
        if time.time() > deadline:
            break
        g0 = project(init)
        g = g0
        for tau in taus:
            if time.time() > deadline:
                break
            def obj(gg, tau=tau):
                f = overlaps(gg)
                return soft_max(f, tau) * (2.0 / N) + penalty * (np.sum(gg) - N / 2.0) ** 2
            try:
                res = minimize(obj, g, method='L-BFGS-B',
                               bounds=[(0.0, 1.0)] * N,
                               options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12})
                g = res.x
            except Exception:
                pass
        g = project(g)
        val = raw_value(g)
        if val < best_val:
            best_val = val
            best_g = g

    # Polish step: reduce the active max
    for _ in range(100):
        if time.time() > deadline:
            break
        f = overlaps(best_g)
        k_max = int(np.argmax(f))
        grad = np.zeros(N)
        mask1 = np.arange(N) < (N - k_max)
        grad[mask1] += 1.0
        grad[mask1] -= best_g[mask1 + k_max]
        mask2 = np.arange(N) >= k_max
        grad[mask2] -= best_g[mask2 - k_max]
        for step in [0.01, 0.005, 0.001, 0.0005]:
            g_new = project(best_g - step * grad)
            v = raw_value(g_new)
            if v < best_val:
                best_val = v
                best_g = g_new
                break

    best_g = project(best_g)
    return best_g
