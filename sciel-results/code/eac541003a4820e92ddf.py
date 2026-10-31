import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    deadline = time.time() + 110

    def make_constraints():
        cons = []
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[1 + 2 * i] - x[1 + 2 * N + i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[0] - x[1 + 2 * i] - x[1 + 2 * N + i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 + 2 * i] - x[1 + 2 * N + i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 2 - x[0] - x[2 + 2 * i] - x[1 + 2 * N + i]})
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: (x[1 + 2 * i] - x[1 + 2 * j]) ** 2 +
                                                (x[2 + 2 * i] - x[2 + 2 * j]) ** 2 -
                                                (x[1 + 2 * N + i] + x[1 + 2 * N + j]) ** 2
                })
        return cons

    cons = make_constraints()
    bounds = [(0.1, 1.9)] + [(0.001, 1.999)] * (2 * N) + [(0.001, 1.999)] * N

    def obj(x):
        return -x[1 + 2 * N:].sum()

    def check_feasible(x, tol=1e-4):
        w = x[0]
        h = 2 - w
        centers = x[1:1 + 2 * N].reshape(N, 2)
        radii = x[1 + 2 * N:]
        for i in range(N):
            if radii[i] < -tol:
                return False
            if centers[i, 0] < radii[i] - tol or centers[i, 0] > w - radii[i] + tol:
                return False
            if centers[i, 1] < radii[i] - tol or centers[i, 1] > h - radii[i] + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                d2 = (centers[i, 0] - centers[j, 0]) ** 2 + (centers[i, 1] - centers[j, 1]) ** 2
                if d2 < (radii[i] + radii[j]) ** 2 - tol:
                    return False
        return True

    best_value = -np.inf
    best_x = None

    def try_solve(x0, maxiter=150):
        nonlocal best_value, best_x
        if time.time() > deadline:
            return
        try:
            res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': maxiter, 'ftol': 1e-8})
            if res.success and check_feasible(res.x):
                val = -res.fun
                if val > best_value:
                    best_value = val
                    best_x = res.x
        except Exception:
            pass

    # Grid initializations
    for w_init in [0.6, 0.8, 1.0, 1.2, 1.4]:
        h_init = 2 - w_init
        for cols in [5, 6]:
            for rows in [4, 5]:
                if cols * rows < N:
                    continue
                xs = np.linspace(0.1 * w_init, 0.9 * w_init, cols)
                ys = np.linspace(0.1 * h_init, 0.9 * h_init, rows)
                centers = []
                radii = []
                cnt = 0
                for j in range(rows):
                    for i in range(cols):
                        if cnt >= N:
                            break
                        cx, cy = xs[i], ys[j]
                        r = min(cx, w_init - cx, cy, h_init - cy) * 0.7
                        centers.append([cx, cy])
                        radii.append(r)
                        cnt += 1
                if len(centers) == N:
                    x0 = np.concatenate([[w_init], np.array(centers).flatten(), np.array(radii)])
                    try_solve(x0)

    # Random initializations
    np.random.seed(1)
    for _ in range(8):
        if time.time() > deadline:
            break
        w_init = np.random.uniform(0.5, 1.5)
        h_init = 2 - w_init
        centers = np.column_stack([
            np.random.uniform(0.05, w_init - 0.05, N),
            np.random.uniform(0.05, h_init - 0.05, N)
        ])
        radii = np.full(N, 0.06)
        x0 = np.concatenate([[w_init], centers.flatten(), radii])
        try_solve(x0)

    if best_x is not None:
        w = best_x[0]
        centers = best_x[1:1 + 2 * N].reshape(N, 2)
        radii = np.maximum(best_x[1 + 2 * N:], 0)
        result = np.zeros((N, 3))
        for i in range(N):
            result[i] = [centers[i, 0], centers[i, 1], radii[i]]
        return result
    else:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
