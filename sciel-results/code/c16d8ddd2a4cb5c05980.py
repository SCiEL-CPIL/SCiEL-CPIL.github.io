# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _valid(centers, sides, tol=1e-7):
    """Check validity of a packing configuration."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if not np.all(sides > 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol or cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True


def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    deadline = time.time() + 570

    # Grid baseline
    k = 4
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx0 = (cols + 0.5) * side
    cy0 = (rows + 0.5) * side
    s0 = np.full(n, side)
    best_cx = cx0.copy()
    best_cy = cy0.copy()
    best_s = s0.copy()
    best_obj = np.sum(s0)

    def check_feasible(cx, cy, s):
        for i in range(n):
            if cx[i] - s[i]/2 < -tol or cx[i] + s[i]/2 > 1 + tol:
                return False
            if cy[i] - s[i]/2 < -tol or cy[i] + s[i]/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                if t <= 0:
                    continue
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                if dx < t and dy < t:
                    return False
        return True

    rng = np.random.default_rng(42)

    def local_improve(cx, cy, s):
        improved = True
        max_iter = 500
        it = 0
        while improved and it < max_iter and time.time() < deadline - 120:
            improved = False
            it += 1
            for i in range(n):
                for dx in [-0.005, 0.005]:
                    for dy in [-0.005, 0.005]:
                        cx_new = cx.copy()
                        cy_new = cy.copy()
                        cx_new[i] += dx
                        cy_new[i] += dy
                        if check_feasible(cx_new, cy_new, s):
                            for j in range(n):
                                s_try = s.copy()
                                s_try[j] += 0.003
                                if check_feasible(cx_new, cy_new, s_try):
                                    cx, cy, s = cx_new, cy_new, s_try
                                    improved = True
                                    break
            for i in range(n):
                s_new = s.copy()
                s_new[i] += 0.003
                if check_feasible(cx, cy, s_new):
                    s = s_new
                    improved = True
        return cx, cy, s

    n_attempts = 30
    for attempt in range(n_attempts):
        if time.time() > deadline - 120:
            break

        if attempt == 0:
            cx, cy, s = cx0.copy(), cy0.copy(), s0.copy()
        elif attempt == 1:
            kx, ky = 2, 5
            idx = np.arange(n)
            cols = idx % kx
            rows = idx // kx
            cx = (cols + 0.5) * (1.0 / kx)
            cy = (rows + 0.5) * (1.0 / ky)
            s = np.full(n, min(1.0/kx, 1.0/ky))
        elif attempt == 2:
            kx, ky = 5, 2
            idx = np.arange(n)
            cols = idx % kx
            rows = idx // kx
            cx = (cols + 0.5) * (1.0 / kx)
            cy = (rows + 0.5) * (1.0 / ky)
            s = np.full(n, min(1.0/kx, 1.0/ky))
        else:
            cx = rng.uniform(0.1, 0.9, n)
            cy = rng.uniform(0.1, 0.9, n)
            s = rng.uniform(0.1, 0.3, n)
            for i in range(n):
                s[i] = min(s[i], 0.4)
                cx[i] = np.clip(cx[i], s[i]/2 + tol, 1 - s[i]/2 - tol)
                cy[i] = np.clip(cy[i], s[i]/2 + tol, 1 - s[i]/2 - tol)

        if not check_feasible(cx, cy, s):
            for i in range(n):
                while not check_feasible(cx, cy, s) and s[i] > 0.05:
                    s[i] *= 0.9
            if not check_feasible(cx, cy, s):
                continue

        cx, cy, s = local_improve(cx, cy, s)
        obj = np.sum(s)
        if obj > best_obj:
            best_obj = obj
            best_cx = cx.copy()
            best_cy = cy.copy()
            best_s = s.copy()

    # SLSQP polish pass
    if time.time() < deadline - 30:
        try:
            # Determine separation pattern from best heuristic solution
            # For each pair, determine which axis separates them
            sep_pattern = {}
            for i in range(n):
                for j in range(i + 1, n):
                    dx = abs(best_cx[i] - best_cx[j])
                    dy = abs(best_cy[i] - best_cy[j])
                    if dx >= dy:
                        # Separated in x
                        if best_cx[i] < best_cx[j]:
                            sep_pattern[(i, j)] = 0  # i left of j
                        else:
                            sep_pattern[(i, j)] = 1  # i right of j
                    else:
                        # Separated in y
                        if best_cy[i] < best_cy[j]:
                            sep_pattern[(i, j)] = 2  # i below j
                        else:
                            sep_pattern[(i, j)] = 3  # i above j

            # Build SLSQP problem
            # Variables: [cx_0, ..., cx_9, cy_0, ..., cy_9, s_0, ..., s_9]
            x0 = np.concatenate([best_cx, best_cy, best_s])

            def objective(x):
                return -np.sum(x[2*n:])

            def objective_grad(x):
                g = np.zeros(3*n)
                g[2*n:] = -np.ones(n)
                return g

            constraints = []
            # Boundary constraints: s_i/2 <= cx_i <= 1 - s_i/2, same for cy
            for i in range(n):
                # cx_i - s_i/2 >= 0
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: x[i] - x[2*n+i]/2,
                    'jac': lambda x, i=i: np.concatenate([
                        np.eye(1)[0] - np.zeros(1),  # cx_i coeff: 1
                        np.zeros(1),  # cy_i coeff: 0
                        -0.5  # s_i coeff
                    ] + np.zeros(3*n - 3))
                })
                # 1 - cx_i - s_i/2 >= 0
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: 1 - x[i] - x[2*n+i]/2,
                    'jac': lambda x, i=i: np.concatenate([
                        -np.eye(1)[0],  # cx_i coeff: -1
                        np.zeros(1),
                        -0.5
                    ] + np.zeros(3*n - 3))
                })
                # cy_i - s_i/2 >= 0
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: x[n+i] - x[2*n+i]/2,
                    'jac': lambda x, i=i: np.concatenate([
                        np.zeros(n),
                        np.eye(1)[0] - np.zeros(1),
                        -0.5
                    ] + np.zeros(3*n - 3))
                })
                # 1 - cy_i - s_i/2 >= 0
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: 1 - x[n+i] - x[2*n+i]/2,
                    'jac': lambda x, i=i: np.concatenate([
                        np.zeros(n),
                        -np.eye(1)[0],
                        -0.5
                    ] + np.zeros(3*n - 3))
                })
                # s_i > 0 (small lower bound)
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: x[2*n+i] - 0.01,
                    'jac': lambda x, i=i: np.concatenate([
                        np.zeros(2*n),
                        np.eye(1)[0]
                    ] + np.zeros(n - 1))
                })

            # Separation constraints based on pattern
            for (i, j), pat in sep_pattern.items():
                if pat == 0:  # i left of j: cx_i + s_i/2 <= cx_j - s_j/2
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, i=i, j=j: x[j] - x[i] - x[2*n+i]/2 - x[2*n+j]/2,
                        'jac': lambda x, i=i, j=j: np.concatenate([
                            np.array([-1 if k == i else (1 if k == j else 0) for k in range(n)], dtype=float),
                            np.zeros(n),
                            np.array([-0.5 if k == i else (-0.5 if k == j else 0) for k in range(n)], dtype=float)
                        ])
                    })
                elif pat == 1:  # i right of j: cx_j + s_j/2 <= cx_i - s_i/2
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, i=i, j=j: x[i] - x[j] - x[2*n+i]/2 - x[2*n+j]/2,
                        'jac': lambda x, i=i, j=j: np.concatenate([
                            np.array([1 if k == i else (-1 if k == j else 0) for k in range(n)], dtype=float),
                            np.zeros(n),
                            np.array([-0.5 if k == i else (-0.5 if k == j else 0) for k in range(n)], dtype=float)
                        ])
                    })
                elif pat == 2:  # i below j: cy_i + s_i/2 <= cy_j - s_j/2
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, i=i, j=j: x[n+j] - x[n+i] - x[2*n+i]/2 - x[2*n+j]/2,
                        'jac': lambda x, i=i, j=j: np.concatenate([
                            np.zeros(n),
                            np.array([-1 if k == i else (1 if k == j else 0) for k in range(n)], dtype=float),
                            np.array([-0.5 if k == i else (-0.5 if k == j else 0) for k in range(n)], dtype=float)
                        ])
                    })
                elif pat == 3:  # i above j: cy_j + s_j/2 <= cy_i - s_i/2
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, i=i, j=j: x[n+i] - x[n+j] - x[2*n+i]/2 - x[2*n+j]/2,
                        'jac': lambda x, i=i, j=j: np.concatenate([
                            np.zeros(n),
                            np.array([1 if k == i else (-1 if k == j else 0) for k in range(n)], dtype=float),
                            np.array([-0.5 if k == i else (-0.5 if k == j else 0) for k in range(n)], dtype=float)
                        ])
                    })

            bounds = [(0, 1)] * (2*n) + [(0.01, 1)] * n

            result = minimize(
                objective, x0, jac=objective_grad,
                method='SLSQP', bounds=bounds, constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )

            if result.success:
                x_sol = result.x
                cx_sol = x_sol[:n]
                cy_sol = x_sol[n:2*n]
                s_sol = x_sol[2*n:]
                centers_sol = np.stack([cx_sol, cy_sol], axis=1)
                if _valid(centers_sol, s_sol) and np.sum(s_sol) > best_obj:
                    best_obj = np.sum(s_sol)
                    best_cx = cx_sol.copy()
                    best_cy = cy_sol.copy()
                    best_s = s_sol.copy()
        except Exception:
            pass

    # Final cleanup
    best_s = np.maximum(best_s, 1e-8)
    for i in range(n):
        best_cx[i] = np.clip(best_cx[i], best_s[i]/2 - tol, 1.0 - best_s[i]/2 + tol)
        best_cy[i] = np.clip(best_cy[i], best_s[i]/2 - tol, 1.0 - best_s[i]/2 + tol)

    centers = np.stack([best_cx, best_cy], axis=1).astype(np.float64)
    sides = best_s.astype(np.float64)

    return centers, sides


# EVOLVE-BLOCK-END
