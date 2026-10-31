import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin

    # Variables: [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
    # Size: 3 * N = 78

    # Define Objective
    def objective(vars):
        radii = vars[2::3]
        return -np.sum(radii)

    def objective_grad(vars):
        grad = np.zeros_like(vars)
        grad[2::3] = -1.0
        return grad

    # Define Constraints
    # 1. Boundary Constraints (Linear)
    A = np.zeros((5 * N, 3 * N))
    lb = np.zeros(5 * N)
    ub = np.full(5 * N, np.inf)

    for i in range(N):
        col_x = 3 * i
        col_y = 3 * i + 1
        col_r = 3 * i + 2

        # x_i - r_i >= 0
        A[5 * i, col_x] = 1
        A[5 * i, col_r] = -1

        # 1 - x_i - r_i >= 0  =>  -x_i - r_i >= -1
        A[5 * i + 1, col_x] = -1
        A[5 * i + 1, col_r] = -1
        lb[5 * i + 1] = -1

        # y_i - r_i >= 0
        A[5 * i + 2, col_y] = 1
        A[5 * i + 2, col_r] = -1

        # 1 - y_i - r_i >= 0  =>  -y_i - r_i >= -1
        A[5 * i + 3, col_y] = -1
        A[5 * i + 3, col_r] = -1
        lb[5 * i + 3] = -1

        # r_i >= 0
        A[5 * i + 4, col_r] = 1

    lin_con = LinearConstraint(A, lb, ub)

    # 2. Overlap Constraints (Nonlinear)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    num_pairs = len(pairs)

    def overlap_func(vars):
        vals = np.zeros(num_pairs)
        for k, (i, j) in enumerate(pairs):
            xi = vars[3 * i]
            yi = vars[3 * i + 1]
            ri = vars[3 * i + 2]
            xj = vars[3 * j]
            yj = vars[3 * j + 1]
            rj = vars[3 * j + 2]
            dx = xi - xj
            dy = yi - yj
            dr = ri + rj
            vals[k] = dx * dx + dy * dy - dr * dr
        return vals

    def overlap_jac(vars):
        jac = np.zeros((num_pairs, 3 * N))
        for k, (i, j) in enumerate(pairs):
            xi = vars[3 * i]
            yi = vars[3 * i + 1]
            ri = vars[3 * i + 2]
            xj = vars[3 * j]
            yj = vars[3 * j + 1]
            rj = vars[3 * j + 2]

            dx = xi - xj
            dy = yi - yj
            dr = ri + rj

            jac[k, 3 * i] = 2 * dx
            jac[k, 3 * i + 1] = 2 * dy
            jac[k, 3 * i + 2] = -2 * dr

            jac[k, 3 * j] = -2 * dx
            jac[k, 3 * j + 1] = -2 * dy
            jac[k, 3 * j + 2] = -2 * dr
        return jac

    nonlin_con = NonlinearConstraint(overlap_func, 0, np.inf, jac=overlap_jac)

    constraints = [lin_con, nonlin_con]

    # Bounds for variables
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N

    best_score = -np.inf
    best_vars = None

    # Initializations
    inits = []

    # 1. Grid Init
    side = int(np.ceil(np.sqrt(N)))
    grid_cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01) for i in range(side) for j in range(side)]
    grid_vars = np.zeros(3 * N)
    for i in range(N):
        grid_vars[3 * i] = grid_cells[i][0]
        grid_vars[3 * i + 1] = grid_cells[i][1]
        grid_vars[3 * i + 2] = grid_cells[i][2]
    inits.append(grid_vars)

    # 2. Random Inits
    rng = np.random.default_rng(42)
    for _ in range(5):
        vars = np.zeros(3 * N)
        for i in range(N):
            vars[3 * i] = rng.uniform(0.1, 0.9)
            vars[3 * i + 1] = rng.uniform(0.1, 0.9)
            vars[3 * i + 2] = rng.uniform(0.01, 0.05)
        inits.append(vars)

    # 3. Hexagonal-ish Init
    r_init = 0.1
    dx = 2 * r_init
    dy = np.sqrt(3) * r_init
    hex_vars = np.zeros(3 * N)
    idx = 0
    row = 0
    col = 0
    while idx < N:
        x = 0.5 + col * dx + (0.5 * dx if row % 2 else 0)
        y = 0.5 + row * dy
        if x < 0.9 and y < 0.9:
            hex_vars[3 * idx] = x
            hex_vars[3 * idx + 1] = y
            hex_vars[3 * idx + 2] = r_init
            idx += 1
        if idx < N:
            col += 1
            if col > 10:
                col = 0
                row += 1
    inits.append(hex_vars)

    # Run optimization for each init
    for init_vars in inits:
        if time.time() > deadline:
            break

        try:
            res = minimize(
                objective,
                init_vars,
                method='trust-constr',
                jac=objective_grad,
                constraints=constraints,
                bounds=bounds,
                options={'maxiter': 100, 'gtol': 1e-6, 'verbose': 0}
            )

            if res.success or res.status == 0:
                score = -res.fun
                if score > best_score:
                    x = res.x[0::3]
                    y = res.x[1::3]
                    r = res.x[2::3]

                    # Check bounds
                    if np.any(x < -1e-5) or np.any(x > 1 + 1e-5) or \
                       np.any(y < -1e-5) or np.any(y > 1 + 1e-5) or \
                       np.any(r < -1e-5):
                        continue

                    # Check overlaps
                    feasible = True
                    for i in range(N):
                        for j in range(i + 1, N):
                            dist_sq = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
                            rad_sum_sq = (r[i] + r[j]) ** 2
                            if dist_sq < rad_sum_sq - 1e-5:
                                feasible = False
                                break
                        if not feasible:
                            break

                    if feasible:
                        best_score = score
                        best_vars = res.x.copy()

        except Exception:
            pass

    if best_vars is None:
        # Fallback to grid if optimization failed
        return np.array(grid_cells[:N])
    else:
        result = np.zeros((N, 3))
        result[:, 0] = best_vars[0::3]
        result[:, 1] = best_vars[1::3]
        result[:, 2] = best_vars[2::3]
        return result
