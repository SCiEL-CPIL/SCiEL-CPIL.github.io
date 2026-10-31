"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = np.datetime64('now')
    best_result = None
    best_score = -np.inf

    # Pre-compute constraint indices for speed
    pair_indices = [(i, j) for i in range(N) for j in range(i + 1, N)]

    def make_pair_constraint(i, j):
        def con(vars):
            xi, yi = vars[2 * i], vars[2 * i + 1]
            xj, yj = vars[2 * j], vars[2 * j + 1]
            ri, rj = vars[2 * N + 2 * i], vars[2 * N + 2 * j]
            return (xi - xj) ** 2 + (yi - yj) ** 2 - (ri + rj) ** 2
        return con

    pair_constraints = [{'type': 'ineq', 'fun': make_pair_constraint(i, j)} for i, j in pair_indices]

    def make_wall_constraint(axis, side, idx):
        # axis 0: x, 1: y
        # side 0: min (coord - r >= 0), side 1: max (1 - coord - r >= 0)
        def con(vars):
            coord_idx = 2 * idx + axis
            r_idx = 2 * N + 2 * idx
            if side == 0:
                return vars[coord_idx] - vars[r_idx]
            else:
                return 1.0 - vars[coord_idx] - vars[r_idx]
        return con

    wall_constraints = []
    for i in range(N):
        wall_constraints.append({'type': 'ineq', 'fun': make_wall_constraint(0, 0, i)})
        wall_constraints.append({'type': 'ineq', 'fun': make_wall_constraint(0, 1, i)})
        wall_constraints.append({'type': 'ineq', 'fun': make_wall_constraint(1, 0, i)})
        wall_constraints.append({'type': 'ineq', 'fun': make_wall_constraint(1, 1, i)})

    all_constraints = pair_constraints + wall_constraints

    def objective(vars):
        rs = vars[2 * N:]
        return -np.sum(rs)

    # Initializations: Grid and Random
    initializations = []
    
    # 1. Grid Initialization
    side = int(np.ceil(np.sqrt(N)))
    xs_grid = []
    ys_grid = []
    rs_grid = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            xs_grid.append((i + 0.5) / side)
            ys_grid.append((j + 0.5) / side)
            rs_grid.append(0.5 / side * 0.8) # Start smaller to allow optimization
            count += 1
    initializations.append(np.array(xs_grid + ys_grid + rs_grid))

    # 2. Random Initialization (a few)
    for _ in range(3):
        xs_rand = np.random.uniform(0.1, 0.9, N)
        ys_rand = np.random.uniform(0.1, 0.9, N)
        rs_rand = np.full(N, 0.05)
        initializations.append(np.concatenate([xs_rand, ys_rand, rs_rand]))

    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0, 0.5))  # r

    for x0 in initializations:
        # Check time
        elapsed = (np.datetime64('now') - start_time).astype('timedelta64[s]').astype(float)
        if elapsed > 100:
            break

        try:
            # SLSQP is robust for this size
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=all_constraints,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.success:
                # Validate
                xs = res.x[:N]
                ys = res.x[N:2 * N]
                rs = res.x[2 * N:]
                
                valid = True
                # Check walls
                for i in range(N):
                    if xs[i] < rs[i] - 1e-5 or xs[i] > 1 - rs[i] + 1e-5:
                        valid = False
                    if ys[i] < rs[i] - 1e-5 or ys[i] > 1 - rs[i] + 1e-5:
                        valid = False
                
                # Check pairs
                if valid:
                    for i in range(N):
                        for j in range(i + 1, N):
                            d = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
                            if d < rs[i] + rs[j] - 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(rs)
                    if score > best_score:
                        best_score = score
                        best_result = res.x

        except Exception:
            continue

    if best_result is not None:
        xs = best_result[:N]
        ys = best_result[N:2 * N]
        rs = best_result[2 * N:]
        return np.column_stack((xs, ys, rs))
    else:
        # Fallback: Return the grid solution
        return np.column_stack((np.array(xs_grid), np.array(ys_grid), np.array(rs_grid)))

# EVOLVE-BLOCK-END
