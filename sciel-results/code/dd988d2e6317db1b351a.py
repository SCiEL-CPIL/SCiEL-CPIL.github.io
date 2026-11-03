"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26


def _objective_and_constraints(x):
    """
    x: (3N,) vector: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    Maximize sum(r_i) subject to:
      - Circles inside [0,1]^2: x_i - r_i >= 0, x_i + r_i <= 1, y_i - r_i >= 0, y_i + r_i <= 1
      - Non-overlap: dist(i,j) >= r_i + r_j
    """
    n = len(x) // 3
    centers = x[:2*n].reshape(n, 2)
    radii = x[2*n:]

    # Objective: negative sum of radii (minimize)
    obj = -np.sum(radii)

    # Constraints (all should be >= 0)
    cons = []

    # Boundary constraints
    for i in range(n):
        xi, yi, ri = centers[i, 0], centers[i, 1], radii[i]
        cons.append(xi - ri)       # left
        cons.append(1.0 - xi - ri) # right
        cons.append(yi - ri)       # bottom
        cons.append(1.0 - yi - ri) # top

    # Non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            cons.append(d - radii[i] - radii[j])

    return obj, cons


def _gradient(x):
    """Gradient of the objective (negative sum of radii)."""
    n = len(x) // 3
    grad = np.zeros_like(x)
    grad[2*n:] = -1.0  # d(-sum(r))/dr_i = -1
    return grad


def _init_grid():
    """Initial guess: grid of equal circles."""
    side = int(np.ceil(np.sqrt(N)))
    r = 0.5 / side
    centers = []
    radii = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            centers.append(((i + 0.5) / side, (j + 0.5) / side))
            radii.append(r)
            count += 1
    centers = np.array(centers)
    radii = np.array(radii)
    x = np.concatenate([centers.ravel(), radii])
    return x


def _check_validity(x, tol=1e-7):
    """Check if the configuration is valid (all constraints satisfied within tolerance)."""
    n = len(x) // 3
    centers = x[:2*n].reshape(n, 2)
    radii = x[2*n:]

    # Check radii positive
    if np.any(radii <= 0):
        return False

    # Check boundaries
    for i in range(n):
        if centers[i,0] - radii[i] < -tol:
            return False
        if 1.0 - centers[i,0] - radii[i] < -tol:
            return False
        if centers[i,1] - radii[i] < -tol:
            return False
        if 1.0 - centers[i,1] - radii[i] < -tol:
            return False

    # Check non-overlap
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            if d < radii[i] + radii[j] - tol:
                return False

    return True


def _project_to_valid(x, tol=1e-9):
    """Project a slightly invalid configuration to a valid one by shrinking radii."""
    n = len(x) // 3
    centers = x[:2*n].reshape(n, 2)
    radii = x[2*n:].copy()

    # Iteratively shrink radii to satisfy constraints
    for _ in range(100):
        valid = True
        # Check boundaries
        for i in range(n):
            max_r = min(centers[i,0], 1.0 - centers[i,0], centers[i,1], 1.0 - centers[i,1])
            if radii[i] > max_r + tol:
                radii[i] = max(0.0, max_r - tol)
                valid = False

        # Check non-overlap
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                if d < radii[i] + radii[j] + tol:
                    # Shrink both proportionally
                    excess = radii[i] + radii[j] - d + tol
                    if excess > 0:
                        # Reduce each by half the excess
                        shrink_i = excess / 2.0
                        shrink_j = excess / 2.0
                        radii[i] = max(0.0, radii[i] - shrink_i)
                        radii[j] = max(0.0, radii[j] - shrink_j)
                        valid = False

        if valid:
            break

    x_out = np.concatenate([centers.ravel(), radii])
    return x_out


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to be safe under 150s limit

    # Initial grid configuration
    x0 = _init_grid()

    # Try multiple random initializations and optimize
    best_x = x0.copy()
    best_val = -np.sum(x0[2*N:])  # negative sum of radii

    # Evaluate initial
    if _check_validity(x0):
        best_x = x0.copy()
        best_val = -np.sum(x0[2*N:])

    # Define constraints for scipy
    def make_constraints(n):
        cons_list = []
        # Boundary constraints
        for i in range(n):
            xi = 2*i
            yi = 2*i + 1
            ri = 2*n + i
            # x_i - r_i >= 0
            cons_list.append({
                'type': 'ineq',
                'fun': lambda x, xi=xi, ri=ri: x[xi] - x[ri],
                'jac': lambda x, xi=xi, ri=ri: np.concatenate([[1.0], [0.0], [-1.0]])  # simplified
            })
            # 1 - x_i - r_i >= 0
            cons_list.append({
                'type': 'ineq',
                'fun': lambda x, xi=xi, ri=ri: 1.0 - x[xi] - x[ri],
                'jac': lambda x, xi=xi, ri=ri: np.concatenate([[-1.0], [0.0], [-1.0]])
            })
            # y_i - r_i >= 0
            cons_list.append({
                'type': 'ineq',
                'fun': lambda x, yi=yi, ri=ri: x[yi] - x[ri],
                'jac': lambda x, yi=yi, ri=ri: np.concatenate([[0.0], [1.0], [-1.0]])
            })
            # 1 - y_i - r_i >= 0
            cons_list.append({
                'type': 'ineq',
                'fun': lambda x, yi=yi, ri=ri: 1.0 - x[yi] - x[ri],
                'jac': lambda x, yi=yi, ri=ri: np.concatenate([[0.0], [-1.0], [-1.0]])
            })

        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                xi, yi = 2*i, 2*i+1
                xj, yj = 2*j, 2*j+1
                ri, rj = 2*n+i, 2*n+j
                cons_list.append({
                    'type': 'ineq',
                    'fun': lambda x, xi=xi, yi=yi, xj=xj, yj=yj, ri=ri, rj=rj:
                        np.sqrt((x[xi]-x[xj])**2 + (x[yi]-x[yj])**2) - x[ri] - x[rj],
                })

        return cons_list

    constraints = make_constraints(N)

    # Bounds: centers in [0,1], radii in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
    for i in range(N):
        bounds.append((0.0, 0.5))  # r

    # Try several random restarts
    np.random.seed(42)
    n_restarts = 20

    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break

        if restart == 0:
            x_init = x0.copy()
        else:
            # Random perturbation of grid
            x_init = x0.copy()
            # Add small random noise
            noise_scale = 0.05
            x_init[:2*N] += np.random.uniform(-noise_scale, noise_scale, 2*N)
            x_init[2*N:] *= np.random.uniform(0.8, 1.2, N)
            # Clamp
            x_init[:2*N] = np.clip(x_init[:2*N], 0.0, 1.0)
            x_init[2*N:] = np.clip(x_init[2*N:], 0.01, 0.5)

        try:
            res = minimize(
                fun=lambda x: -np.sum(x[2*N:]),
                x0=x_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10}
            )

            if res.success or res.status in [0, 8]:
                x_sol = res.x
                # Project to valid if needed
                if not _check_validity(x_sol, tol=1e-6):
                    x_sol = _project_to_valid(x_sol)

                if _check_validity(x_sol):
                    val = -np.sum(x_sol[2*N:])
                    if val < best_val:  # More negative = better (higher sum of radii)
                        best_val = val
                        best_x = x_sol.copy()

        except Exception as e:
            pass

        # Check time
        if time.time() > deadline - 15.0:
            break

    # Final polish: try to improve the best solution with tighter tolerances
    if time.time() < deadline - 5.0 and _check_validity(best_x):
        try:
            res2 = minimize(
                fun=lambda x: -np.sum(x[2*N:]),
                x0=best_x,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res2.success:
                x_sol2 = res2.x
                if not _check_validity(x_sol2, tol=1e-6):
                    x_sol2 = _project_to_valid(x_sol2)
                if _check_validity(x_sol2):
                    val2 = -np.sum(x_sol2[2*N:])
                    if val2 < best_val:
                        best_x = x_sol2.copy()
                        best_val = val2
        except Exception:
            pass

    # Ensure final validity
    if not _check_validity(best_x, tol=1e-7):
        best_x = _project_to_valid(best_x)

    # Construct output: (n, 3) array of (x, y, radius)
    centers = best_x[:2*N].reshape(N, 2)
    radii = best_x[2*N:]
    result = np.column_stack([centers[:, 0], centers[:, 1], radii])

    return result


# Quick self-test
if __name__ == "__main__":
    result = construct()
    print("Shape:", result.shape)
    print("Sum of radii:", np.sum(result[:, 2]))
    # Verify validity
    valid = _check_validity(result.ravel())
    print("Valid:", valid)
