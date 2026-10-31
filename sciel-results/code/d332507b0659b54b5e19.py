"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26
DEADLINE = 120  # seconds, safety margin from 150s limit


def _evaluate(circles):
    """Compute sum of radii and check validity."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(r < 0):
        return 0.0
    # Check inside unit square
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9) or \
       np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return 0.0
    # Check pairwise non-overlap
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < r[i] + r[j] - 1e-9:
                return 0.0
    return float(np.sum(r))


def _random_init(n, seed=None):
    """Random initial configuration with small radii."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    x = rng.uniform(0.05, 0.95, n)
    y = rng.uniform(0.05, 0.95, n)
    r = np.full(n, 0.02)
    return np.column_stack([x, y, r])


def _slsqp_optimize(circles, max_iter=500):
    """Optimize circle packing using SLSQP with explicit constraints."""
    n = len(circles)
    x0 = circles.flatten()

    def objective(v):
        # Negate sum of radii for minimization
        return -np.sum(v[2::3])

    def objective_grad(v):
        grad = np.zeros_like(v)
        grad[2::3] = -1.0
        return grad

    # Inequality constraints: g_i(v) >= 0
    constraints = []

    # Boundary constraints: x_i - r_i >= 0, x_i + r_i <= 1, y_i - r_i >= 0, y_i + r_i <= 1
    for i in range(n):
        xi, yi, ri = 3*i, 3*i+1, 3*i+2
        # x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i] - v[3*i+2],
            'jac': lambda v, i=i: np.concatenate([np.zeros(3*i), [1], [0], [0], np.zeros(3*(n-i)-3)])
        })
        # 1 - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2],
            'jac': lambda v, i=i: np.concatenate([np.zeros(3*i), [-1], [0], [-1], np.zeros(3*(n-i)-3)])
        })
        # y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i+1] - v[3*i+2],
            'jac': lambda v, i=i: np.concatenate([np.zeros(3*i), [0], [1], [0], np.zeros(3*(n-i)-3)])
        })
        # 1 - y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2],
            'jac': lambda v, i=i: np.concatenate([np.zeros(3*i), [0], [-1], [-1], np.zeros(3*(n-i)-3)])
        })

    # Pairwise non-overlap: ||c_i - c_j|| - r_i - r_j >= 0
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = 3*i, 3*i+1, 3*i+2
            xj, yj, rj = 3*j, 3*j+1, 3*j+2

            def pair_fun(v, i=i, j=j):
                dx = v[xi] - v[xj]
                dy = v[yi] - v[yj]
                d = np.sqrt(dx*dx + dy*dy + 1e-12)
                return d - v[ri] - v[rj]

            def pair_jac(v, i=i, j=j):
                dx = v[xi] - v[xj]
                dy = v[yi] - v[yj]
                d = np.sqrt(dx*dx + dy*dy + 1e-12)
                jv = np.zeros(3*n)
                jv[xi] = dx / d
                jv[yi] = dy / d
                jv[xj] = -dx / d
                jv[yj] = -dy / d
                jv[ri] = -1.0
                jv[rj] = -1.0
                return jv

            constraints.append({
                'type': 'ineq',
                'fun': pair_fun,
                'jac': pair_jac
            })

    # Bounds: radii > 0, positions in [0,1]
    bounds = []
    for i in range(n):
        bounds.append((1e-6, 1.0))  # x
        bounds.append((1e-6, 1.0))  # y
        bounds.append((1e-6, 0.5))  # r

    result = minimize(
        objective, x0, jac=objective_grad,
        method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': max_iter, 'ftol': 1e-8}
    )

    return result.x.reshape(n, 3)


def _gradient_descent(circles, lr=0.01, steps=2000, tol=1e-8):
    """Improved gradient descent with pairwise gradient computation."""
    n = len(circles)
    circles = circles.copy().astype(float)
    r_min = 1e-6

    for step in range(steps):
        grad = np.zeros_like(circles)

        # Objective: maximize sum of radii -> gradient of -sum(r) is -1 for r components
        # But we want to maximize, so we add to radii
        # Force gradient: increase radii
        grad[:, 2] += 1.0  # push radii up

        # Boundary penalties (soft): if circle goes outside, push back
        for i in range(n):
            x, y, r = circles[i]
            # Left boundary
            if x - r < 0:
                grad[i, 0] += 10.0  # push x right
                grad[i, 2] -= 10.0  # reduce r
            # Right boundary
            if x + r > 1:
                grad[i, 0] -= 10.0
                grad[i, 2] -= 10.0
            # Bottom boundary
            if y - r < 0:
                grad[i, 1] += 10.0
                grad[i, 2] -= 10.0
            # Top boundary
            if y + r > 1:
                grad[i, 1] -= 10.0
                grad[i, 2] -= 10.0

        # Pairwise overlap: if circles overlap, push them apart and reduce radii
        for i in range(n):
            for j in range(i+1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                d = np.sqrt(dx*dx + dy*dy + 1e-12)
                r_sum = circles[i, 2] + circles[j, 2]
                if d < r_sum:
                    # Overlap: push apart
                    overlap = r_sum - d
                    fx = dx / d
                    fy = dy / d
                    grad[i, 0] += fx * 5.0
                    grad[i, 1] += fy * 5.0
                    grad[j, 0] -= fx * 5.0
                    grad[j, 1] -= fy * 5.0
                    # Reduce radii
                    grad[i, 2] -= 2.5
                    grad[j, 2] -= 2.5

        # Adaptive learning rate
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break

        # Clip gradient
        if grad_norm > 1.0:
            grad = grad / grad_norm

        circles += lr * grad

        # Project to feasible region
        for i in range(n):
            # Keep radius positive
            circles[i, 2] = max(circles[i, 2], r_min)
            # Keep center such that circle stays in [0,1]^2
            r = circles[i, 2]
            circles[i, 0] = np.clip(circles[i, 0], r, 1 - r)
            circles[i, 1] = np.clip(circles[i, 1], r, 1 - r)

        # Adaptive LR: decrease if not improving
        if step % 100 == 0 and step > 0:
            lr *= 0.95

    return circles


def _perturb_restarts(circles, n_restarts=5, seed_base=0):
    """Try perturbations and re-optimize to escape local optima."""
    best = circles.copy()
    best_val = _evaluate(best)

    for k in range(n_restarts):
        rng = np.random.RandomState(seed_base + k)
        perturbed = best.copy()
        # Small random perturbation
        perturbed[:, :2] += rng.uniform(-0.02, 0.02, (len(perturbed), 2))
        perturbed[:, 2] *= (1 + rng.uniform(-0.05, 0.05, len(perturbed)))
        # Project
        for i in range(len(perturbed)):
            r = max(perturbed[i, 2], 1e-6)
            perturbed[i, 2] = r
            perturbed[i, 0] = np.clip(perturbed[i, 0], r, 1 - r)
            perturbed[i, 1] = np.clip(perturbed[i, 1], r, 1 - r)

        # Quick gradient descent polish
        polished = _gradient_descent(perturbed, lr=0.005, steps=500)
        val = _evaluate(polished)
        if val > best_val:
            best = polished
            best_val = val

    return best


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5  # 5s safety margin

    best_result = None
    best_score = -1

    # Multiple random restarts
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break

        seed = 42 + restart * 1000
        circles = _random_init(N, seed=seed)

        # Phase 1: Gradient descent to get a reasonable feasible solution
        gd_result = _gradient_descent(circles, lr=0.02, steps=3000)
        score = _evaluate(gd_result)

        if score > best_score:
            best_score = score
            best_result = gd_result.copy()

        # Phase 2: SLSQP polish if time allows
        if time.time() < deadline - 20:
            slsqp_result = _slsqp_optimize(gd_result, max_iter=200)
            slsqp_score = _evaluate(slsqp_result)
            if slsqp_score > best_score:
                best_score = slsqp_score
                best_result = slsqp_result.copy()

    # Final perturbation-restart polish
    if best_result is not None and time.time() < deadline - 5:
        polished = _perturb_restarts(best_result, n_restarts=3, seed_base=9999)
        polish_score = _evaluate(polished)
        if polish_score > best_score:
            best_score = polish_score
            best_result = polished

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])

    return best_result
# EVOLVE-BLOCK-END
