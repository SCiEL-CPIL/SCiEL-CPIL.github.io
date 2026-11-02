"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay, distance_matrix
import sys

def _initial_config():
    """Create a hexagonal-ish initial packing for 26 circles."""
    # Use a hexagonal lattice pattern, then trim to N circles
    # Hexagonal packing: rows offset
    pts = []
    r_target = 0.12  # initial radius guess
    dy = 2 * r_target * 0.866  # sqrt(3)*r
    dx = 2 * r_target
    row = 0
    while len(pts) < N + 4:
        x_start = (row % 2) * dx * 0.5
        col = 0
        while True:
            x = x_start + col * dx + r_target + 0.02
            y = r_target + row * dy + 0.02
            if x > 1.0 - r_target or y > 1.0 - r_target:
                break
            pts.append((x, y))
            col += 1
        row += 1
    # Take first N
    pts = pts[:N]
    return np.array(pts)


def _compute_forces(x, y, r, dt=0.001):
    """Compute gradient of sum(r) subject to constraints via penalty method.
    Returns (grad_x, grad_y, grad_r, penalty_value)."""
    n = len(x)
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    grad_r = np.zeros(n)
    penalty = 0.0

    # Wall constraints: x_i >= r_i, x_i <= 1-r_i, y_i >= r_i, y_i <= 1-r_i
    # Circle-circle: dist(i,j) >= r_i + r_j

    # Use soft penalty: for each violated constraint, add quadratic penalty
    # Wall violations
    for i in range(n):
        # Left wall
        v = r[i] - x[i]
        if v > 0:
            penalty += v * v
            grad_x[i] -= 2 * v
            grad_r[i] += 2 * v
        # Right wall
        v = x[i] - (1 - r[i])
        if v > 0:
            penalty += v * v
            grad_x[i] += 2 * v
            grad_r[i] -= 2 * v
        # Bottom wall
        v = r[i] - y[i]
        if v > 0:
            penalty += v * v
            grad_y[i] -= 2 * v
            grad_r[i] += 2 * v
        # Top wall
        v = y[i] - (1 - r[i])
        if v > 0:
            penalty += v * v
            grad_y[i] += 2 * v
            grad_r[i] -= 2 * v

    # Circle-circle constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            d = np.sqrt(dx * dx + dy * dy)
            min_d = r[i] + r[j]
            v = min_d - d
            if v > 0 and d > 1e-12:
                penalty += v * v
                inv_d = 1.0 / d
                # grad w.r.t. x[i]: d/dx_i (min_d - d)^2 = -2v * (-dx/d) = 2v*dx/d
                gx_i = 2 * v * dx * inv_d
                gy_i = 2 * v * dy * inv_d
                # grad w.r.t. x[j]: -2v * (dx/d)
                gx_j = -2 * v * dx * inv_d
                gy_j = -2 * v * dy * inv_d
                # grad w.r.t. r[i]: -2v
                gr_i = -2 * v
                gr_j = -2 * v

                grad_x[i] += gx_i
                grad_y[i] += gy_i
                grad_x[j] += gx_j
                grad_y[j] += gy_j
                grad_r[i] += gr_i
                grad_r[j] += gr_j

    return grad_x, grad_y, grad_r, penalty


def _evaluate(x, y, r):
    """Check validity and return sum of radii."""
    n = len(x)
    # Check walls
    for i in range(n):
        if r[i] < 1e-9:
            return 0.0
        if x[i] < r[i] - 1e-7 or x[i] > 1 - r[i] + 1e-7:
            return 0.0
        if y[i] < r[i] - 1e-7 or y[i] > 1 - r[i] + 1e-7:
            return 0.0
    # Check circles
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            d = np.sqrt(dx * dx + dy * dy)
            if d < r[i] + r[j] - 1e-7:
                return 0.0
    return np.sum(r)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Initial configuration
    x, y = _initial_config().T
    n = N
    # Initial radii: small
    r = np.full(n, 0.05)

    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_val = _evaluate(best_x, best_y, best_r)

    # Phase 1: Gradient ascent with penalty on sum of radii
    # Objective: maximize sum(r) - lambda * penalty
    # Use adaptive learning rate

    lr = 0.002
    lambda_pen = 50.0
    no_improve_count = 0
    max_no_improve = 200

    for iteration in range(50000):
        if time.time() > deadline - 10.0:
            break

        grad_x, grad_y, grad_r, penalty = _compute_forces(x, y, r)

        # Gradient ascent: x += lr * grad
        # But we want to maximize sum(r) - lambda*penalty
        # So effective gradient for r is 1 - lambda * grad_r_penalty
        # For x,y: -lambda * grad_xy_penalty (we want to reduce penalty)
        # Actually: maximize F = sum(r) - lambda * penalty
        # dF/dx = -lambda * d(penalty)/dx
        # dF/dy = -lambda * d(penalty)/dy
        # dF/dr = 1 - lambda * d(penalty)/dr

        eff_grad_x = -lambda_pen * grad_x
        eff_grad_y = -lambda_pen * grad_y
        eff_grad_r = 1.0 - lambda_pen * grad_r

        # Adaptive step: limit movement
        x += lr * eff_grad_x
        y += lr * eff_grad_y
        r += lr * eff_grad_r

        # Clamp r to positive
        r = np.maximum(r, 0.01)

        # Clamp positions to stay roughly in bounds
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)

        # Evaluate
        val = _evaluate(x, y, r)
        if val > best_val + 1e-9:
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
            best_val = val
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Adaptive learning rate
        if no_improve_count > 50:
            lr *= 0.95
            lambda_pen *= 1.05
        if lr < 1e-7:
            break

        # Every 100 iterations, try to grow radii uniformly
        if iteration % 100 == 0:
            r_try = r * 1.02
            x_try, y_try = x.copy(), y.copy()
            # Quick check if feasible
            feasible = True
            for i in range(n):
                if x_try[i] < r_try[i] - 1e-6 or x_try[i] > 1 - r_try[i] + 1e-6:
                    feasible = False
                    break
                if y_try[i] < r_try[i] - 1e-6 or y_try[i] > 1 - r_try[i] + 1e-6:
                    feasible = False
                    break
            if feasible:
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = x_try[j] - x_try[i]
                        dy = y_try[j] - y_try[i]
                        d = np.sqrt(dx * dx + dy * dy)
                        if d < r_try[i] + r_try[j] - 1e-6:
                            feasible = False
                            break
                    if not feasible:
                        break
            if feasible:
                r = r_try
                val = _evaluate(x, y, r)
                if val > best_val + 1e-9:
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                    best_val = val
                    no_improve_count = 0

    # Phase 2: Polish - try to maximize each radius individually
    if time.time() < deadline - 5.0:
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in range(n):
                # Try to increase r[i] while adjusting positions
                for dr in [0.005, 0.002, 0.001]:
                    r_try = best_r.copy()
                    r_try[i] += dr
                    x_try = best_x.copy()
                    y_try = best_y.copy()

                    # Check feasibility with small position adjustments
                    feasible = True
                    for k in range(n):
                        if x_try[k] < r_try[k] - 1e-6 or x_try[k] > 1 - r_try[k] + 1e-6:
                            feasible = False
                            break
                        if y_try[k] < r_try[k] - 1e-6 or y_try[k] > 1 - r_try[k] + 1e-6:
                            feasible = False
                            break
                    if feasible:
                        for k in range(n):
                            for l in range(k + 1, n):
                                dx = x_try[l] - x_try[k]
                                dy = y_try[l] - y_try[k]
                                d = np.sqrt(dx * dx + dy * dy)
                                if d < r_try[k] + r_try[l] - 1e-6:
                                    feasible = False
                                    break
                            if not feasible:
                                break
                    if feasible:
                        best_r[i] += dr
                        improved = True
                        break
            if not improved:
                break

    # Final validation
    final_val = _evaluate(best_x, best_y, best_r)
    if final_val <= 0:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return np.column_stack([best_x, best_y, best_r])

# EVOLVE-BLOCK-END
