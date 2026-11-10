# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square [0,1]^2
    to maximize the sum of their side lengths, with no overlapping interiors.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 580  # safety margin before 630s kill

    def is_valid(centers, sides):
        """Check if the configuration is valid."""
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        # Boundary check
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        # Overlap check
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    return False
        return True

    def objective(params):
        """Negative sum of sides (to minimize)."""
        centers = params[:2*n].reshape(n, 2)
        sides = params[2*n:]
        return -np.sum(sides)

    def penalty(params):
        """Penalty for violations."""
        centers = params[:2*n].reshape(n, 2)
        sides = params[2*n:]
        pen = 0.0
        # Boundary penalties
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if s <= 0:
                pen += 1e6
            if cx - s/2 < -tol:
                pen += 1e6 * (-tol - (cx - s/2))**2
            if cx + s/2 > 1 + tol:
                pen += 1e6 * ((cx + s/2) - (1 + tol))**2
            if cy - s/2 < -tol:
                pen += 1e6 * (-tol - (cy - s/2))**2
            if cy + s/2 > 1 + tol:
                pen += 1e6 * ((cy + s/2) - (1 + tol))**2
        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                dx = abs(centers[i,0] - centers[j,0])
                dy = abs(centers[i,1] - centers[j,1])
                if dx < t and dy < t:
                    # Overlap: penalize based on how deep the overlap is
                    pen += 1e6 * (t - dx)**2 + 1e6 * (t - dy)**2
        return pen

    def objective_with_penalty(params):
        return objective(params) + penalty(params)

    # Generate multiple diverse initial guesses
    def make_initial_guess(seed):
        rng = np.random.RandomState(seed)
        params = np.zeros(3*n)
        # Random positions and sizes
        for i in range(n):
            s = rng.uniform(0.1, 0.4)
            cx = rng.uniform(s/2, 1 - s/2)
            cy = rng.uniform(s/2, 1 - s/2)
            params[2*i] = cx
            params[2*i+1] = cy
            params[2*n+i] = s
        return params

    def make_grid_guess():
        """4x4 grid baseline."""
        k = 4
        side = 1.0 / k
        params = np.zeros(3*n)
        for i in range(n):
            col = i % k
            row = i // k
            params[2*i] = (col + 0.5) * side
            params[2*i+1] = (row + 0.5) * side
            params[2*n+i] = side
        return params

    def make_mixed_guess():
        """Mixed size initial guess: some large, some small."""
        params = np.zeros(3*n)
        # 2 large squares (side 0.45)
        for i in range(2):
            s = 0.45
            params[2*i] = 0.25 if i == 0 else 0.75
            params[2*i+1] = 0.5
            params[2*n+i] = s
        # 3 medium squares (side 0.3)
        for i in range(3):
            s = 0.3
            params[2*(i+2)] = 0.25 + 0.25 * i
            params[2*(i+2)+1] = 0.15
            params[2*n+i+2] = s
        # 5 small squares (side 0.18)
        for i in range(5):
            s = 0.18
            params[2*(i+5)] = 0.1 + 0.2 * i
            params[2*(i+5)+1] = 0.85
            params[2*n+i+5] = s
        return params

    best_params = None
    best_score = -np.inf

    # Try multiple initializations
    initializations = [make_grid_guess(), make_mixed_guess()]
    for seed in range(20):
        initializations.append(make_initial_guess(seed))

    for init_params in initializations:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                objective_with_penalty,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            continue

    # Also try SLSQP on the best result so far
    if best_params is not None and time.time() < deadline:
        try:
            result = minimize(
                objective_with_penalty,
                best_params,
                method='SLSQP',
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Extract result
    if best_params is None:
        # Fallback to grid
        k = 4
        side = 1.0 / k
        centers = np.zeros((n, 2))
        sides = np.full(n, side)
        for i in range(n):
            col = i % k
            row = i // k
            centers[i, 0] = (col + 0.5) * side
            centers[i, 1] = (row + 0.5) * side
    else:
        centers = best_params[:2*n].reshape(n, 2)
        sides = best_params[2*n:]
        # Clamp to valid range
        for i in range(n):
            s = max(sides[i], 1e-6)
            sides[i] = s
            centers[i, 0] = np.clip(centers[i, 0], s/2, 1 - s/2)
            centers[i, 1] = np.clip(centers[i, 1], s/2, 1 - s/2)

    # Final validation and fallback
    if not is_valid(centers, sides):
        k = 4
        side = 1.0 / k
        centers = np.zeros((n, 2))
        sides = np.full(n, side)
        for i in range(n):
            col = i % k
            row = i // k
            centers[i, 0] = (col + 0.5) * side
            centers[i, 1] = (row + 0.5) * side

    return centers.astype(np.float64), sides.astype(np.float64)


# EVOLVE-BLOCK-END
