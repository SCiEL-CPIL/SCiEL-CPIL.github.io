"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 110.0  # seconds, safety margin before 120s kill


def _perimeter_penalty(W, H):
    p = 2.0 * (W + H)
    return max(0.0, p - 4.0)


def _check_validity(circles, W, H):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -1e-9 or y - r < -1e-9 or x + r > W + 1e-9 or y + r > H + 1e-9:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - 1e-9:
                return False
    return True


def _objective(params, W, H):
    """Minimize negative sum of radii + penalties for overlaps and boundary violations."""
    circles = params.reshape(N, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    # Penalty for negative radii
    penalty = 100.0 * np.sum(np.maximum(0, -r))

    # Penalty for going outside rectangle
    penalty += 100.0 * np.sum(np.maximum(0, -x + r))
    penalty += 100.0 * np.sum(np.maximum(0, -y + r))
    penalty += 100.0 * np.sum(np.maximum(0, x - W + r))
    penalty += 100.0 * np.sum(np.maximum(0, y - H + r))

    # Penalty for overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += 10.0 * overlap * overlap

    # Also penalize if perimeter > 4
    p = 2.0 * (W + H)
    if p > 4.0:
        penalty += 1000.0 * (p - 4.0) ** 2

    return -np.sum(r) + penalty


def _local_optimize(circles, W, H, max_iter=5000):
    """Use Nelder-Mead to locally optimize the circle positions and radii."""
    params = circles.flatten()
    result = minimize(
        _objective, params, args=(W, H),
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    )
    return result.x.reshape(N, 3)


def _grid_initialization(W, H):
    """Create a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            r = min(W, H) / (2 * side) - 1e-4
            cells.append([cx, cy, r])
    return np.array(cells[:N])


def _random_initialization(W, H, rng):
    """Create a random initial configuration with small circles."""
    circles = []
    for _ in range(N):
        r = rng.uniform(0.01, min(W, H) / 6)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def _best_known(W, H):
    """Return the best configuration found so far for given W, H."""
    # Start with grid
    circles = _grid_initialization(W, H)
    best_val = -np.sum(circles[:, 2])
    if _check_validity(circles, W, H):
        best_circles = circles.copy()
    else:
        best_circles = circles.copy()

    # Try local optimization
    circles_opt = _local_optimize(circles, W, H, max_iter=3000)
    # Clip to valid range
    circles_opt[:, 0] = np.clip(circles_opt[:, 0], circles_opt[:, 2], W - circles_opt[:, 2])
    circles_opt[:, 1] = np.clip(circles_opt[:, 1], circles_opt[:, 2], H - circles_opt[:, 2])
    circles_opt[:, 2] = np.clip(circles_opt[:, 2], 0.001, min(W, H) / 2)

    val = -np.sum(circles_opt[:, 2])
    if _check_validity(circles_opt, W, H) and val < best_val:
        best_val = val
        best_circles = circles_opt.copy()

    return best_circles, -best_val


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None

    # Try different aspect ratios
    # Perimeter = 2(W+H) = 4, so W+H = 2
    # Aspect ratio a = W/H, so W = a*H, H(a+1) = 2, H = 2/(a+1), W = 2a/(a+1)
    aspect_ratios = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 3.0]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE - 10:
            break
        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        # Grid initialization + local optimization
        circles, score = _best_known(W, H)
        if score > best_score:
            best_score = score
            best_result = circles.copy()

        # Try a few random restarts with local optimization
        n_restarts = 3
        for _ in range(n_restarts):
            if time.time() - start_time > DEADLINE - 10:
                break
            circ_rand = _random_initialization(W, H, rng)
            circ_opt = _local_optimize(circ_rand, W, H, max_iter=2000)
            circ_opt[:, 0] = np.clip(circ_opt[:, 0], circ_opt[:, 2], W - circ_opt[:, 2])
            circ_opt[:, 1] = np.clip(circ_opt[:, 1], circ_opt[:, 2], H - circ_opt[:, 2])
            circ_opt[:, 2] = np.clip(circ_opt[:, 2], 0.001, min(W, H) / 2)

            if _check_validity(circ_opt, W, H):
                s = np.sum(circ_opt[:, 2])
                if s > best_score:
                    best_score = s
                    best_result = circ_opt.copy()

    # Final fallback: simple grid in square
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])

    # Ensure the result is valid; if not, fall back to a safe grid
    if not _check_validity(best_result, 1.0, 1.0):
        # Try to find W, H for the best result
        W = 2.0 / 2.0  # default square
        H = 2.0 / 2.0
        # Check with the aspect ratio that was used
        # For simplicity, just use a safe grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])

    return best_result
# EVOLVE-BLOCK-END
