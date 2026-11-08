"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 21
DEADLINE = time.time() + 140.0  # safety margin before 150s kill
EPS = 1e-10


def _compute_radii(centers, W, H):
    """Compute the max disjoint radii for given centers in a W x H rectangle.
    r_i = min over j!=i of dist(i,j)/2, and min over sides of dist-to-side.
    """
    n = len(centers)
    if n == 0:
        return np.array([])
    d = np.sqrt(np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    radii = d.min(axis=1) / 2.0
    # boundary distances
    radii = np.minimum(radii, centers[:, 0])
    radii = np.minimum(radii, centers[:, 1])
    radii = np.minimum(radii, W - centers[:, 0])
    radii = np.minimum(radii, H - centers[:, 1])
    return np.maximum(radii, 0.0)


def _sum_radii(centers, W, H):
    return _compute_radii(centers, W, H).sum()


def _validate(centers, W, H):
    """Check that all circles are inside the rectangle with disjoint interiors."""
    n = len(centers)
    r = _compute_radii(centers, W, H)
    # every radius must be positive
    if np.any(r <= 0):
        return False
    # check pairwise non-overlap
    d = np.sqrt(np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2))
    if np.any(d < np.outer(r, r) - 1e-6):
        return False
    # check inside rectangle
    if np.any(centers[:, 0] < -1e-9) or np.any(centers[:, 1] < -1e-9):
        return False
    if np.any(centers[:, 0] > W + 1e-9) or np.any(centers[:, 1] > H + 1e-9):
        return False
    return True


def _objective(params):
    """Objective for L-BFGS-B: minimize negative sum of radii.
    params = [W, x_0, y_0, ..., x_{n-1}, y_{n-1}]
    """
    W = params[0]
    H = 2.0 - W  # perimeter 4 => 2(W+H)=4 => H=2-W
    if W <= 0 or H <= 0:
        return 1e10
    centers = params[1:].reshape(-1, 2)
    # penalize out-of-bounds
    penalty = 0.0
    penalty += np.sum(np.maximum(0, -centers[:, 0]))
    penalty += np.sum(np.maximum(0, -centers[:, 1]))
    penalty += np.sum(np.maximum(0, centers[:, 0] - W))
    penalty += np.sum(np.maximum(0, centers[:, 1] - H))
    if penalty > 0:
        return 1e6 * penalty - _sum_radii(centers, W, H)
    return -_sum_radii(centers, W, H)


def _sa_step(centers, W, H, T, rng):
    """One SA step: perturb a random circle, accept if better or with probability exp(-dE/T)."""
    n = len(centers)
    idx = rng.integers(n)
    old = centers[idx].copy()
    # random perturbation
    step = 0.05 * T
    centers[idx, 0] += rng.normal(0, step)
    centers[idx, 1] += rng.normal(0, step)
    # clamp to bounds
    centers[idx, 0] = np.clip(centers[idx, 0], 0, W)
    centers[idx, 1] = np.clip(centers[idx, 1], 0, H)
    new_sum = _sum_radii(centers, W, H)
    old_sum = _sum_radii(np.where(np.arange(n) == idx, old, centers), W, H)
    dE = new_sum - old_sum  # we maximize, so dE>0 is good
    if dE >= 0 or rng.random() < np.exp(-dE / max(T, 1e-10)):
        return centers, new_sum
    else:
        centers[idx] = old
        return centers, old_sum


def _sa_optimize(centers, W, H, T0, T1, n_iter, rng):
    """Simulated annealing to optimize circle positions for fixed W, H."""
    best_sum = _sum_radii(centers, W, H)
    best_centers = centers.copy()
    current_sum = best_sum
    for it in range(n_iter):
        T = T0 * (T1 / T0) ** (it / max(n_iter - 1, 1))
        centers, current_sum = _sa_step(centers, W, H, T, rng)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
        if time.time() > DEADLINE - 15:
            break
    return best_centers, best_sum


def _lbgfb_optimize(centers, W, H, rng):
    """L-BFGS-B refinement of the full parameter vector (W + centers)."""
    params = np.concatenate([[W], centers.ravel()])
    n = len(centers)
    bounds = [(0.5, 1.5)] + [(0, W)] * n + [(0, H)] * n
    # Rebuild bounds properly
    bounds = [(0.5, 1.5)]
    for i in range(n):
        bounds.append((0, W))
    for i in range(n):
        bounds.append((0, H))
    res = minimize(_objective, params, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
    W_new = res.x[0]
    H_new = 2.0 - W_new
    centers_new = res.x[1:].reshape(-1, 2)
    # clamp
    centers_new[:, 0] = np.clip(centers_new[:, 0], 0, W_new)
    centers_new[:, 1] = np.clip(centers_new[:, 1], 0, H_new)
    return centers_new, W_new, H_new, -res.fun


def _make_grid(n, W, H):
    """Initial grid placement."""
    cols = int(np.ceil(np.sqrt(n * W / H)))
    rows = int(np.ceil(n / cols))
    centers = []
    for i in range(n):
        c = i % cols
        r = i // cols
        centers.append([(c + 0.5) * W / cols, (r + 0.5) * H / rows])
    return np.array(centers[:n])


def construct():
    rng = np.random.default_rng(42)
    best_overall = None
    best_score = -1.0

    # Try several aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.67, 0.5, 2.0, 0.4]
    # W + H = 2, so W = 2*ar/(1+ar)
    for ar in aspect_ratios:
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 - W
        if W <= 0 or H <= 0:
            continue

        centers = _make_grid(N, W, H)

        # Phase 1: SA for global exploration
        centers, sa_sum = _sa_optimize(centers, W, H, T0=0.3, T1=0.01, n_iter=300, rng=rng)

        # Phase 2: L-BFGS-B refinement
        try:
            centers2, W2, H2, score = _lbgfb_optimize(centers, W, H, rng)
            if score > best_score:
                best_score = score
                best_overall = (centers2, W2, H2)
        except Exception:
            if sa_sum > best_score:
                best_score = sa_sum
                best_overall = (centers, W, H)

        if time.time() > DEADLINE - 30:
            break

    if best_overall is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    centers, W, H = best_overall
    radii = _compute_radii(centers, W, H)
    # Build result: [x, y, r]
    result = np.column_stack([centers, radii])
    return result


if __name__ == "__main__":
    result = construct()
    print(result.shape)
    print(result)
