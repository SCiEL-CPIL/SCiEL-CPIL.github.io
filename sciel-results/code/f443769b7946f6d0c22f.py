"""Pack 100 equal spheres in a unit cube; maximize common radius."""
import time
import numpy as np

N = 100
DEADLINE = 135.0  # safety margin under 150s wall-clock


def _compute_radius(centers):
    """Return the largest common radius for given centers."""
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distance
    wall = 0.5 - np.abs(centers)
    r_wall = np.min(wall.min(axis=1))
    if r_wall <= 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    r_pair = 0.5 * np.min(dist)
    return min(r_wall, r_pair)


def _project_to_box(centers):
    """Clip centers into the box [-0.5, 0.5]^3."""
    return np.clip(centers, -0.5, 0.5)


def _initial_grid():
    """Simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _force_displacement(centers, r_target, alpha=0.01):
    """Compute a displacement that pushes overlapping spheres apart."""
    n = len(centers)
    disp = np.zeros_like(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)

    # For each pair that is too close, push apart
    for i in range(n):
        for j in range(i + 1, n):
            d = dist[i, j]
            if d < 2 * r_target and d > 1e-12:
                push = (2 * r_target - d) / d
                diff_ij = centers[i] - centers[j]
                disp[i] += alpha * push * diff_ij
                disp[j] -= alpha * push * diff_ij

    # Wall constraints: push inward if too close to wall
    for k in range(3):
        for i in range(n):
            if centers[i, k] > 0.5 - r_target:
                disp[i, k] -= alpha * (centers[i, k] - (0.5 - r_target))
            elif centers[i, k] < -0.5 + r_target:
                disp[i, k] += alpha * (-0.5 + r_target - centers[i, k])

    return disp


def _local_search(centers, max_iters=200, alpha_init=0.05):
    """Iterative local search: push apart overlapping spheres, clip to box."""
    centers = centers.copy()
    r = _compute_radius(centers)
    alpha = alpha_init
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        r = _compute_radius(centers)
        if r <= 0:
            break
        disp = _force_displacement(centers, r * 1.01, alpha)
        centers = _project_to_box(centers + disp)
        alpha *= 0.995
        new_r = _compute_radius(centers)
        if new_r <= r * (1 - 1e-10):
            # No improvement, reduce step
            alpha *= 0.9
            if alpha < 1e-6:
                break
        else:
            r = new_r
    return centers, r


def _perturb(centers, sigma=0.01):
    """Add random perturbation."""
    return centers + np.random.normal(0, sigma, size=centers.shape)


def _fourier_perturb(centers, scale=0.02):
    """Add a smooth Fourier-based perturbation for better exploration."""
    n = len(centers)
    t = np.random.uniform(0, 2 * np.pi, size=(n, 3))
    freqs = np.random.uniform(0.5, 3.0, size=(3,))
    phases = np.random.uniform(0, 2 * np.pi, size=(3,))
    perturb = scale * np.sin(t * freqs + phases)
    return centers + perturb


def construct():
    start_time = time.time()
    np.random.seed(42)

    # Start from grid
    centers = _initial_grid()
    best_centers = centers.copy()
    best_r = _compute_radius(centers)

    # Phase 1: Local search from grid
    centers, r = _local_search(centers, max_iters=300)
    if r > best_r:
        best_r = r
        best_centers = centers.copy()

    if time.time() > DEADLINE:
        return _project_to_box(best_centers)

    # Phase 2: Multi-restart with perturbation
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > DEADLINE:
            break

        # Perturb the best solution
        if restart % 3 == 0:
            perturbed = _perturb(best_centers, sigma=0.015)
        elif restart % 3 == 1:
            perturbed = _fourier_perturb(best_centers, scale=0.02)
        else:
            # Larger perturbation occasionally
            perturbed = _perturb(best_centers, sigma=0.04)

        perturbed = _project_to_box(perturbed)

        # Local search
        centers, r = _local_search(perturbed, max_iters=150)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()

    # Phase 3: Final polish with smaller steps
    if time.time() < DEADLINE - 5:
        centers, r = _local_search(best_centers, max_iters=500, alpha_init=0.01)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()

    # Ensure validity
    best_centers = _project_to_box(best_centers)
    final_r = _compute_radius(best_centers)
    if final_r <= 0:
        # Fallback to grid
        return _initial_grid()

    return best_centers
