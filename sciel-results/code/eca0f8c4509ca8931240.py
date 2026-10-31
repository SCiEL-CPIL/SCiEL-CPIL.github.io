"""Circle packing in perimeter-4 rectangle: DE + local refinement hybrid.

Strategy:
  1. Use differential evolution to globally explore (aspect_ratio + 21 centers)
     with an objective that is the negative sum of maximum feasible radii.
  2. After DE, run a coordinate-descent local refinement to squeeze out extra
     sum-of-radii by nudging centers.
  3. Return the best feasible packing found.
"""
import numpy as np
import time
import random

N = 21
START = time.time()
DEADLINE = START + 135.0  # safety margin before 150 s kill


def _time_ok():
    return time.time() < DEADLINE


def _max_radii(centers, w, h):
    """Compute the maximum feasible radius for each circle given centers and box.
    Returns array of radii (non-negative)."""
    n = len(centers)
    radii = np.zeros(n)
    for i in range(n):
        xi, yi = centers[i]
        r = min(xi, w - xi, yi, h - yi)
        if r <= 0:
            radii[i] = 0.0
            continue
        for j in range(n):
            if i == j:
                continue
            dx = xi - centers[j][0]
            dy = yi - centers[j][1]
            d = np.sqrt(dx * dx + dy * dy)
            r = min(r, d / 2.0)
            if r <= 0:
                break
        radii[i] = max(0.0, r)
    return radii


def _objective(x):
    """Objective: negative sum of max feasible radii.
    x = [aspect_ratio, x1, y1, ..., xN, yN]
    aspect_ratio = w, h = 2 - w.
    """
    w = x[0]
    h = 2.0 - w
    if w < 0.2 or h < 0.2:
        return 1e6
    centers = x[1:].reshape(N, 2)
    # Clamp centers into box
    centers = np.clip(centers, 0.0, np.array([w, h]))
    radii = _max_radii(centers, w, h)
    return -np.sum(radii)


def _differential_evolution(objective, bounds, pop_size=35, max_iter=300,
                            mutation=(0.5, 1.0), crossover_prob=0.7,
                            time_limit=100.0):
    """Simple DE. Returns (best_x, best_f)."""
    n_dims = len(bounds)
    start = time.time()

    pop = np.zeros((pop_size, n_dims))
    for i in range(pop_size):
        for j in range(n_dims):
            pop[i, j] = random.uniform(bounds[j][0], bounds[j][1])

    fitness = np.array([objective(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]

    for gen in range(max_iter):
        if time.time() - start > time_limit:
            break

        new_pop = []
        new_fitness = []
        for i in range(pop_size):
            r1, r2, r3 = random.sample(range(pop_size), 3)
            F = random.uniform(*mutation)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            for j in range(n_dims):
                mutant[j] = np.clip(mutant[j], bounds[j][0], bounds[j][1])
            trial = pop[i].copy()
            for j in range(n_dims):
                if random.random() < crossover_prob:
                    trial[j] = mutant[j]
            f_trial = objective(trial)
            if f_trial <= fitness[i]:
                new_pop.append(trial)
                new_fitness.append(f_trial)
            else:
                new_pop.append(pop[i])
                new_fitness.append(fitness[i])

        pop = np.array(new_pop)
        fitness = np.array(new_fitness)
        gi = np.argmin(fitness)
        if fitness[gi] < best_f:
            best_f = fitness[gi]
            best_x = pop[gi].copy()

    return best_x, best_f


def _local_refine(centers, w, h, max_rounds=120, step_size=0.015):
    """Coordinate-descent refinement: nudge centers to increase sum of max radii."""
    centers = centers.copy()
    current_radii = _max_radii(centers, w, h)
    current_score = np.sum(current_radii)

    for _ in range(max_rounds):
        if not _time_ok():
            break
        improved = False
        for i in range(N):
            xi, yi = centers[i]
            # Try x perturbations
            for dx in (-step_size, step_size):
                nx = xi + dx
                if 0.0 < nx < w:
                    tc = centers.copy()
                    tc[i][0] = nx
                    tr = _max_radii(tc, w, h)
                    if np.sum(tr) > current_score + 1e-12:
                        centers = tc
                        current_radii = tr
                        current_score = np.sum(tr)
                        improved = True
            # Try y perturbations
            for dy in (-step_size, step_size):
                ny = yi + dy
                if 0.0 < ny < h:
                    tc = centers.copy()
                    tc[i][1] = ny
                    tr = _max_radii(tc, w, h)
                    if np.sum(tr) > current_score + 1e-12:
                        centers = tc
                        current_radii = tr
                        current_score = np.sum(tr)
                        improved = True
        if not improved:
            step_size *= 0.5
            if step_size < 1e-7:
                break

    return centers, current_radii, current_score


def _check_validity(centers, w, h, radii):
    """Verify circles are disjoint and inside box. Returns True if valid."""
    for i in range(N):
        xi, yi = centers[i]
        ri = radii[i]
        if xi < -1e-9 or xi > w + 1e-9 or yi < -1e-9 or yi > h + 1e-9:
            return False
        if ri < 1e-9:
            return False
        if ri > min(xi, w - xi, yi, h - yi) + 1e-9:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
            if d < radii[i] + radii[j] - 1e-9:
                return False
    return True


def _grid_init(w, h):
    """Generate a reasonable grid initial layout."""
    # Try a few grid configurations
    best_centers = None
    best_score = -1
    for kx, ky in [(3, 4), (4, 3), (3, 3), (4, 4), (5, 3), (3, 5)]:
        if kx * ky < N:
            continue
        cx = np.linspace(0.0, w, kx + 2)[1:-1]
        cy = np.linspace(0.0, h, ky + 2)[1:-1]
        pts = []
        for yy in cy:
            for xx in cx:
                pts.append([xx, yy])
        if len(pts) >= N:
            centers = np.array(pts[:N])
            radii = _max_radii(centers, w, h)
            s = np.sum(radii)
            if s > best_score:
                best_score = s
                best_centers = centers
    if best_centers is None:
        # Fallback: uniform grid
        kx, ky = 5, 5
        cx = np.linspace(0.1, w - 0.1, kx)
        cy = np.linspace(0.1, h - 0.1, ky)
        pts = np.array([[x, y] for y in cy for x in cx])
        best_centers = pts[:N]
    return best_centers


def construct():
    best_score = 0.0
    best_arr = None

    # Multi-start with different aspect ratios
    for w0 in [1.0, 1.2, 1.3, 1.4, 1.5]:
        if not _time_ok():
            break
        h0 = 2.0 - w0
        if h0 < 0.2:
            continue

        # Grid init
        centers0 = _grid_init(w0, h0)

        # Build bounds
        bounds = [(max(0.3, w0 - 0.3), min(1.7, w0 + 0.3))]
        for _ in range(N * 2):
            bounds.append((0.01, 1.99))

        # DE
        x0 = np.concatenate([[w0], centers0.flatten()])
        # Perturb for diversity
        x0 = x0 + 0.05 * np.random.randn(len(x0))
        x0[0] = np.clip(x0[0], 0.3, 1.7)

        time_left = DEADLINE - time.time() - 15.0  # leave 15 s for refinement
        if time_left < 10:
            continue

        pop_size = 30 if time_left > 60 else 20
        max_iter = 200 if time_left > 60 else 100

        try:
            best_x, best_f = _differential_evolution(
                _objective, bounds,
                pop_size=pop_size,
                max_iter=max_iter,
                mutation=(0.5, 1.0),
                crossover_prob=0.7,
                time_limit=time_left
            )
        except Exception:
            continue

        w = best_x[0]
        h = 2.0 - w
        centers = best_x[1:].reshape(N, 2)
        centers = np.clip(centers, 0.0, np.array([w, h]))

        # Local refinement
        if _time_ok():
            centers, radii, score = _local_refine(centers, w, h,
                                                   max_rounds=100, step_size=0.012)
        else:
            radii = _max_radii(centers, w, h)
            score = np.sum(radii)

        # Validate
        if _check_validity(centers, w, h, radii):
            if score > best_score:
                best_score = score
                best_arr = np.column_stack([centers, radii])

    # Fallback if nothing valid found
    if best_arr is None:
        w = h = 1.0
        centers = _grid_init(w, h)
        radii = _max_radii(centers, w, h)
        best_arr = np.column_stack([centers, radii])
        best_score = np.sum(radii)

    # Final safety: ensure output shape is (N, 3)
    out = np.array(best_arr[:N])
    return out
