import numpy as np
import time

N = 50


def _score(points):
    """Compute the packing radius for a set of points."""
    if len(points) != N:
        return -1.0
    if not np.all(np.isfinite(points)):
        return -1.0
    dists = np.sqrt(np.sum(points ** 2, axis=1))
    min_boundary = 1.0 - np.max(dists)
    if min_boundary <= 0:
        return -1.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    return min(min_boundary, min_pair)


def _hexagonal_init():
    """Generate a hexagonal lattice packing of 50 circles."""
    # Start with a hexagonal grid and select 50 points that fit in unit circle
    # Hexagonal packing: rows offset by half spacing
    best_points = None
    best_score = -1.0

    for spacing in np.arange(0.20, 0.28, 0.005):
        points = []
        row = 0
        while len(points) < N:
            y = row * spacing * np.sqrt(3.0) / 2.0
            if abs(y) > 1.0:
                break
            offset = 0.5 * spacing if row % 2 == 1 else 0.0
            col = -15
            while True:
                x = col * spacing + offset
                if abs(x) > 1.0:
                    break
                r = np.sqrt(x * x + y * y)
                if r + spacing / 2.0 <= 1.0:
                    points.append([x, y])
                col += 1
            row += 1
            if row > 30:
                break

        if len(points) >= N:
            # Select N points closest to center for better packing
            points = np.array(points[:N])
            # Sort by distance from center and take closest N
            dists = np.sqrt(np.sum(points ** 2, axis=1))
            idx = np.argsort(dists)
            points = points[idx[:N]]
            s = _score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()

    if best_points is None:
        # Fallback: simple ring
        pts = [[0.0, 0.0]]
        for ring in range(1, 10):
            n = 6 * ring
            for k in range(n):
                a = 2 * np.pi * k / n
                r = ring * 0.15
                pts.append([r * np.cos(a), r * np.sin(a)])
                if len(pts) >= N:
                    break
            if len(pts) >= N:
                break
        best_points = np.array(pts[:N])

    return best_points


def _anneal(points, max_time, seed=42):
    """Simulated annealing to improve packing."""
    rng = np.random.RandomState(seed)
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score

    # Adaptive temperature
    T = 0.02
    T_min = 1e-6
    cooling = 0.9997  # per step

    n_steps = 0
    max_steps = 500000
    start = time.time()

    while n_steps < max_steps:
        elapsed = time.time() - start
        if elapsed > max_time:
            break

        # Generate perturbation: move a random point (or a few points)
        # Bias toward the most constrained point
        n_pts = rng.randint(1, 4)  # move 1-3 points
        idx = rng.randint(0, N, size=n_pts)

        # Compute current constraints to bias perturbation
        dists = np.sqrt(np.sum(current ** 2, axis=1))
        boundary = 1.0 - dists

        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        pair_min = np.min(d2, axis=1) / 4.0  # min pair distance / 2 for each point

        # Constraint tightness: smaller means more constrained
        constraint = np.minimum(boundary, pair_min)
        # Probability proportional to how constrained (more constrained = more likely to move)
        weights = 1.0 / (constraint + 0.01)
        weights /= np.sum(weights)

        # Pick indices with probability based on constraint
        idx = rng.choice(N, size=n_pts, replace=False, p=weights)

        # Perturbation magnitude proportional to temperature
        step_size = T * rng.uniform(0.5, 2.0)
        for i in idx:
            current[i] += step_size * rng.randn(2)

        new_score = _score(current)

        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = current.copy()
        else:
            # Accept with probability exp((new - current) / T)
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_score = new_score

        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            # Reheat occasionally
            if rng.random() < 0.01:
                T = 0.005

        n_steps += 1

    return best, best_score


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Phase 1: Hexagonal initialization
    init_points = _hexagonal_init()
    best_points = init_points.copy()
    best_score = _score(best_points)

    # Phase 2: Multiple annealing runs with different seeds
    seeds = [42, 123, 7, 999, 31415]
    for i, seed in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5:
            break

        # Give more time to earlier runs
        time_budget = remaining * (1.0 / (len(seeds) - i)) * 0.9

        # For later runs, start from the best found so far with perturbation
        if i == 0:
            start_pts = init_points.copy()
        else:
            rng = np.random.RandomState(seed)
            start_pts = best_points + rng.randn(N, 2) * 0.005

        result_pts, result_score = _anneal(start_pts, max_time=time_budget, seed=seed)
        if result_score > best_score:
            best_score = result_score
            best_points = result_pts.copy()

    # Phase 3: Final local polish with very small perturbations
    elapsed = time.time() - start_time
    if elapsed < deadline - 3 and best_score > 0:
        # Tight local search: small random moves, only accept improvements
        current = best_points.copy()
        current_score = best_score
        rng = np.random.RandomState(2024)
        n_polish = 20000

        for _ in range(n_polish):
            if time.time() - start_time > deadline - 2:
                break
            idx = rng.randint(0, N)
            current[idx] += rng.randn(2) * 1e-4
            new_score = _score(current)
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = current.copy()
            else:
                # Revert
                current[idx] -= rng.randn(2) * 1e-4  # approximate revert (not exact)
                # Actually, let's just keep the worse point and try again
                # Better: store and revert properly
                pass

        # Redo polish more carefully
        current = best_points.copy()
        current_score = best_score
        rng2 = np.random.RandomState(555)
        for _ in range(10000):
            if time.time() - start_time > deadline - 2:
                break
            idx = rng2.randint(0, N)
            orig = current[idx].copy()
            current[idx] += rng2.randn(2) * 1e-5
            new_score = _score(current)
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = current.copy()
            else:
                current[idx] = orig

    # Final validation
    final_score = _score(best_points)
    if final_score <= 0:
        # Fallback
        best_points = init_points

    return best_points
