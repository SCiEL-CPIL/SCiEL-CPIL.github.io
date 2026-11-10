"""Ring-based packing of 300 equal circles in the unit disk."""
import numpy as np
import time

N = 300


def _score(points):
    """Compute the packing radius for a set of points."""
    if len(points) != N:
        return -1.0
    dists = np.linalg.norm(points, axis=1)
    if np.any(dists >= 1.0):
        return -1.0
    min_boundary = 1.0 - np.max(dists)
    idx_i, idx_j = np.triu_indices(N, k=1)
    diffs = points[idx_i] - points[idx_j]
    pair_dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    min_pair = np.min(pair_dists)
    return min(min_boundary, min_pair / 2.0)


def _build_ring_points(config, radii, offsets):
    """Build points from ring configuration."""
    pts = [np.array([0.0, 0.0])]
    for i in range(len(radii)):
        n = config[i + 1]
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + offsets[i]
        x = radii[i] * np.cos(angles)
        y = radii[i] * np.sin(angles)
        pts.append(np.column_stack([x, y]))
    return np.vstack(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    best_score = 0.0
    best_points = None

    # Get hexagonal lattice as baseline
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points_hex = np.array(rows)
        radius_hex = np.hypot(points_hex[:, 0], points_hex[:, 1])
        mask = radius_hex <= 1.0 - spacing / 2.0
        points_hex = points_hex[mask]
        if len(points_hex) >= N:
            points_hex = points_hex[np.argsort(radius_hex, kind="stable")[:N]]
            break
        spacing *= 0.98

    hex_score = _score(points_hex)
    if hex_score > best_score:
        best_score = hex_score
        best_points = points_hex.copy()

    # Try ring-based configurations
    configs = []

    # Hand-crafted configurations
    configs.append([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29])
    configs.append([1, 8, 16, 24, 32, 40, 48, 56, 64, 11])
    configs.append([1, 6, 12, 18, 24, 30, 36, 42, 48, 53, 30])

    # Generate more configurations
    for num_rings in range(8, 15):
        remaining = 299
        weights = np.arange(1, num_rings + 1, dtype=float)
        weights /= weights.sum()
        counts = np.round(remaining * weights).astype(int)
        diff = remaining - counts.sum()
        if diff > 0:
            for i in range(diff):
                counts[i % num_rings] += 1
        elif diff < 0:
            for i in range(-diff):
                idx = i % num_rings
                if counts[idx] > 1:
                    counts[idx] -= 1
        counts = np.maximum(counts, 1)
        diff = remaining - counts.sum()
        if diff != 0:
            idx = np.argmax(counts) if diff > 0 else np.argmin(counts)
            counts[idx] += diff
        config = [1] + list(counts)
        if sum(config) == 300 and len(config) == num_rings:
            configs.append(config)

    # Remove duplicates
    unique_configs = []
    seen = set()
    for c in configs:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)
    configs = unique_configs

    for config in configs:
        if time.time() > deadline:
            break

        num_rings = len(config)
        if num_rings < 2:
            continue

        r_est = 0.053
        outer_radius = 1.0 - r_est

        radii = np.linspace(0.1, outer_radius, num_rings - 1)
        offsets = np.zeros(num_rings - 1)

        current_points = _build_ring_points(config, radii, offsets)
        current_score = _score(current_points)

        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()

        # Optimize radii
        for iteration in range(30):
            if time.time() > deadline:
                break

            improved = False
            for i in range(len(radii)):
                for delta in [-0.01, -0.005, -0.001, 0.001, 0.005, 0.01]:
                    new_radii = radii.copy()
                    new_radii[i] += delta
                    if new_radii[i] < 0.01 or new_radii[i] > 0.99:
                        continue
                    if i > 0 and new_radii[i] <= new_radii[i - 1] + 0.005:
                        continue
                    if i < len(radii) - 1 and new_radii[i] >= new_radii[i + 1] - 0.005:
                        continue

                    new_points = _build_ring_points(config, new_radii, offsets)
                    new_score = _score(new_points)
                    if new_score > current_score:
                        radii = new_radii
                        current_points = new_points
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_points = current_points.copy()
                        break

            if not improved:
                break

        # Optimize offsets
        for iteration in range(15):
            if time.time() > deadline:
                break

            improved = False
            for i in range(len(offsets)):
                for delta in [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1]:
                    new_offsets = offsets.copy()
                    new_offsets[i] += delta
                    new_points = _build_ring_points(config, radii, new_offsets)
                    new_score = _score(new_points)
                    if new_score > current_score:
                        offsets = new_offsets
                        current_points = new_points
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_points = current_points.copy()
                        break

            if not improved:
                break

    if best_points is not None:
        return best_points
    return points_hex
