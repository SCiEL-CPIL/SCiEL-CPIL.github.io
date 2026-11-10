"""Rewritten solver: use a force-directed / repulsive optimization with boundary constraints.
Start from a reasonable grid and then optimize positions to maximize the minimum distance."""
import numpy as np
import time

N = 100


def _min_radius(centers):
    """Compute the maximum common radius for given centers inside [-0.5, 0.5]^3."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside cube
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    r = min(min_pair / 2.0, min_wall)
    return max(r, 0.0)


def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Initial placement: a jittered grid
    m = 5  # 5x5x5 = 125, we take 100
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].astype(np.float64)

    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.001, size=centers.shape)

    # Clamp to inside cube with small margin
    centers = np.clip(centers, -0.49, 0.49)

    best_centers = centers.copy()
    best_r = _min_radius(best_centers)

    # Force-directed optimization
    # Repulsive force between spheres: push apart
    # Attractive force toward center: keep inside cube
    # Boundary repulsion: push away from walls

    learning_rate = 0.001
    decay = 0.999
    min_lr = 1e-7

    for step in range(200000):
        if time.time() > deadline - 5:
            break
        if learning_rate < min_lr:
            break

        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e-10)  # avoid division by zero

        # Repulsive force: F = k * (1/d^2) * direction
        # Use a soft repulsion that is strong when close
        inv_dist = 1.0 / dists
        inv_dist2 = inv_dist * inv_dist
        # Force magnitude: proportional to 1/d^2, but capped
        force_mag = np.minimum(inv_dist2, 100.0)
        np.fill_diagonal(force_mag, 0.0)

        # Force vector: direction * magnitude
        forces = diff * (force_mag * inv_dist)[:, :, np.newaxis]
        net_forces = np.sum(forces, axis=1)

        # Boundary repulsion
        for axis in range(3):
            # Left wall
            left_dist = centers[:, axis] + 0.5
            left_force = np.where(left_dist < 0.1, 1.0 / (left_dist + 1e-6), 0.0)
            # Right wall
            right_dist = 0.5 - centers[:, axis]
            right_force = np.where(right_dist < 0.1, 1.0 / (right_dist + 1e-6), 0.0)
            net_forces[:, axis] += -left_force + right_force

        # Center attraction (weak) to keep spheres centered
        net_forces -= 0.01 * centers

        # Update positions
        centers += learning_rate * net_forces

        # Clamp to inside cube
        centers = np.clip(centers, -0.499, 0.499)

        # Decay learning rate
        learning_rate *= decay

        # Check improvement
        r = _min_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()

    # Final refinement: local optimization on the best configuration
    # Try a few more iterations with smaller steps
    centers = best_centers.copy()
    for step in range(50000):
        if time.time() > deadline - 2:
            break

        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e-10)

        inv_dist = 1.0 / dists
        inv_dist2 = inv_dist * inv_dist
        force_mag = np.minimum(inv_dist2, 100.0)
        np.fill_diagonal(force_mag, 0.0)

        forces = diff * (force_mag * inv_dist)[:, :, np.newaxis]
        net_forces = np.sum(forces, axis=1)

        for axis in range(3):
            left_dist = centers[:, axis] + 0.5
            left_force = np.where(left_dist < 0.05, 1.0 / (left_dist + 1e-6), 0.0)
            right_dist = 0.5 - centers[:, axis]
            right_force = np.where(right_dist < 0.05, 1.0 / (right_dist + 1e-6), 0.0)
            net_forces[:, axis] += -left_force + right_force

        net_forces -= 0.005 * centers

        centers += 1e-5 * net_forces
        centers = np.clip(centers, -0.499, 0.499)

        r = _min_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()

    # Final check and return
    final_r = _min_radius(best_centers)
    if final_r > best_r:
        best_r = final_r

    return best_centers
