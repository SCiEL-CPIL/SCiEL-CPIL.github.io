"""Packing 50 equal circles in a unit circle: hex init + force optimization + soft-min polish."""
import numpy as np
import time

N = 50
_START = time.time()
_DEADLINE = _START + 140.0  # 10s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_cont, r_pair)


def _hex_lattice_init(n, jitter=0.0):
    """Generate initial positions on a hexagonal lattice, then select n closest to center."""
    spacing = 0.2
    points = []
    k = 10
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r < 0.95:
                points.append((x, y))

    points = np.array(points)
    dists = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(dists, kind='stable')[:n]
    pts = points[idx].copy()
    if jitter > 0:
        pts += np.random.randn(n, 2) * jitter
    return pts


def _soft_min(vals, beta):
    """Soft-min using log-sum-exp for smoothness."""
    # soft_min ≈ -log(sum(exp(-beta*v)))/beta
    v = -beta * vals
    v_max = np.max(v)
    s = np.sum(np.exp(v - v_max))
    return -(v_max + np.log(s)) / beta


def _force_optimize(positions, max_time=90.0):
    """Optimize positions using force-based relaxation with time limit."""
    start_time = time.time()
    positions = positions.copy()
    n = len(positions)

    current_r = _score(positions)
    best_r = current_r
    best_positions = positions.copy()

    step = 0.05
    lr = 0.995

    while time.time() - start_time < max_time:
        if time.time() > _DEADLINE - 15.0:
            break

        forces = np.zeros_like(positions)

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[i] - positions[j]
                dist = np.sqrt(np.sum(diff ** 2))
                if dist < 1e-10:
                    forces[i] += np.random.randn(2) * 0.01
                    forces[j] -= np.random.randn(2) * 0.01
                    continue
                min_dist = 2.0 * current_r
                if dist < min_dist:
                    force_mag = (min_dist - dist) / max(dist, 1e-10)
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist

        # Boundary constraint: push inward if outside
        for i in range(n):
            r = np.hypot(positions[i, 0], positions[i, 1])
            max_r = 1.0 - current_r
            if r > max_r:
                force_mag = (r - max_r) / max(r, 1e-10)
                forces[i] -= force_mag * positions[i] / r

        # Apply forces with step size
        positions += step * forces

        # Keep inside container (hard constraint)
        radii = np.hypot(positions[:, 0], positions[:, 1])
        max_allowed = 1.0 - current_r * 0.99
        mask = radii > max_allowed
        if np.any(mask):
            positions[mask] *= max_allowed / radii[mask]

        # Update radius estimate
        new_r = _score(positions)
        if new_r > best_r:
            best_r = new_r
            best_positions = positions.copy()

        current_r = max(new_r, best_r * 0.99)

        # Decay step size
        step *= lr

        # If step gets too small, try a restart with larger step
        if step < 0.001:
            positions = best_positions + np.random.randn(n, 2) * 0.01
            step = 0.02
            current_r = best_r

    return best_positions, best_r


def _soft_min_polish(positions, max_time=20.0):
    """Final high-precision polish using soft-min surrogate with increasing beta."""
    start_time = time.time()
    positions = positions.copy()
    n = len(positions)
    best_r = _score(positions)
    best_positions = positions.copy()

    # Progressive beta values for soft-min
    betas = [50, 200, 500, 1000, 2000, 5000]
    gtol = 1e-9

    for beta in betas:
        if time.time() > _DEADLINE - 5.0:
            break
        if time.time() - start_time > max_time:
            break

        # Gradient descent on soft-min objective
        lr = 0.002
        for _ in range(500):
            if time.time() > _DEADLINE - 5.0:
                break

            # Compute constraints: pairwise distances/2 and boundary distances
            # Pairwise
            dx = positions[:, 0:1] - positions[0:1, 0]
            dy = positions[:, 1:2] - positions[0:1, 1]
            dmat = np.sqrt(dx ** 2 + dy ** 2)
            np.fill_diagonal(dmat, np.inf)
            pair_vals = dmat / 2.0

            # Boundary
            radii = np.hypot(positions[:, 0], positions[:, 1])
            bound_vals = 1.0 - radii

            # Soft-min over all constraints
            all_vals = np.concatenate([pair_vals.flatten(), bound_vals])
            r_soft = _soft_min(all_vals, beta)

            # Gradient with respect to positions (approximate via finite differences or analytic)
            # Use subgradient approach: weight each constraint by its softmax weight
            # For pairwise i,j: gradient of d_ij/2 w.r.t. pos_i = (pos_i - pos_j)/d_ij
            # For boundary i: gradient of (1-|pos_i|) w.r.t. pos_i = -pos_i/|pos_i|

            # Compute softmax weights
            v = -beta * all_vals
            v_max = np.max(v)
            exp_v = np.exp(v - v_max)
            weights = exp_v / np.sum(exp_v)

            grad = np.zeros_like(positions)

            # Pairwise gradients
            # all_vals layout: [pair_vals (n*(n-1)/2 flattened), bound_vals (n)]
            # Reconstruct pairwise from dmat
            n_pairs = n * (n - 1) // 2
            pair_weights = weights[:n_pairs]
            bound_weights = weights[n_pairs:]

            # Map flattened pair indices back to (i, j)
            pair_idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    w = pair_weights[pair_idx]
                    if w > 1e-12:
                        diff = positions[i] - positions[j]
                        dist = np.sqrt(np.sum(diff ** 2))
                        if dist > 1e-10:
                            g = w * diff / dist
                            grad[i] += g
                            grad[j] -= g
                    pair_idx += 1

            # Boundary gradients
            for i in range(n):
                w = bound_weights[i]
                if w > 1e-12:
                    r = np.hypot(positions[i, 0], positions[i, 1])
                    if r > 1e-10:
                        grad[i] -= w * positions[i] / r

            # Update
            positions += lr * grad

            # Project inside container
            radii = np.hypot(positions[:, 0], positions[:, 1])
            max_allowed = 1.0 - r_soft * 0.999
            mask = radii > max_allowed
            if np.any(mask):
                positions[mask] *= max_allowed / radii[mask]

            # Check convergence
            if np.max(np.abs(lr * grad)) < gtol:
                break

        # Update best
        new_r = _score(positions)
        if new_r > best_r:
            best_r = new_r
            best_positions = positions.copy()

    return best_positions, best_r


def construct():
    """Construct positions for N circles in the unit disk."""
    # Initialize with hexagonal lattice
    positions = _hex_lattice_init(N, jitter=0.01)

    # Force-based optimization
    positions, best_r = _force_optimize(positions, max_time=90.0)

    # Soft-min polish pass
    if time.time() < _DEADLINE - 25.0:
        positions, polish_r = _soft_min_polish(positions, max_time=20.0)
        if polish_r > best_r:
            best_r = polish_r

    # Final validation: ensure all circles are inside
    radii = np.hypot(positions[:, 0], positions[:, 1])
    r_final = _score(positions)
    max_allowed = 1.0 - r_final * 0.999
    mask = radii > max_allowed
    if np.any(mask):
        positions[mask] *= max_allowed / radii[mask]

    return positions
