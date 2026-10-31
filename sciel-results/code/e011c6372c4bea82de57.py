"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time

N = 26


def _make_initial(seed: int):
    """Create initial configuration with random jittered grid."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(N)))
    base = []
    for i in range(side):
        for j in range(side):
            if len(base) >= N:
                break
            base.append(((i + 0.5) / side, (j + 0.5) / side))
    base = base[:N]
    xs = np.array([b[0] for b in base])
    ys = np.array([b[1] for b in base])
    # Add small random perturbation
    xs += rng.uniform(-0.02, 0.02, size=N)
    ys += rng.uniform(-0.02, 0.02, size=N)
    # Initial radii: small
    r = np.full(N, 0.04)
    return xs, ys, r


def _compute_forces(xs, ys, r, dt=0.01, k_repel=50.0, k_wall=100.0, k_radius=1.0):
    """
    Compute gradient of a smooth objective:
    - Maximize sum(r)
    - Penalize overlaps (smooth)
    - Penalize wall violations
    """
    n = len(xs)
    # Pairwise distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, 1.0)  # avoid div by zero

    # Overlap penalty: for each pair, if dist < r_i + r_j, penalize
    # Use smooth soft-min: penalty = max(0, (r_i + r_j - dist))^2
    # Gradient w.r.t. positions and radii
    rsum = r[:, None] + r[None, :]
    overlap = np.maximum(0, rsum - dist)
    # Avoid zero division
    dist_safe = np.where(dist > 1e-8, dist, 1e-8)
    # Unit vectors
    ux = dx / dist_safe
    uy = dy / dist_safe

    # Gradient of -sum(overlap^2) w.r.t. positions (we want to minimize overlap)
    # d/dx_i of overlap_ij^2 = 2 * overlap_ij * (-(r_i+r_j - dist_ij) * d(dist)/dx_i)
    # Actually: overlap = max(0, r_i+r_j - dist)
    # d(overlap^2)/dx_i = 2*overlap * d(overlap)/dx_i
    # d(overlap)/dx_i = -d(dist)/dx_i = -ux_ij (for i, j pair, moving i toward j decreases dist)
    # Wait: dist_ij = |x_i - x_j|, d(dist_ij)/dx_i = (x_i - x_j)/dist = ux_ij
    # overlap_ij = r_i + r_j - dist_ij
    # d(overlap_ij)/dx_i = -ux_ij
    # d(overlap_ij^2)/dx_i = 2 * overlap_ij * (-ux_ij)
    # We want to MINIMIZE overlap, so gradient ascent on -overlap^2 means:
    # move in direction of -d(overlap^2)/dx = +2*overlap*ux
    # So force on i from j is +2*overlap_ij * ux_ij (pushing i away from j when overlapping)

    # Sum over j for each i
    # Force on i: sum_j 2 * overlap_ij * ux_ij (for i != j)
    force_x = 2.0 * np.sum(overlap * ux, axis=1)
    force_y = 2.0 * np.sum(overlap * uy, axis=1)

    # Wall constraints: circles must be inside [0,1]^2
    # Left wall: x_i - r_i >= 0 -> violation if x_i < r_i
    # Right wall: 1 - x_i - r_i >= 0 -> violation if x_i + r_i > 1
    # Bottom: y_i - r_i >= 0
    # Top: 1 - y_i - r_i >= 0
    # Penalty: max(0, r_i - x_i)^2 + max(0, x_i + r_i - 1)^2 + ...
    # Gradient w.r.t. x_i:
    # d/dx_i max(0, r_i - x_i)^2 = 2 * max(0, r_i - x_i) * (-1) = -2 * max(0, r_i - x_i)
    # We want to minimize penalty, so move in direction of -grad = +2*max(0, r_i-x_i)
    # This pushes x_i to the right (away from left wall)
    left_viol = np.maximum(0, r - xs)
    right_viol = np.maximum(0, xs + r - 1)
    bottom_viol = np.maximum(0, r - ys)
    top_viol = np.maximum(0, ys + r - 1)

    force_x += k_wall * (2.0 * left_viol - 2.0 * right_viol)
    force_y += k_wall * (2.0 * bottom_viol - 2.0 * top_viol)

    # Radius gradient: we want to maximize sum(r), but penalize overlaps and wall violations
    # d(overlap^2)/dr_i = sum_j 2 * overlap_ij * 1 (since d(overlap_ij)/dr_i = 1)
    # So penalty on radius: sum_j 2 * overlap_ij
    # Wall penalty on radius:
    # d/d r_i max(0, r_i - x_i)^2 = 2 * max(0, r_i - x_i)
    # d/d r_i max(0, x_i + r_i - 1)^2 = 2 * max(0, x_i + r_i - 1)
    # etc.
    overlap_radius_pen = np.sum(2.0 * overlap, axis=1)
    wall_radius_pen = 2.0 * (left_viol + right_viol + bottom_viol + top_viol)

    # Total gradient for radius: 1 (from sum(r)) - k_repel * overlap_pen - k_wall * wall_pen
    # We do gradient ascent on the objective
    grad_r = 1.0 - k_repel * overlap_radius_pen - k_wall * wall_radius_pen

    return force_x, force_y, grad_r


def _enforce_constraints(xs, ys, r):
    """Hard project onto feasible set: no overlaps, inside square."""
    # Clamp to square
    for _ in range(10):
        # Wall constraints
        r = np.clip(r, 0.001, 0.5)
        xs = np.clip(xs, r, 1 - r)
        ys = np.clip(ys, r, 1 - r)

        # Pairwise separation: push apart overlapping circles
        improved = True
        while improved:
            improved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[j] - xs[i]
                    dy = ys[j] - ys[i]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist - 1e-9:
                        # Push apart
                        if dist < 1e-8:
                            # Random direction
                            angle = np.random.uniform(0, 2 * np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1.0
                        else:
                            dist = max(dist, 1e-8)
                        # Equal push (or weighted by radius)
                        push = (min_dist - dist) / 2.0
                        ux = dx / dist
                        uy = dy / dist
                        xs[i] -= push * ux
                        ys[i] -= push * uy
                        xs[j] += push * ux
                        ys[j] += push * uy
                        improved = True
                        # Re-clamp to square
                        xs[i] = np.clip(xs[i], r[i], 1 - r[i])
                        ys[i] = np.clip(ys[i], r[i], 1 - r[i])
                        xs[j] = np.clip(xs[j], r[j], 1 - r[j])
                        ys[j] = np.clip(ys[j], r[j], 1 - r[j])
    return xs, ys, r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    best_score = -np.inf
    best_xs = None
    best_ys = None
    best_r = None

    # Try multiple random restarts
    for seed in range(50):
        if time.time() > deadline - 5.0:
            break

        xs, ys, r = _make_initial(seed)
        # Initial projection
        xs, ys, r = _enforce_constraints(xs, ys, r)

        # Gradient-based optimization
        lr = 0.005
        for step in range(5000):
            if time.time() > deadline - 2.0:
                break

            force_x, force_y, grad_r = _compute_forces(xs, ys, r)

            # Update
            xs += lr * force_x
            ys += lr * force_y
            r += lr * grad_r

            # Clamp radii to reasonable range
            r = np.clip(r, 0.005, 0.45)

            # Project to feasible set
            xs, ys, r = _enforce_constraints(xs, ys, r)

        # Final polish: more iterations with smaller step
        for step in range(2000):
            if time.time() > deadline - 1.0:
                break
            force_x, force_y, grad_r = _compute_forces(xs, ys, r)
            xs += 0.002 * force_x
            ys += 0.002 * force_y
            r += 0.002 * grad_r
            r = np.clip(r, 0.005, 0.45)
            xs, ys, r = _enforce_constraints(xs, ys, r)

        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_xs = xs.copy()
            best_ys = ys.copy()
            best_r = r.copy()

    # If no valid result, fallback to grid
    if best_xs is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return np.column_stack([best_xs, best_ys, best_r])
