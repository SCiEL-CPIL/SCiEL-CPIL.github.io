"""Improved packing: deterministic seed + gradient ascent on smooth surrogate."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """Compute the feasible common radius for given centers."""
    if len(centers) != N:
        return 0.0
    # Distance to boundary
    dists = 1.0 - np.sqrt(np.sum(centers**2, axis=1))
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    r = min(np.min(dists), min_pair / 2.0)
    if not np.isfinite(r) or r < 0:
        return 0.0
    return r


def _make_seed():
    """Create a good initial configuration using known packings heuristics."""
    # Use a multi-ring structure
    # Ring 0: center
    # Ring 1: 6 circles
    # Ring 2: 12 circles
    # Ring 3: 18 circles
    # Ring 4: 14 circles (to fill up to 50)
    # Total: 1+6+12+18+14 = 51, use 50

    centers = []
    # Center
    centers.append([0.0, 0.0])

    # Ring 1: 6 circles
    r1 = 0.25
    for i in range(6):
        angle = 2 * np.pi * i / 6
        centers.append([r1 * np.cos(angle), r1 * np.sin(angle)])

    # Ring 2: 12 circles
    r2 = 0.5
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        centers.append([r2 * np.cos(angle), r2 * np.sin(angle)])

    # Ring 3: 18 circles
    r3 = 0.75
    for i in range(18):
        angle = 2 * np.pi * i / 18
        centers.append([r3 * np.cos(angle), r3 * np.sin(angle)])

    # Ring 4: 13 circles (to make total 50)
    r4 = 0.92
    for i in range(13):
        angle = 2 * np.pi * i / 13 + np.pi / 13
        centers.append([r4 * np.cos(angle), r4 * np.sin(angle)])

    centers = np.array(centers[:N])
    return centers


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    centers = _make_seed()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()

    # Gradient ascent with smooth surrogate
    # Objective: maximize r = min(min_pair_dist/2, min_boundary_dist)
    # Use log-sum-exp soft-min to get gradients

    beta = 50.0  # temperature for soft-min (higher = sharper)
    lr = 0.001
    decay = 0.999
    momentum = 0.9
    velocity = np.zeros_like(centers)

    for iteration in range(20000):
        if time.time() > deadline:
            break

        # Compute distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(d, np.inf)

        boundary = 1.0 - np.sqrt(np.sum(centers**2, axis=1))

        # Soft-min of pair distances
        # We want to maximize the minimum, so we use -soft-min = soft-max of negatives
        # soft-min(x) = -1/beta * log(sum(exp(-beta * x)))
        # gradient of soft-min w.r.t. x_i = -sum_j exp(-beta*x_j) / sum_j exp(-beta*x_j) * (dx_i/dx)

        # For pair distances: d_ij = |c_i - c_j|
        # d d_ij / d c_i = (c_i - c_j) / d_ij

        # Boundary distances: b_i = 1 - |c_i|
        # d b_i / d c_i = -c_i / |c_i|

        # Compute soft-min for pairs
        # Flatten upper triangle
        iu = np.triu_indices(N, k=1)
        pair_dists = d[iu]

        # Soft-min weights
        exp_neg = np.exp(-beta * pair_dists)
        sum_exp = np.sum(exp_neg)
        weights = exp_neg / sum_exp  # these are the "attention" weights

        # Gradient of soft-min w.r.t. each center
        grad = np.zeros_like(centers)
        # For each pair (i,j), the contribution to grad[i] is -w_ij * (c_i - c_j)/d_ij
        # and to grad[j] is -w_ij * (c_j - c_i)/d_ij
        # But we want to MAXIMIZE the min, so we want to move AWAY from close pairs
        # The soft-min is a lower bound on the true min, and its gradient points
        # in the direction of decreasing the soft-min. So we want -gradient to increase it.

        # Actually, let's think more carefully.
        # f = soft_min(pair_dists) = -1/beta * log(sum(exp(-beta * d_ij)))
        # df/d(c_i) = sum over j of w_ij * (c_i - c_j) / d_ij  (where w_ij = exp(-beta*d_ij)/sum)
        # This is the gradient of the soft-min. To maximize f, we move in +gradient direction.

        for k in range(len(iu[0])):
            i, j = iu[0][k], iu[1][k]
            w = weights[k]
            dij = pair_dists[k]
            if dij > 1e-10:
                dir_ij = (centers[i] - centers[j]) / dij
                grad[i] += w * dir_ij
                grad[j] -= w * dir_ij

        # Soft-min for boundary distances
        exp_neg_b = np.exp(-beta * boundary)
        sum_exp_b = np.sum(exp_neg_b)
        weights_b = exp_neg_b / sum_exp_b

        # Gradient of soft-min of boundary:
        # g = soft_min(boundary) = -1/beta * log(sum(exp(-beta * b_i)))
        # dg/d(c_i) = w_i * (-c_i / |c_i|)
        for i in range(N):
            r_i = np.sqrt(np.sum(centers[i]**2))
            if r_i > 1e-10:
                grad[i] += weights_b[i] * (-centers[i] / r_i)

        # Update with momentum
        velocity = momentum * velocity + lr * grad
        centers = centers + velocity

        # Project back into feasible region (clip to unit disk)
        norms = np.sqrt(np.sum(centers**2, axis=1))
        mask = norms > 1.0
        if np.any(mask):
            centers[mask] = centers[mask] / norms[mask][:, np.newaxis]

        # Compute actual radius
        current_r = _compute_radius(centers)
        if current_r > best_r:
            best_r = current_r
            best_centers = centers.copy()

        # Decay learning rate
        lr *= decay
        # Increase sharpness
        beta *= 1.0005

    # Final polish: try small random perturbations around best
    if time.time() < deadline - 5:
        for _ in range(100):
            if time.time() > deadline:
                break
            perturbed = best_centers + 0.01 * np.random.randn(N, 2)
            norms = np.sqrt(np.sum(perturbed**2, axis=1))
            mask = norms > 1.0
            if np.any(mask):
                perturbed[mask] = perturbed[mask] / norms[mask][:, np.newaxis]
            r = _compute_radius(perturbed)
            if r > best_r:
                best_r = r
                best_centers = perturbed.copy()

    return best_centers
# EVOLVE-BLOCK-END
