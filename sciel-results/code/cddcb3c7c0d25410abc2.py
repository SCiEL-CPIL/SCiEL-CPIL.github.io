"""
Fifty equal circles in a circle - physics-based relaxation with Newton refinement.
Uses inverse-power repulsion + soft wall potential, then Newton-Raphson on the
contact graph to polish the solution to high precision.
"""
import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    best_score = 0.0
    best_centers = None

    def _score_raw(centers):
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        norms = np.sqrt(np.sum(centers**2, axis=1))
        if np.any(norms >= 1.0):
            return 0.0
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        min_edge_dist = np.min(1.0 - norms)
        r = min(min_pair_dist / 2.0, min_edge_dist)
        return max(r, 0.0)

    def _physics_energy_and_grad(centers, alpha=6.0, wall_strength=100.0, r_target=0.13):
        """
        Energy: sum over pairs of (r_target/d)^alpha for d < 2*r_target
              + wall potential for points near boundary
        Returns (energy, gradient).
        """
        centers = centers.reshape(N, 2)
        grad = np.zeros_like(centers)
        energy = 0.0

        # Pair repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dvec = centers[i] - centers[j]
                d = np.sqrt(np.sum(dvec**2))
                if d < 1e-12:
                    d = 1e-12
                    dvec = np.array([1e-12, 0.0])
                if d < 2.0 * r_target * 1.5:
                    ratio = r_target / d
                    energy += ratio ** alpha
                    # d/d c_i of (r_target/d)^alpha = -alpha * r_target^alpha * d^(-alpha-1) * dvec
                    coeff = alpha * (r_target ** alpha) * (d ** (-(alpha + 1)))
                    grad[i] += coeff * dvec
                    grad[j] -= coeff * dvec

        # Wall potential: push points inward if they're too close to boundary
        for i in range(N):
            norm = np.sqrt(np.sum(centers[i]**2))
            if norm < 1e-12:
                continue
            margin = 1.0 - norm
            if margin < r_target:
                # Smooth wall: (r_target - margin)^2 type
                excess = r_target - margin
                if excess > 0:
                    energy += wall_strength * excess * excess
                    # Gradient: -2 * wall_strength * excess * (c_i / |c_i|)
                    grad[i] -= 2.0 * wall_strength * excess * (centers[i] / norm)

        return energy, grad

    def _physics_relax(centers, alpha=6.0, wall_strength=100.0, r_target=0.13,
                       max_iter=5000, lr_init=0.01, lr_decay=0.999, min_lr=1e-6):
        """Gradient descent with adaptive learning rate."""
        centers = centers.copy().reshape(N, 2)
        lr = lr_init
        prev_energy = np.inf

        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            energy, grad = _physics_energy_and_grad(centers, alpha, wall_strength, r_target)

            # Adaptive step: reduce if energy increased
            if energy > prev_energy + 1e-10:
                lr *= 0.5
            else:
                lr = min(lr * 1.02, lr_init)

            centers -= lr * grad
            prev_energy = energy

            # Ensure inside disk
            norms = np.sqrt(np.sum(centers**2, axis=1))
            mask = norms > 0.999
            if np.any(mask):
                centers[mask] *= 0.999 / norms[mask, np.newaxis]

            if lr < min_lr and it > 100:
                break

        return centers

    def _newton_refine(centers, max_iter=200, tol=1e-14):
        """
        Newton-Raphson refinement for the packing problem.
        At optimum, the active constraints (pairs at distance 2r, and points on boundary
        at distance 1-r) form a rigid framework. We solve for the configuration where
        all active constraints are exactly satisfied and the gradient of the objective
        is zero (KKT conditions).

        Simplified: iterate by finding the minimum constraint, then adjusting.
        """
        centers = centers.copy().reshape(N, 2)

        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break

            norms = np.sqrt(np.sum(centers**2, axis=1))
            edge_constraints = 1.0 - norms
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            pair_constraints = dists / 2.0

            # Find current radius
            r = min(np.min(pair_constraints), np.min(edge_constraints))

            # Find active constraints (within tolerance of the minimum)
            tol_active = 1e-8
            active_pairs = []
            active_edges = []

            for i in range(N):
                if edge_constraints[i] < r + tol_active:
                    active_edges.append(i)
            for i in range(N):
                for j in range(i + 1, N):
                    if pair_constraints[i, j] < r + tol_active:
                        active_pairs.append((i, j))

            if len(active_pairs) + len(active_edges) < 3:
                # Not enough constraints to determine improvement
                break

            # Build the linear system: for each active constraint, the gradient
            # of the constraint should be orthogonal to the displacement.
            # Actually, we want to maximize r. Use a different approach:
            # Perturb along the direction that increases the minimum constraint.

            # Compute gradient of the soft-min objective
            beta = 2000.0
            all_constraints = np.concatenate([edge_constraints, pair_constraints[
                np.triu_indices(N, 1)]])
            neg_beta_x = -beta * all_constraints
            max_val = np.max(neg_beta_x)
            exp_vals = np.exp(neg_beta_x - max_val)
            sum_exp = np.sum(exp_vals)
            weights = exp_vals / sum_exp

            # Gradient of -soft_min = sum of weights * gradient of constraint
            grad = np.zeros_like(centers)
            edge_w = weights[:N]
            for i in range(N):
                if norms[i] > 1e-12:
                    grad[i] += edge_w[i] * (-centers[i] / norms[i])

            iu = np.triu_indices(N, 1)
            pair_w = weights[N:]
            for k in range(len(pair_w)):
                i, j = iu[0][k], iu[1][k]
                if dists[i, j] > 1e-12:
                    dir_vec = (centers[i] - centers[j]) / dists[i, j]
                    grad[i] += pair_w[k] * dir_vec
                    grad[j] -= pair_w[k] * dir_vec

            # Take a step in the gradient direction (increases soft-min)
            # Normalize and take small step
            grad_norm = np.sqrt(np.sum(grad**2))
            if grad_norm < 1e-15:
                break

            # Line search
            step = 0.01
            for _ in range(20):
                new_centers = centers + step * grad / grad_norm
                new_norms = np.sqrt(np.sum(new_centers**2, axis=1))
                if np.any(new_norms >= 1.0):
                    step *= 0.5
                    continue
                new_diff = new_centers[:, np.newaxis, :] - new_centers[np.newaxis, :, :]
                new_dists = np.sqrt(np.sum(new_diff**2, axis=2))
                np.fill_diagonal(new_dists, np.inf)
                new_r = min(np.min(new_dists / 2.0), np.min(1.0 - new_norms))
                if new_r > r:
                    centers = new_centers
                    r = new_r
                    break
                else:
                    step *= 0.5

            if step < 1e-12:
                break

        return centers

    def _make_ring_config(ring_radii, ring_counts, angles_offset=0.0):
        centers = []
        if ring_counts[0] > 0:
            centers.append([0.0, 0.0])
        for ring_idx in range(1, len(ring_radii)):
            r = ring_radii[ring_idx]
            count = ring_counts[ring_idx]
            for k in range(count):
                theta = 2 * np.pi * k / count + angles_offset * ring_idx
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                centers.append([x, y])
        return np.array(centers[:N])

    def _hexagonal_config(spacing=0.26):
        points = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.sqrt(x*x + y*y)
                if r <= 1.0 - 0.12 - 1e-9:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            return None
        norms = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(norms, kind='stable')[:N]
        return points[idx]

    # Generate diverse initial configurations
    initial_configs = []

    # Ring configurations
    ring_params = [
        ([0.0, 0.25, 0.5, 0.75, 0.95], [1, 6, 12, 18, 13], 0.0),
        ([0.0, 0.22, 0.48, 0.72, 0.94], [1, 6, 12, 18, 13], 0.0),
        ([0.0, 0.28, 0.52, 0.74, 0.93], [1, 6, 12, 18, 13], 0.0),
        ([0.0, 0.25, 0.5, 0.75, 0.95], [1, 6, 12, 18, 13], 0.2),
        ([0.0, 0.25, 0.5, 0.75, 0.95], [1, 6, 12, 18, 13], 0.5),
        ([0.0, 0.25, 0.5, 0.75, 0.95], [1, 6, 12, 18, 13], 0.3),
        ([0.0, 0.25, 0.5, 0.75, 0.95], [1, 6, 12, 18, 13], 0.1),
        ([0.0, 0.25, 0.5, 0.75, 0.95], [1, 6, 12, 18, 13], 0.7),
    ]
    for radii, counts, offset in ring_params:
        initial_configs.append(_make_ring_config(radii, counts, offset))

    # Hexagonal
    hex_cfg = _hexagonal_config(0.26)
    if hex_cfg is not None:
        initial_configs.append(hex_cfg)
    hex_cfg2 = _hexagonal_config(0.27)
    if hex_cfg2 is not None:
        initial_configs.append(hex_cfg2)

    # Random configurations
    for seed in range(10):
        np.random.seed(1000 + seed * 7919)
        cfg = np.random.uniform(-0.9, 0.9, size=(N, 2))
        norms = np.sqrt(np.sum(cfg**2, axis=1))
        mask = norms > 0.9
        if np.any(mask):
            cfg[mask] *= 0.9 / norms[mask, np.newaxis]
        initial_configs.append(cfg)

    # Process each initial config
    for idx, x0 in enumerate(initial_configs):
        if time.time() > deadline - 15.0:
            break

        # Physics relaxation with different parameters
        for alpha, wall_s, r_t, lr_i in [(6.0, 100.0, 0.13, 0.01),
                                          (8.0, 100.0, 0.13, 0.005),
                                          (4.0, 50.0, 0.13, 0.02),
                                          (6.0, 200.0, 0.128, 0.01)]:
            if time.time() > deadline - 10.0:
                break

            try:
                relaxed = _physics_relax(x0, alpha, wall_s, r_t, lr_init=lr_i, max_iter=3000)
                score = _score_raw(relaxed)
                if score > best_score:
                    best_score = score
                    best_centers = relaxed.copy()
            except Exception:
                continue

        # Newton refinement on the best from physics
        if time.time() > deadline - 8.0:
            break
        try:
            refined = _newton_refine(relaxed, max_iter=100)
            score = _score_raw(refined)
            if score > best_score:
                best_score = score
                best_centers = refined.copy()
        except Exception:
            pass

    # Final high-precision Newton polish
    if best_centers is not None and time.time() < deadline - 5.0:
        for _ in range(3):
            if time.time() > deadline - 3.0:
                break
            try:
                polished = _newton_refine(best_centers, max_iter=300, tol=1e-15)
                score = _score_raw(polished)
                if score > best_score:
                    best_score = score
                    best_centers = polished.copy()
            except Exception:
                break

    # Final validation
    if best_centers is not None:
        final_score = _score_raw(best_centers)
        if final_score > 0.0 and np.all(np.isfinite(best_centers)):
            return best_centers

    # Fallback
    return _make_ring_config([0.0, 0.25, 0.5, 0.75, 0.95], [1, 6, 12, 18, 13])
