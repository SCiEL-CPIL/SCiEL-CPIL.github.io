import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def compute_score(points):
        n = len(points)
        if n < 2:
            return 0.0
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        radii = np.linalg.norm(points, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        return min(min_pair_dist / 2.0, min_boundary_dist)

    def gen_hex_lattice(spacing):
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing * 0.5:
                    pts.append((x, y))
        return np.array(pts)

    initial_spacing = 2.0 * np.sqrt(0.907 / N)
    spacing = initial_spacing
    while True:
        pts = gen_hex_lattice(spacing)
        if len(pts) >= N:
            break
        spacing *= 0.98

    radii = np.hypot(pts[:, 0], pts[:, 1])
    points = pts[np.argsort(radii, kind="stable")[:N]].copy()

    best_score = compute_score(points)
    best_points = points.copy()

    # Phase 1: Coarse force-directed optimization
    step_size = spacing * 0.1
    lr_decay = 0.9995
    max_iter = 3000

    for iteration in range(max_iter):
        if time.time() > deadline - 40:
            break

        n = len(points)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        dist[dist < 1e-10] = 1e-10
        np.fill_diagonal(dist, np.inf)

        close_mask = (dist < 5 * spacing) & (dist < np.inf)
        inv_dist_sq = np.zeros_like(dist)
        inv_dist_sq[close_mask] = 1.0 / (dist[close_mask] ** 2)

        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)

        radii = np.linalg.norm(points, axis=1)
        boundary_force = np.zeros_like(points)
        boundary_margin = 0.1
        over = radii > (1.0 - boundary_margin)
        if np.any(over):
            mag = (radii[over] - (1.0 - boundary_margin)) * 5.0
            boundary_force[over] = -mag[:, np.newaxis] * points[over] / radii[over, np.newaxis]

        forces += boundary_force

        force_norms = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_norms)
        if max_force > 1e-10:
            forces /= max_force

        points += step_size * forces

        radii = np.linalg.norm(points, axis=1)
        mask_out = radii > 1.0
        if np.any(mask_out):
            points[mask_out] /= radii[mask_out, np.newaxis]

        current_score = compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()

        step_size *= lr_decay
        if step_size < 1e-6:
            break

    # Phase 2: Smooth surrogate optimization using log-sum-exp
    # We maximize r where r = min(min_pair_dist/2, min_boundary_dist)
    # Use soft-min: r_soft = -beta * log-sum-exp(-values/beta)
    # This is smooth and approximates min as beta -> 0
    if time.time() < deadline - 20:
        points = best_points.copy()
        n = len(points)
        
        # Precompute pair indices for efficiency
        i_idx, j_idx = np.triu_indices(n, k=1)
        pairs = np.column_stack([i_idx, j_idx])
        num_pairs = len(pairs)
        
        def soft_min_score(pts, beta):
            """Compute smooth approximation of the packing score."""
            # Pair distances
            diff = pts[pairs[:, 0]] - pts[pairs[:, 1]]
            pair_dists = np.linalg.norm(diff, axis=1)
            pair_vals = pair_dists / 2.0
            
            # Boundary distances
            radii = np.linalg.norm(pts, axis=1)
            bound_vals = 1.0 - radii
            
            # Combine all values
            all_vals = np.concatenate([pair_vals, bound_vals])
            
            # Soft min: -beta * log(sum(exp(-val/beta)))
            # For numerical stability, subtract max
            shifted = -all_vals / beta
            shifted -= np.max(shifted)
            log_sum = np.log(np.sum(np.exp(shifted)))
            soft_min_val = -beta * log_sum + beta * np.max(-all_vals / beta)
            
            return soft_min_val, all_vals
        
        def soft_min_gradient(pts, beta):
            """Compute gradient of the soft-min score w.r.t. points."""
            diff = pts[pairs[:, 0]] - pts[pairs[:, 1]]
            pair_dists = np.linalg.norm(diff, axis=1)
            pair_dists = np.maximum(pair_dists, 1e-10)
            pair_vals = pair_dists / 2.0
            
            radii = np.linalg.norm(pts, axis=1)
            bound_vals = 1.0 - radii
            
            all_vals = np.concatenate([pair_vals, bound_vals])
            all_vals = np.maximum(all_vals, 1e-10)
            
            # Soft min gradient: d/dx soft_min = sum(exp(-v_i/beta) / sum(exp(-v_j/beta)) * (-1/beta) * dv_i/dx)
            # = -sum(w_i * dv_i/dx) where w_i = exp(-v_i/beta) / sum(exp(-v_j/beta))
            
            shifted = -all_vals / beta
            shifted -= np.max(shifted)
            exp_vals = np.exp(shifted)
            weights = exp_vals / np.sum(exp_vals)
            
            grad = np.zeros_like(pts)
            
            # Pair contributions
            pair_weights = weights[:num_pairs]
            for k in range(num_pairs):
                w = pair_weights[k]
                if w < 1e-10:
                    continue
                i, j = pairs[k]
                d = diff[k] / pair_dists[k]
                # d(pair_dist)/d(pts[i]) = d, d(pair_dist)/d(pts[j]) = -d
                # d(pair_val)/d(pts[i]) = d/2, d(pair_val)/d(pts[j]) = -d/2
                grad[i] += w * d
                grad[j] -= w * d
            
            # Boundary contributions
            bound_weights = weights[num_pairs:]
            for i in range(n):
                w = bound_weights[i]
                if w < 1e-10:
                    continue
                r = radii[i]
                if r > 1e-10:
                    # d(bound_val)/d(pts[i]) = -pts[i]/r
                    grad[i] -= w * pts[i] / r
            
            return grad, all_vals
        
        # Adaptive beta schedule: start large, decrease
        beta_start = 0.05
        beta_end = 0.005
        num_steps = 2000
        polish_deadline = deadline - 5
        
        step_size = 0.003
        for step in range(num_steps):
            if time.time() > polish_deadline:
                break
            
            # Linearly decrease beta
            frac = step / max(num_steps - 1, 1)
            beta = beta_start + (beta_end - beta_start) * frac
            
            score, all_vals = soft_min_score(points, beta)
            grad, _ = soft_min_gradient(points, beta)
            
            # Normalize gradient
            g_norm = np.linalg.norm(grad)
            if g_norm > 1e-10:
                grad /= g_norm
            
            # Update
            points += step_size * grad
            
            # Project inside unit circle
            radii = np.linalg.norm(points, axis=1)
            outside = radii > 0.999
            if np.any(outside):
                points[outside] *= (0.999 / radii[outside, np.newaxis])
            
            # Check actual score
            actual_score = compute_score(points)
            if actual_score > best_score:
                best_score = actual_score
                best_points = points.copy()
            
            # Decay step size
            step_size *= 0.9995
            if step_size < 1e-6:
                break

    # Phase 3: Final high-precision polish with tight tolerances
    if time.time() < deadline - 3:
        points = best_points.copy()
        n = len(points)
        
        # Get the closest pairs and boundary points
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        flat_dist = dist.flatten()
        k = min(50, n * (n - 1) // 2)
        smallest_indices = np.argpartition(flat_dist, k)[:k]
        
        pairs = []
        for idx in smallest_indices:
            i, j = np.unravel_index(idx, (n, n))
            if i < j:
                pairs.append((i, j))
        
        radii = np.linalg.norm(points, axis=1)
        boundary_pts = np.where(radii > 0.90)[0]
        
        # Use a very small step size and many iterations
        step_size = 0.0005
        polish_deadline = deadline - 1
        for step in range(5000):
            if time.time() > polish_deadline:
                break
            
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            min_pair_dist = np.min(dists)
            norms = np.linalg.norm(points, axis=1)
            min_boundary_dist = np.min(1.0 - norms)
            
            current_score = min(min_pair_dist / 2.0, min_boundary_dist)
            
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
            
            # Gradient from closest pairs
            flat_dists = dists.flatten()
            k2 = min(30, N * (N - 1) // 2)
            part_idx = np.argpartition(flat_dists, k2)[:k2]
            
            pair_grad = np.zeros_like(points)
            for idx in part_idx:
                i = int(idx // N)
                j = int(idx % N)
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-8:
                    continue
                vec = (points[i] - points[j]) / d
                pair_grad[i] += vec
                pair_grad[j] -= vec
            
            # Boundary gradient
            boundary_grad = np.zeros_like(points)
            close_to_boundary = norms > (1.0 - current_score - 0.001)
            for i in np.where(close_to_boundary)[0]:
                if norms[i] > 1e-8:
                    boundary_grad[i] = -points[i] / norms[i]
            
            pair_score = min_pair_dist / 2.0
            bound_score = min_boundary_dist
            
            if pair_score <= bound_score:
                w_pair = 1.0
                w_bound = 0.1
            else:
                w_pair = 0.1
                w_bound = 1.0
            
            grad = w_pair * pair_grad + w_bound * boundary_grad
            
            g_norm = np.linalg.norm(grad)
            if g_norm > 1e-8:
                grad = grad / g_norm
            
            points += step_size * grad
            
            norms_new = np.linalg.norm(points, axis=1)
            outside = norms_new > 0.9995
            if np.any(outside):
                points[outside] *= (0.9995 / norms_new[outside, np.newaxis])
            
            step_size *= 0.9998
            if step_size < 1e-7:
                break

    return best_points
