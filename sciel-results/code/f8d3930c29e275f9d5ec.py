"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer before 150s limit

    def get_radius(c):
        """Compute the exact maximum common radius for centers c (N,3)."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        return min(min_pair_dist / 2.0, min_wall_dist)

    # Precompute triangle indices once
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = len(tri_i)

    def smooth_objective_and_grad(c_flat, beta):
        """
        Vectorized soft-min surrogate for the packing objective.
        Returns (value, gradient) for minimization of -soft_min(g_k).
        """
        c = c_flat.reshape(N, 3)

        if np.any(np.abs(c) > 0.5 + 1e-9):
            return 1e6, np.zeros_like(c_flat)

        # Pairwise: compute all N×N distances, extract upper triangle
        diff_full = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d_full = np.sqrt(np.sum(diff_full ** 2, axis=2))
        d_pairs = d_full[tri_i, tri_j] / 2.0

        # Wall distances
        d_x_pos = 0.5 - c[:, 0]
        d_x_neg = c[:, 0] + 0.5
        d_y_pos = 0.5 - c[:, 1]
        d_y_neg = c[:, 1] + 0.5
        d_z_pos = 0.5 - c[:, 2]
        d_z_neg = c[:, 2] + 0.5

        wall_dists = np.concatenate([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg])
        g_all = np.concatenate([d_pairs, wall_dists])

        # Soft min value
        neg_beta_g = -beta * g_all
        log_sum_exp = logsumexp(neg_beta_g)
        val = -log_sum_exp / beta

        # Weights
        weights = np.exp(neg_beta_g - log_sum_exp)

        # Gradient: vectorized pairwise contribution
        grad_c = np.zeros_like(c)

        # Pairwise gradient: for each pair (i,j) with weight w_k,
        # grad_c[i] += w_k * 0.5 * (c_i - c_j) / d_ij
        # grad_c[j] -= w_k * 0.5 * (c_i - c_j) / d_ij
        diff_ij = c[tri_i] - c[tri_j]  # (n_pairs, 3)
        d_ij = d_full[tri_i, tri_j]    # (n_pairs,)
        safe_d = np.where(d_ij > 1e-14, d_ij, 1.0)
        unit_vecs = diff_ij / safe_d[:, np.newaxis]  # (n_pairs, 3)
        w_pair = weights[:n_pairs] * 0.5 / safe_d   # (n_pairs,)
        contrib = w_pair[:, np.newaxis] * unit_vecs  # (n_pairs, 3)

        # Accumulate using np.add.at for unbuffered scatter-add
        np.add.at(grad_c, tri_i, contrib)
        np.add.at(grad_c, tri_j, -contrib)

        # Wall gradients (vectorized)
        ws = n_pairs
        grad_c[:, 0] -= weights[ws:ws+N]
        grad_c[:, 0] += weights[ws+N:ws+2*N]
        grad_c[:, 1] -= weights[ws+2*N:ws+3*N]
        grad_c[:, 1] += weights[ws+3*N:ws+4*N]
        grad_c[:, 2] -= weights[ws+4*N:ws+5*N]
        grad_c[:, 2] += weights[ws+5*N:ws+6*N]

        return -val, -grad_c.flatten()

    def run_smooth_optimization(c_init, beta, max_iter):
        """Run L-BFGS-B on the smooth surrogate."""
        x0 = c_init.flatten()
        bounds = [(-0.499, 0.499)] * (N * 3)

        def fun(x):
            val, grad = smooth_objective_and_grad(x, beta)
            return val

        def jac(x):
            val, grad = smooth_objective_and_grad(x, beta)
            return grad

        try:
            result = minimize(
                fun, x0, jac=jac,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100,
                         'maxfun': max_iter * 10, 'maxcor': 50}
            )
            return result.x.reshape(N, 3)
        except Exception:
            return c_init.copy()

    def annealed_optimization(c_init):
        """
        Multi-stage annealing with vectorized gradient.
        """
        betas = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 100000, 200000]

        c_best = c_init.copy()
        r_best = get_radius(c_init)

        for beta in betas:
            if time.time() > deadline:
                break

            remaining = deadline - time.time()
            idx = betas.index(beta)
            n_remaining = len(betas) - idx
            # Earlier stages get slightly more time; later stages need fewer iterations
            stage_time = max(2.0, remaining / n_remaining * 0.85)
            # Scale iterations by time and beta (higher beta needs fewer iters)
            max_iter = int(min(400, max(50, stage_time * 25 / (1 + 0.1 * np.log2(beta)))))

            c_new = run_smooth_optimization(c_best, beta, max_iter)
            r_new = get_radius(c_new)
            if r_new > r_best:
                r_best = r_new
                c_best = c_new.copy()

        return c_best, r_best

    def local_refinement_exact(c_init, max_iter=1000):
        """
        Use Nelder-Mead or Powell on the exact objective to polish the solution.
        This handles the non-smoothness of the min function better in the final stages.
        """
        def exact_neg_radius(x):
            c = x.reshape(N, 3)
            # Penalize if outside bounds
            if np.any(np.abs(c) > 0.499):
                return 1e6
            return -get_radius(c)

        x0 = c_init.flatten()
        try:
            # Powell is often better for this kind of problem than Nelder-Mead
            result = minimize(
                exact_neg_radius, x0,
                method='Powell',
                options={'maxiter': max_iter, 'ftol': 1e-15, 'xtol': 1e-15, 'maxfev': max_iter * 20}
            )
            return result.x.reshape(N, 3)
        except Exception:
            return c_init.copy()

    def push_apart_polish(c_init, steps=1000):
        """
        Simple local search: repeatedly find the closest pair and push them apart.
        This is very effective for the final polishing of sphere packings.
        """
        c = c_init.copy()
        for _ in range(steps):
            if time.time() > deadline - 2:
                break
            
            # Find closest pair
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            min_idx = np.unravel_index(np.argmin(d), d.shape)
            i, j = min_idx[0], min_idx[1]
            
            # Find closest wall
            dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
            min_wall_idx = np.argmin(dist_to_walls)
            
            # Compare distances
            pair_dist = d[i, j]
            wall_dist = dist_to_walls[min_wall_idx]
            
            # If pair is closer than wall, push pair apart
            if pair_dist < wall_dist:
                vec = c[i] - c[j]
                norm = np.linalg.norm(vec)
                if norm > 1e-10:
                    vec = vec / norm
                    # Push apart by a small amount
                    step = 0.01
                    c[i] += vec * step
                    c[j] -= vec * step
            else:
                # Push sphere away from wall
                idx = min_wall_idx
                # Determine which wall is closest
                for k in range(3):
                    if c[idx, k] > 0:
                        # Closest to +0.5
                        if 0.5 - c[idx, k] == dist_to_walls[idx]:
                            c[idx, k] -= 0.01
                            break
                    else:
                        # Closest to -0.5
                        if c[idx, k] + 0.5 == dist_to_walls[idx]:
                            c[idx, k] += 0.01
                            break
            
            # Ensure points stay within bounds
            c = np.clip(c, -0.499, 0.499)
            
            # Check if we improved
            r_current = get_radius(c)
            # We don't track best here to keep it simple, just return final state
            # But we should only accept moves that don't decrease radius too much? 
            # Actually, push-apart usually increases radius if we push the bottleneck.
            
        return c

    def generate_initializations():
        """
        Generate diverse initial configurations.
        """
        inits = []

        # 1. Simple grid (4x4x4 = 64, take first 50)
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        inits.append(grid[:N].copy())

        # 2. BCC-inspired: body-centered cubic lattice
        n_cell = 4
        cell_size = 1.0 / n_cell
        bcc_points = []
        for iz in range(n_cell):
            for iy in range(n_cell):
                for ix in range(n_cell):
                    cx = -0.5 + cell_size * (ix + 0.5)
                    cy = -0.5 + cell_size * (iy + 0.5)
                    cz = -0.5 + cell_size * (iz + 0.5)
                    bcc_points.append([cx, cy, cz])
                    # Body center offset
                    bcc_points.append([cx + cell_size/2, cy + cell_size/2, cz + cell_size/2])
        bcc_arr = np.array(bcc_points)
        # Keep only points inside [-0.5, 0.5]
        mask = np.all(np.abs(bcc_arr) <= 0.5, axis=1)
        bcc_arr = bcc_arr[mask]
        # Take first N
        if len(bcc_arr) >= N:
            inits.append(bcc_arr[:N].copy())
        else:
            # Pad with random if not enough
            extra = np.random.uniform(-0.45, 0.45, size=(N - len(bcc_arr), 3))
            inits.append(np.vstack([bcc_arr, extra]))

        # 3. Random rotations of grid
        np.random.seed(42)
        for trial in range(3):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(theta), -np.sin(theta)],
                          [0, np.sin(theta), np.cos(theta)]])
            Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                          [0, 1, 0],
                          [-np.sin(phi), 0, np.cos(phi)]])
            R = Ry @ Rx
            base = np.random.uniform(-0.3, 0.3, size=(N, 3))
            rotated = base @ R.T
            max_abs = np.max(np.abs(rotated))
            if max_abs > 0:
                rotated = rotated * (0.45 / max_abs)
            inits.append(rotated)

        # 4. Layered initialization
        n_layers = 5
        layer_height = 0.8 / n_layers
        points_per_layer = N // n_layers
        remaining_pts = N - points_per_layer * n_layers
        c_layered = np.zeros((N, 3))
        idx = 0
        for layer in range(n_layers):
            n_in_layer = points_per_layer + (1 if layer < remaining_pts else 0)
            z_coord = -0.4 + layer_height * (layer + 0.5)
            m_layer = int(np.ceil(np.sqrt(n_in_layer)))
            spacing = 0.8 / m_layer
            for i in range(n_in_layer):
                ix = i % m_layer
                iy = i // m_layer
                x = -0.4 + spacing * (ix + 0.5)
                y = -0.4 + spacing * (iy + 0.5)
                c_layered[idx] = [x, y, z_coord]
                idx += 1
        inits.append(c_layered)

        # 5. Random uniform
        inits.append(np.random.uniform(-0.45, 0.45, size=(N, 3)))

        return inits

    # Main optimization loop
    inits = generate_initializations()

    best_c = None
    best_r = -1.0

    for c_init in inits:
        if time.time() > deadline - 10:
            break

        c_init = np.clip(c_init, -0.49, 0.49)
        
        # Stage 1: Smooth optimization
        c_opt, r_opt = annealed_optimization(c_init)
        
        # Stage 2: Exact local refinement (Powell)
        if time.time() < deadline - 5:
            c_refined = local_refinement_exact(c_opt, max_iter=200)
            r_refined = get_radius(c_refined)
            if r_refined > r_opt:
                c_opt = c_refined
                r_opt = r_refined

        # Stage 3: Push-apart polish
        if time.time() < deadline - 2:
            c_polished = push_apart_polish(c_opt, steps=500)
            r_polished = get_radius(c_polished)
            if r_polished > r_opt:
                c_opt = c_polished
                r_opt = r_polished

        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()

    # Final high-precision polish on the best found so far
    if best_c is not None and time.time() < deadline - 5:
        # Try another round of Powell on the best result
        c_final = local_refinement_exact(best_c, max_iter=300)
        r_final = get_radius(c_final)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
        
        # And push-apart
        c_final2 = push_apart_polish(best_c, steps=500)
        r_final2 = get_radius(c_final2)
        if r_final2 > best_r:
            best_r = r_final2
            best_c = c_final2.copy()

    # Fallback
    if best_c is None:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]

    best_c = np.clip(best_c, -0.4999, 0.4999)

    final_r = get_radius(best_c)
    if final_r <= 0:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]

    return best_c
# EVOLVE-BLOCK-END
