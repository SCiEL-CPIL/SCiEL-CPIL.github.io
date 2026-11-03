"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer before 150s limit

    def get_radius(c):
        """Compute the exact maximum common radius for centers c (N,3)."""
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)

        # Wall distances: for each point, min over 6 walls
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)

        return min(min_pair_dist / 2.0, min_wall_dist)

    def smooth_objective_and_grad(c_flat, beta):
        """
        Soft-min surrogate for the packing objective.
        Returns (value, gradient) for minimization.
        We minimize -soft_min(g_k) where g_k are the constraint values (half-pair-dists and wall-dists).
        """
        c = c_flat.reshape(N, 3)

        # Check bounds - if violated, return large penalty
        if np.any(np.abs(c) > 0.5 + 1e-9):
            return 1e6, np.zeros_like(c_flat)

        # Pairwise distances (upper triangle only)
        tri_idx = np.triu_indices(N, k=1)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        d_pairs = d[tri_idx] / 2.0  # half pairwise distances

        # Wall distances (6 per point)
        d_x_pos = 0.5 - c[:, 0]
        d_x_neg = c[:, 0] + 0.5
        d_y_pos = 0.5 - c[:, 1]
        d_y_neg = c[:, 1] + 0.5
        d_z_pos = 0.5 - c[:, 2]
        d_z_neg = c[:, 2] + 0.5

        wall_dists = np.concatenate([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg])

        # Combine all constraint values
        g_all = np.concatenate([d_pairs, wall_dists])

        # Soft min: f = -1/beta * logsumexp(-beta * g)
        log_sum_exp = logsumexp(-beta * g_all)
        val = -log_sum_exp / beta

        # Weights for gradient: p_j = exp(-beta * g_j - log_sum_exp)
        weights = np.exp(-beta * g_all - log_sum_exp)

        # Gradient accumulation
        grad_c = np.zeros_like(c)

        # Pairwise part
        n_pairs = len(d_pairs)
        for k in range(n_pairs):
            w = weights[k]
            if w < 1e-14:
                continue
            i, j = tri_idx[0][k], tri_idx[1][k]
            diff_ij = c[i] - c[j]
            d_ij = np.linalg.norm(diff_ij)
            if d_ij > 1e-14:
                # d(d_ij/2)/dc_i = 0.5 * (c_i - c_j) / d_ij
                grad_c[i] += w * 0.5 * diff_ij / d_ij
                grad_c[j] -= w * 0.5 * diff_ij / d_ij

        # Wall part
        w_start = n_pairs
        # d_x_pos: weight * (-1) on c[:,0]
        grad_c[:, 0] -= weights[w_start : w_start+N]
        # d_x_neg: weight * (+1) on c[:,0]
        grad_c[:, 0] += weights[w_start+N : w_start+2*N]
        # d_y_pos
        grad_c[:, 1] -= weights[w_start+2*N : w_start+3*N]
        # d_y_neg
        grad_c[:, 1] += weights[w_start+3*N : w_start+4*N]
        # d_z_pos
        grad_c[:, 2] -= weights[w_start+4*N : w_start+5*N]
        # d_z_neg
        grad_c[:, 2] += weights[w_start+5*N : w_start+6*N]

        # We minimize -soft_min, so return (-val, -grad_c)
        return -val, -grad_c.flatten()

    def run_smooth_optimization(c_init, beta, max_iter, time_budget):
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
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100,
                         'maxfun': max_iter * 10}
            )
            return result.x.reshape(N, 3)
        except Exception:
            return c_init.copy()

    def annealed_optimization(c_init, time_budget):
        """
        Multi-stage annealing: start with low beta (smooth), increase to high beta (sharp).
        This helps escape shallow local minima and converge to sharper minima.
        """
        # Beta schedule: geometric progression
        betas = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 100000]

        c_best = c_init.copy()
        r_best = get_radius(c_init)

        for beta in betas:
            if time.time() > deadline:
                break

            # Allocate time proportional to 1/beta (earlier stages get more time)
            # But ensure at least ~5s per stage if time permits
            remaining = deadline - time.time()
            n_remaining_stages = len(betas) - betas.index(beta)
            stage_time = max(3.0, remaining / n_remaining_stages * 0.8)

            max_iter = int(min(500, max(100, stage_time * 30)))

            c_new = run_smooth_optimization(c_best, beta, max_iter, stage_time)

            r_new = get_radius(c_new)
            if r_new > r_best:
                r_best = r_new
                c_best = c_new.copy()

        return c_best, r_best

    def spectral_initialization():
        """
        Generate diverse initial configurations using spectral methods.
        Use random positive-definite matrices and take top eigenvectors.
        """
        inits = []

        # 1. Grid-based initialization
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        inits.append(grid[:N])

        # 2. Spectral initialization: random SPD matrix, top N eigenvectors
        np.random.seed(42)
        for trial in range(4):
            # Create random SPD matrix
            A = np.random.randn(3, 3)
            M = A.T @ A + 3 * np.eye(3)  # SPD
            # Add some structure to encourage spread
            M += 0.5 * np.diag(np.random.uniform(0.1, 1.0, 3))

            # Get eigenvectors
            eigvals, eigvecs = np.linalg.eigh(M)

            # We need N=50 points, but only 3 eigenvectors from 3x3.
            # Instead, use a different approach: create a random matrix and
            # use its columns plus perturbations to generate N points.

            # Alternative spectral approach: use random Fourier-like basis
            # Generate points on a 3D grid with random rotations and translations
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            # Rotation matrix
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(theta), -np.sin(theta)],
                          [0, np.sin(theta), np.cos(theta)]])
            Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                          [0, 1, 0],
                          [-np.sin(phi), 0, np.cos(phi)]])
            R = Ry @ Rx

            # Generate base points in a smaller cube
            base = np.random.uniform(-0.3, 0.3, size=(N, 3))
            # Apply rotation
            rotated = base @ R.T
            # Scale to fit in [-0.45, 0.45]
            max_abs = np.max(np.abs(rotated))
            if max_abs > 0:
                rotated = rotated * (0.45 / max_abs)
            inits.append(rotated)

        # 3. Random uniform initializations
        for trial in range(3):
            inits.append(np.random.uniform(-0.45, 0.45, size=(N, 3)))

        # 4. Structured initialization: layered approach
        # Place points in layers to encourage good spacing
        n_layers = 5
        layer_height = 0.8 / n_layers
        points_per_layer = N // n_layers
        remaining = N - points_per_layer * n_layers
        c_layered = np.zeros((N, 3))
        idx = 0
        for layer in range(n_layers):
            n_in_layer = points_per_layer + (1 if layer < remaining else 0)
            z_coord = -0.4 + layer_height * (layer + 0.5)
            # Place points in a grid within this layer
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

        return inits

    # Main optimization loop
    inits = spectral_initialization()

    best_c = None
    best_r = -1.0

    for c_init in inits:
        if time.time() > deadline:
            break

        # Ensure initial points are within bounds
        c_init = np.clip(c_init, -0.49, 0.49)

        c_opt, r_opt = annealed_optimization(c_init, deadline)

        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()

    # Final high-precision polish on the best configuration
    if best_c is not None and time.time() < deadline - 10:
        # Use very high beta for fine convergence
        for beta in [200000, 500000, 1000000]:
            if time.time() > deadline - 5:
                break
            max_iter = int(min(300, max(50, (deadline - 5 - time.time()) * 20)))
            c_polished = run_smooth_optimization(best_c, beta, max_iter, deadline - 5)
            r_polished = get_radius(c_polished)
            if r_polished > best_r:
                best_r = r_polished
                best_c = c_polished.copy()

    # Final validation and clipping
    if best_c is None:
        # Fallback: return a grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]

    # Clip to ensure points are strictly inside the cube
    best_c = np.clip(best_c, -0.4999, 0.4999)

    # Verify the result is valid
    final_r = get_radius(best_c)
    if final_r <= 0:
        # Fallback to a safe grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]

    return best_c
# EVOLVE-BLOCK-END
