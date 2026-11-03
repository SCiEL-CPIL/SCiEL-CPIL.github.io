import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 145.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE_SECONDS - elapsed()

    # Precompute triangle indices once
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = len(tri_i)

    def get_radius(c):
        """Compute the exact maximum common radius for centers c (N,3)."""
        c = np.asarray(c, dtype=np.float64)
        if c.shape != (N, 3) or not np.all(np.isfinite(c)):
            return 0.0
        if np.any(np.abs(c) > 0.5 + 1e-9):
            return 0.0
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        dist_to_walls = np.min(0.5 - np.abs(c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        return float(min(min_pair_dist / 2.0, min_wall_dist))

    def smooth_objective_and_grad(c_flat, beta):
        """
        Vectorized soft-min surrogate for the packing objective.
        Returns (value, gradient) for minimization of -soft_min(g_k).
        """
        c = c_flat.reshape(N, 3)

        if np.any(np.abs(c) > 0.5 + 1e-9):
            return 1e6, np.zeros_like(c_flat)

        # Pairwise distances
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
        lse = logsumexp(neg_beta_g)
        val = -lse / beta

        # Weights
        weights = np.exp(neg_beta_g - lse)

        # Gradient: vectorized pairwise contribution
        grad_c = np.zeros_like(c)

        diff_ij = c[tri_i] - c[tri_j]
        d_ij = d_full[tri_i, tri_j]
        safe_d = np.where(d_ij > 1e-14, d_ij, 1.0)
        unit_vecs = diff_ij / safe_d[:, np.newaxis]
        w_pair = weights[:n_pairs] * 0.5 / safe_d
        contrib = w_pair[:, np.newaxis] * unit_vecs

        np.add.at(grad_c, tri_i, contrib)
        np.add.at(grad_c, tri_j, -contrib)

        # Wall gradients
        ws = n_pairs
        grad_c[:, 0] -= weights[ws:ws + N]
        grad_c[:, 0] += weights[ws + N:ws + 2 * N]
        grad_c[:, 1] -= weights[ws + 2 * N:ws + 3 * N]
        grad_c[:, 1] += weights[ws + 3 * N:ws + 4 * N]
        grad_c[:, 2] -= weights[ws + 4 * N:ws + 5 * N]
        grad_c[:, 2] += weights[ws + 5 * N:ws + 6 * N]

        return -val, -grad_c.flatten()

    def run_smooth_optimization(c_init, beta, max_iter, tol=1e-12):
        """Run L-BFGS-B on the smooth surrogate."""
        x0 = c_init.flatten()
        bounds = [(-0.4999, 0.4999)] * (N * 3)

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
                options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': tol,
                         'maxls': 100, 'maxfun': max_iter * 10, 'maxcor': 50}
            )
            return np.clip(result.x.reshape(N, 3), -0.4999, 0.4999)
        except Exception:
            return c_init.copy()

    def make_hex_layer_init():
        """
        Initialize with 5 layers of 10 spheres, arranged in a hexagonal pattern.
        This is a strong heuristic for dense packing in a cube.
        """
        c = np.zeros((N, 3))
        n_layers = 5
        z_spacing = 0.8 / (n_layers - 1)
        
        # Hexagonal grid for 10 points
        # Approximate positions for 10 points in a hex cluster
        # Center + 6 around + 3 more? 
        # Let's use a simpler grid that is then optimized.
        # 5 layers, each 10 points.
        # Points in layer: 2x5 grid? Or hex?
        # Let's use a 2x5 grid per layer, shifted in x/y for alternating layers.
        
        idx = 0
        for l in range(n_layers):
            z = -0.4 + l * z_spacing
            # 10 points in a 2x5 arrangement
            for iy in range(2):
                for ix in range(5):
                    x = -0.4 + (ix + 0.5) * (0.8 / 5)
                    y = -0.4 + (iy + 0.5) * (0.8 / 2)
                    # Shift alternating layers to interlock
                    if l % 2 == 1:
                        x += (0.8 / 5) / 2
                        y += (0.8 / 2) / 2
                    c[idx] = [x, y, z]
                    idx += 1
        return np.clip(c, -0.4999, 0.4999)

    def make_fcc_init():
        """FCC lattice init."""
        a = 0.25
        positions = []
        basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
        n_cells = 4
        for iz in range(n_cells):
            for iy in range(n_cells):
                for ix in range(n_cells):
                    origin = np.array([ix, iy, iz]) * a
                    for b in basis:
                        pos = origin + b * a
                        if np.all(np.abs(pos) <= 0.48):
                            positions.append(pos)
        positions = np.array(positions)
        if len(positions) >= N:
            # Select best 50
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            idx = np.argsort(-min_dists)[:N]
            positions = positions[idx]
        else:
            rng = np.random.RandomState(123)
            extra = rng.uniform(-0.45, 0.45, size=(N - len(positions), 3))
            positions = np.vstack([positions, extra])
        return np.clip(positions[:N].copy(), -0.4999, 0.4999)

    def make_random_init(seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(-0.45, 0.45, size=(N, 3))

    best_positions = None
    best_radius = -1.0

    def update_best(pos):
        nonlocal best_positions, best_radius
        r = get_radius(pos)
        if r > best_radius:
            best_radius = r
            best_positions = pos.copy()
        return r

    # List of initializations
    initializations = [
        ("hex_layer", make_hex_layer_init),
        ("fcc", make_fcc_init),
        ("rand1", lambda: make_random_init(1)),
        ("rand2", lambda: make_random_init(42)),
        ("rand3", lambda: make_random_init(123)),
    ]

    # Optimization pipeline
    def optimize_run(c_init):
        """
        Run a full optimization sequence: 
        1. Coarse optimization with low beta
        2. Fine optimization with high beta
        3. Perturbation and re-optimization if stalled
        """
        c = c_init.copy()
        
        # Stage 1: Coarse
        if remaining() < 5:
            return c
        
        c = run_smooth_optimization(c, beta=50, max_iter=100)
        
        # Stage 2: Medium
        if remaining() < 5:
            return c
            
        c = run_smooth_optimization(c, beta=200, max_iter=100)
        
        # Stage 3: Fine
        if remaining() < 5:
            return c
            
        c = run_smooth_optimization(c, beta=1000, max_iter=200)
        
        # Stage 4: Very Fine
        if remaining() < 5:
            return c
            
        c = run_smooth_optimization(c, beta=5000, max_iter=200)
        
        # Stage 5: Ultra Fine
        if remaining() < 5:
            return c
            
        c = run_smooth_optimization(c, beta=20000, max_iter=200)
        
        return c

    # Main loop
    for name, init_func in initializations:
        if remaining() < 20:
            break
        
        try:
            c_init = init_func()
            c_opt = optimize_run(c_init)
            update_best(c_opt)
        except Exception:
            continue

    # If we have time, do more perturbations on the best solution
    if best_positions is not None and remaining() > 30:
        n_perturb = 5
        for k in range(n_perturb):
            if remaining() < 15:
                break
            rng = np.random.RandomState(1000 + k)
            # Small perturbation
            noise_scale = 0.005
            perturbed = best_positions + rng.normal(0, noise_scale, size=best_positions.shape)
            perturbed = np.clip(perturbed, -0.4999, 0.4999)
            
            # Optimize the perturbed version
            # Use a faster schedule for perturbations
            c_opt = run_smooth_optimization(perturbed, beta=1000, max_iter=100)
            c_opt = run_smooth_optimization(c_opt, beta=5000, max_iter=100)
            
            update_best(c_opt)

    # Final polish if time remains
    if best_positions is not None and remaining() > 10:
        c_final = run_smooth_optimization(best_positions, beta=50000, max_iter=200)
        update_best(c_final)

    if best_positions is None:
        best_positions = make_hex_layer_init()

    return np.clip(best_positions, -0.4999, 0.4999)
