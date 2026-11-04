import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 145.0  # Safety margin before 150s

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
        
    def remaining():
        return DEADLINE_SECONDS - elapsed()

    # Precompute indices for pairwise distances
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = len(tri_i)

    def get_radius(c):
        """Compute the exact maximum common radius for centers c (N,3)."""
        c = np.asarray(c, dtype=np.float64)
        if c.shape != (N, 3) or not np.all(np.isfinite(c)):
            return 0.0
        if np.any(np.abs(c) > 0.5 + 1e-9):
            return 0.0
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        # Wall distances
        dist_to_walls = np.min(0.5 - np.abs(c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return float(min(min_pair_dist / 2.0, min_wall_dist))

    def get_active_gradients(c, r_target):
        """
        Compute gradients for the active constraints (those close to r_target).
        Returns a gradient vector for c (N, 3) that points in the direction
        that increases the minimum radius.
        """
        grad_c = np.zeros_like(c)
        
        # 1. Pairwise constraints
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Identify pairs close to 2*r_target
        # We use a tolerance to handle ties and numerical stability
        tol = 1e-4
        active_pairs = np.where(d < 2.0 * (r_target + tol))[0]
        # Map flat indices to (i, j)
        # We need to be careful with triu mapping. 
        # It's faster to just compute all and mask if N is small, but N=50 is 1225 pairs.
        # Let's use the precomputed indices.
        pair_dists = d[tri_i, tri_j]
        active_mask = pair_dists < 2.0 * (r_target + tol)
        
        if np.any(active_mask):
            idxs = np.where(active_mask)[0]
            i_idx = tri_i[idxs]
            j_idx = tri_j[idxs]
            d_idx = pair_dists[idxs]
            
            # Gradient of d_ij / 2 w.r.t c_i is (c_i - c_j) / (2 * d_ij)
            # We want to increase d, so we move c_i away from c_j.
            vecs = c[i_idx] - c[j_idx]
            # Avoid division by zero
            safe_d = np.where(d_idx > 1e-12, d_idx, 1.0)
            grads = vecs / (2.0 * safe_d[:, np.newaxis])
            
            np.add.at(grad_c, i_idx, grads)
            np.add.at(grad_c, j_idx, -grads)

        # 2. Wall constraints
        # dist = 0.5 - |c_k|
        # If c_k > 0, dist = 0.5 - c_k, grad = -1
        # If c_k < 0, dist = 0.5 + c_k, grad = +1
        # We want to increase dist, so we move away from the wall.
        
        # X-axis
        active_x_pos = c[:, 0] > 0
        active_x_neg = c[:, 0] <= 0
        d_x_pos = 0.5 - c[:, 0]
        d_x_neg = 0.5 + c[:, 0]
        
        # Check if close to r_target
        mask_x_pos = (active_x_pos) & (d_x_pos < r_target + tol)
        mask_x_neg = (active_x_neg) & (d_x_neg < r_target + tol)
        
        if np.any(mask_x_pos):
            grad_c[mask_x_pos, 0] -= 1.0
        if np.any(mask_x_neg):
            grad_c[mask_x_neg, 0] += 1.0
            
        # Y-axis
        active_y_pos = c[:, 1] > 0
        active_y_neg = c[:, 1] <= 0
        d_y_pos = 0.5 - c[:, 1]
        d_y_neg = 0.5 + c[:, 1]
        
        mask_y_pos = (active_y_pos) & (d_y_pos < r_target + tol)
        mask_y_neg = (active_y_neg) & (d_y_neg < r_target + tol)
        
        if np.any(mask_y_pos):
            grad_c[mask_y_pos, 1] -= 1.0
        if np.any(mask_y_neg):
            grad_c[mask_y_neg, 1] += 1.0
            
        # Z-axis
        active_z_pos = c[:, 2] > 0
        active_z_neg = c[:, 2] <= 0
        d_z_pos = 0.5 - c[:, 2]
        d_z_neg = 0.5 + c[:, 2]
        
        mask_z_pos = (active_z_pos) & (d_z_pos < r_target + tol)
        mask_z_neg = (active_z_neg) & (d_z_neg < r_target + tol)
        
        if np.any(mask_z_pos):
            grad_c[mask_z_pos, 2] -= 1.0
        if np.any(mask_z_neg):
            grad_c[mask_z_neg, 2] += 1.0
            
        return grad_c

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

    def run_polish(c_init, max_iter, beta=10000):
        """Run L-BFGS-B on the smooth surrogate for final polishing."""
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
                options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12,
                         'maxls': 100, 'maxfun': max_iter * 10, 'maxcor': 50}
            )
            return np.clip(result.x.reshape(N, 3), -0.499, 0.499)
        except Exception:
            return c_init.copy()

    def simulated_annealing(c_init, max_steps, initial_temp, cooling_rate):
        """
        Perform Simulated Annealing to optimize the sphere packing.
        """
        c = c_init.copy()
        r = get_radius(c)
        best_c = c.copy()
        best_r = r
        temp = initial_temp
        
        rng = np.random.RandomState(42)
        
        for step in range(max_steps):
            if remaining() < 10:
                break
            
            # Pick a random sphere
            i = rng.randint(N)
            # Generate a random move
            move = rng.normal(0, 0.01, size=3)
            c_new = c.copy()
            c_new[i] += move
            c_new[i] = np.clip(c_new[i], -0.499, 0.499)
            
            r_new = get_radius(c_new)
            
            # Acceptance criterion
            delta_r = r_new - r
            if delta_r > 0:
                c = c_new
                r = r_new
                if r > best_r:
                    best_r = r
                    best_c = c.copy()
            else:
                # Accept with probability exp(delta_r / temp)
                # delta_r is negative
                if np.exp(delta_r / temp) > rng.rand():
                    c = c_new
                    r = r_new
            
            # Cool down
            temp *= cooling_rate
            if temp < 1e-6:
                temp = 1e-6
                
        return best_c, best_r

    def make_fcc_init():
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
                        if np.all(np.abs(pos - 0.5) <= 0.48):
                            positions.append(pos)
        positions = np.array(positions, dtype=np.float64)
        if len(positions) >= N:
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
        return np.clip(positions[:N].copy(), -0.499, 0.499)

    def make_grid_init():
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

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

    # Define initializations
    initializations = [
        ("fcc", make_fcc_init),
        ("grid", make_grid_init),
        ("rand0", lambda: make_random_init(0)),
        ("rand1", lambda: make_random_init(123)),
        ("rand2", lambda: make_random_init(777)),
    ]

    # Strategy:
    # 1. Run SA for each initialization.
    # 2. Take the best result from SA.
    # 3. Run a high-beta polish on the best result.
    # 4. If time remains, try a few more SA runs with different seeds or perturbations.

    # Phase 1: SA on initializations
    for name, init_func in initializations:
        if remaining() < 20:
            break
        try:
            positions = init_func()
            positions = np.clip(positions, -0.499, 0.499)
            
            # SA parameters
            # We want to spend about 20-30 seconds per init if we have 5 inits?
            # Total time 145s. 5 inits. ~25s each.
            # max_steps depends on speed of get_radius.
            # get_radius is O(N^2) ~ 2500 ops. Very fast.
            # Let's try 5000 steps.
            max_steps = 5000
            initial_temp = 0.05
            cooling_rate = 0.9995
            
            pos_opt, r_opt = simulated_annealing(positions, max_steps, initial_temp, cooling_rate)
            update_best(pos_opt)
            
        except Exception:
            continue

    # Phase 2: Polish the best result found so far
    if best_positions is not None and remaining() > 10:
        # Use a high beta for the polish to make it very sharp
        polished = run_polish(best_positions, max_iter=200, beta=50000)
        update_best(polished)
        
        # If we still have time, do a second polish with even higher beta or another SA round
        if remaining() > 15:
            # Perturb slightly and SA again
            rng = np.random.RandomState(999)
            perturbed = best_positions + rng.normal(0, 0.005, size=(N, 3))
            perturbed = np.clip(perturbed, -0.499, 0.499)
            pos_opt2, r_opt2 = simulated_annealing(perturbed, 3000, 0.01, 0.999)
            update_best(pos_opt2)
            
            if remaining() > 5:
                polished2 = run_polish(best_positions, max_iter=100, beta=100000)
                update_best(polished2)

    # Final validity check and clip
    if best_positions is not None:
        best_positions = np.clip(best_positions, -0.499, 0.499)
    else:
        # Fallback to grid if something went wrong
        best_positions = make_grid_init()
        best_positions = np.clip(best_positions, -0.499, 0.499)

    return best_positions
