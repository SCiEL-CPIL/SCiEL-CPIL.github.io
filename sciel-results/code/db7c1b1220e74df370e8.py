"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # 8s buffer before 150s limit

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

        # Pairwise gradient
        diff_ij = c[tri_i] - c[tri_j]
        d_ij = d_full[tri_i, tri_j]
        safe_d = np.where(d_ij > 1e-14, d_ij, 1.0)
        unit_vecs = diff_ij / safe_d[:, np.newaxis]
        w_pair = weights[:n_pairs] * 0.5 / safe_d
        contrib = w_pair[:, np.newaxis] * unit_vecs

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
            stage_time = max(2.0, remaining / n_remaining * 0.85)
            max_iter = int(min(400, max(50, stage_time * 25 / (1 + 0.1 * np.log2(beta)))))

            c_new = run_smooth_optimization(c_best, beta, max_iter)
            r_new = get_radius(c_new)
            if r_new > r_best:
                r_best = r_new
                c_best = c_new.copy()

        return c_best, r_best

    def lattice_graph_cut_init(resolution=10):
        """
        Use a graph-cut (max-flow) on a 3D voxel grid to select N non-overlapping
        lattice positions that maximize the minimum clearance to walls and each other.
        
        Strategy: For a given target radius r, build a conflict graph where two lattice
        points conflict if their distance < 2r. Use max-flow to find the largest independent
        set (or equivalently, minimum vertex cover complement). We binary search on r.
        
        Since exact independent set is NP-hard, we use a greedy+local-search approach
        guided by the lattice structure, then refine with the continuous optimizer.
        """
        # Create a 3D lattice of candidate positions
        # We'll use multiple resolutions and pick the best result
        best_c = None
        best_r = -np.inf
        
        for res in [8, 10, 12]:
            if time.time() > deadline - 20:
                break
                
            ticks = (np.arange(res) + 0.5) / res - 0.5
            lattice = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            n_cand = len(lattice)
            
            # Compute pairwise distances on lattice
            diff = lattice[:, np.newaxis, :] - lattice[np.newaxis, :, :]
            dist_mat = np.sqrt(np.sum(diff**2, axis=2))
            
            # Wall distances for each lattice point
            wall_dists = np.min(np.minimum(lattice + 0.5, 0.5 - lattice), axis=1)
            
            # Greedy selection: pick points in order of decreasing wall distance,
            # skipping any that are too close to already-selected points
            order = np.argsort(-wall_dists)
            
            # Try multiple target radii
            for r_target in [0.10, 0.11, 0.12, 0.13, 0.135]:
                min_pair_dist = 2 * r_target
                selected = []
                for idx in order:
                    if len(selected) >= N:
                        break
                    if len(selected) == 0:
                        selected.append(idx)
                    else:
                        sel_arr = np.array(selected)
                        d_to_sel = dist_mat[idx, sel_arr]
                        if np.all(d_to_sel >= min_pair_dist - 1e-10):
                            selected.append(idx)
                
                if len(selected) >= N:
                    c_sel = lattice[selected[:N]]
                    r_sel = get_radius(c_sel)
                    if r_sel > best_r:
                        best_r = r_sel
                        best_c = c_sel.copy()
            
            # Also try a local improvement: swap selected points with unselected
            # to increase minimum distance
            if best_c is not None and res == 10:
                c_improved = best_c.copy()
                r_improved = best_r
                improved = True
                max_swaps = 50
                for _ in range(max_swaps):
                    if time.time() > deadline - 10:
                        break
                    best_swap_gain = 0
                    best_swap = None
                    # Try swapping each selected point with each unselected point
                    sel_mask = np.zeros(n_cand, dtype=bool)
                    for idx in selected:
                        sel_mask[idx] = True
                    unselected_idx = np.where(~sel_mask)[0]
                    
                    for si, sidx in enumerate(selected):
                        for ui in unselected_idx[:50]:  # limit for speed
                            new_c = c_improved.copy()
                            new_c[si] = lattice[ui]
                            r_new = get_radius(new_c)
                            if r_new - r_improved > best_swap_gain:
                                best_swap_gain = r_new - r_improved
                                best_swap = (si, ui)
                    
                    if best_swap is not None:
                        si, ui = best_swap
                        c_improved[si] = lattice[ui]
                        r_improved = get_radius(c_improved)
                        if r_improved > best_r:
                            best_r = r_improved
                            best_c = c_improved.copy()
                    else:
                        break
        
        return best_c, best_r

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
                    bcc_points.append([cx + cell_size/2, cy + cell_size/2, cz + cell_size/2])
        bcc_arr = np.array(bcc_points)
        mask = np.all(np.abs(bcc_arr) <= 0.5, axis=1)
        bcc_arr = bcc_arr[mask]
        if len(bcc_arr) >= N:
            inits.append(bcc_arr[:N].copy())
        else:
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
        pt_idx = 0
        for layer in range(n_layers):
            z = -0.4 + layer * layer_height
            n_pts = points_per_layer + (1 if layer < remaining_pts else 0)
            n_x = int(np.ceil(np.sqrt(n_pts)))
            n_y = int(np.ceil(n_pts / n_x))
            spacing_x = 0.7 / (n_x + 0.5)
            spacing_y = 0.7 / (n_y + 0.5)
            for ix in range(n_x):
                for iy in range(n_y):
                    if pt_idx >= N:
                        break
                    x = -0.35 + ix * spacing_x
                    y = -0.35 + iy * spacing_y
                    if (ix % 2 == 1):
                        y += spacing_y / 2
                    c_layered[pt_idx] = [x, y, z]
                    pt_idx += 1
            if pt_idx >= N:
                break
        inits.append(c_layered)

        # 5. Uniform target distribution initialization (inspiration: OT-inspired)
        # Place points at a fine grid and subsample to get uniform coverage
        n_fine = 8
        fine_ticks = (np.arange(n_fine) + 0.5) / n_fine - 0.5
        fine_grid = np.array([(x, y, z) for z in fine_ticks for y in fine_ticks for x in fine_ticks])
        # Shuffle and take first N for pseudo-random uniform sampling
        np.random.seed(123)
        perm = np.random.permutation(len(fine_grid))
        inits.append(fine_grid[perm[:N]].copy())

        # 6. Lattice graph-cut inspired initialization
        c_lattice, r_lattice = lattice_graph_cut_init()
        if c_lattice is not None:
            inits.append(c_lattice)

        return inits

    # Generate multiple initial configurations
    initial_configs = generate_initializations()

    best_centers = None
    best_r = -np.inf

    # Try each initial configuration
    for idx, init in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break

        # First, run annealed smooth optimization
        c_opt, r_opt = annealed_optimization(init)

        if r_opt > best_r:
            best_r = r_opt
            best_centers = c_opt.copy()

        # Final high-precision polish with very high beta
        if time.time() < deadline - 5:
            beta_final = 500000
            c_polished = run_smooth_optimization(best_centers, beta_final, 200)
            r_polished = get_radius(c_polished)
            if r_polished > best_r:
                best_r = r_polished
                best_centers = c_polished.copy()

    # If no valid result, fall back to simple grid
    if best_centers is None:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()

    return best_centers
# EVOLVE-BLOCK-END
