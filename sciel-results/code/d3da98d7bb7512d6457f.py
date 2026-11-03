"""Combined sphere packing: smooth surrogate optimization with diverse initializations."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 135.0  # Safety margin before 150s kill


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
                options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12,
                         'maxls': 100, 'maxfun': max_iter * 10, 'maxcor': 50}
            )
            return np.clip(result.x.reshape(N, 3), -0.499, 0.499)
        except Exception:
            return c_init.copy()

    def run_direct_optimization(c_init, max_iter):
        """Run L-BFGS-B on the exact non-smooth objective."""
        x0 = c_init.flatten()
        bounds = [(-0.499, 0.499)] * (N * 3)

        def fun(x):
            return -get_radius(x.reshape(N, 3))

        try:
            result = minimize(
                fun, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12}
            )
            return np.clip(result.x.reshape(N, 3), -0.499, 0.499)
        except Exception:
            return c_init.copy()

    def active_set_refinement(c_init, max_iter=100):
        """
        Active-set refinement: identify constraints active and do final polish.
        """
        c = c_init.copy()
        r = get_radius(c)

        # Identify active constraints
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)

        dist_to_walls = np.min(0.5 - np.abs(c), axis=1)
        min_wall_dist = np.min(dist_to_walls)

        # If the bottleneck is pair distance, try to increase it
        if min_pair_dist / 2.0 < min_wall_dist:
            # Find closest pairs
            pair_dists = d[tri_i, tri_j]
            close_pairs = np.where(pair_dists < min_pair_dist * 1.02)[0]
            if len(close_pairs) > 0:
                # Perturb points involved in close pairs
                involved = np.unique(np.concatenate([tri_i[close_pairs], tri_j[close_pairs]]))
                rng = np.random.RandomState(42)
                perturb = rng.normal(0, 0.003, size=(len(involved), 3))
                c[involved] += perturb
                c = np.clip(c, -0.499, 0.499)
                r_new = get_radius(c)
                if r_new > r:
                    c, r = c, r_new

        # Final direct optimization with tighter tolerance
        c_polished = run_direct_optimization(c, max_iter)
        r_polished = get_radius(c_polished)
        if r_polished > r:
            c, r = c_polished, r_polished

        return c, r

    def annealed_optimization(c_init):
        """
        Multi-stage annealing with aggressive beta schedule and time-proportional
        iteration allocation.
        """
        betas = [10, 20, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 100000, 200000, 400000]

        c_best = c_init.copy()
        r_best = get_radius(c_init)

        for idx, beta in enumerate(betas):
            if remaining() < 3.0:
                break

            n_remaining = len(betas) - idx
            stage_time = max(1.0, remaining() / n_remaining * 0.85)
            # Higher beta needs fewer iterations
            max_iter = int(min(200, max(20, stage_time * 15 / (1 + 0.1 * np.log2(beta)))))

            c_new = run_smooth_optimization(c_best, beta, max_iter)
            r_new = get_radius(c_new)
            if r_new > r_best:
                r_best = r_new
                c_best = c_new.copy()

        # Active-set refinement
        if remaining() > 5.0:
            c_refined, r_refined = active_set_refinement(c_best)
            if r_refined > r_best:
                c_best = c_refined.copy()
                r_best = r_refined

        # Final direct polish
        if remaining() > 3.0:
            c_final = run_direct_optimization(c_best, 50)
            r_final = get_radius(c_final)
            if r_final > r_best:
                c_best = c_final.copy()
                r_best = r_final

        return c_best, r_best

    def fcc_lattice_init(n=50, seed=0):
        """Generate FCC-like lattice initialization with proper selection."""
        rng = np.random.default_rng(seed)
        # FCC structure: face-centered cubic
        # Generate a dense grid and select best subset
        m = 6  # 6x6x6 = 216 points
        pts = []
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
                    for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                        x = (i + dx) / m - 0.5
                        y = (j + dy) / m - 0.5
                        z = (k + dz) / m - 0.5
                        if -0.49 <= x <= 0.49 and -0.49 <= y <= 0.49 and -0.49 <= z <= 0.49:
                            pts.append([x, y, z])
        pts = np.array(pts)
        
        # Score each point by its minimum distance to others
        if len(pts) > n:
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            min_dists = np.min(d, axis=1)
            
            # Select top n points by min distance
            idx = np.argsort(-min_dists)[:n]
            pts = pts[idx]
        
        # Shuffle to avoid bias
        idx = rng.permutation(len(pts))
        pts = pts[idx[:n]]
        
        return pts

    def bcc_lattice_init(n=50, seed=0):
        """Generate BCC-like lattice initialization."""
        rng = np.random.default_rng(seed)
        m = 4  # 4x4x4 cells = 64 BCC points
        pts = []
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    # Corner
                    pts.append([i / m - 0.5, j / m - 0.5, k / m - 0.5])
                    # Body center
                    pts.append([(i + 0.5) / m - 0.5, (j + 0.5) / m - 0.5, (k + 0.5) / m - 0.5])
        pts = np.array(pts)
        
        # Score each point by its minimum distance to others
        if len(pts) > n:
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            min_dists = np.min(d, axis=1)
            
            # Select top n points by min distance
            idx = np.argsort(-min_dists)[:n]
            pts = pts[idx]
        
        # Shuffle to avoid bias
        idx = rng.permutation(len(pts))
        pts = pts[idx[:n]]
        
        return pts

    def random_init(n=50, seed=0):
        """Random initialization with some structure."""
        rng = np.random.default_rng(seed)
        x = rng.uniform(-0.45, 0.45, size=(n, 3))
        return x

    def grid_init(n=50):
        """Simple grid initialization."""
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:n]

    def rotated_init(base, angle_rad, axis=0):
        """Rotate a configuration around an axis."""
        c = base.copy()
        if axis == 0:
            # Rotate around x-axis
            y, z = c[:, 1], c[:, 2]
            c[:, 1] = y * np.cos(angle_rad) - z * np.sin(angle_rad)
            c[:, 2] = y * np.sin(angle_rad) + z * np.cos(angle_rad)
        elif axis == 1:
            # Rotate around y-axis
            x, z = c[:, 0], c[:, 2]
            c[:, 0] = x * np.cos(angle_rad) - z * np.sin(angle_rad)
            c[:, 2] = x * np.sin(angle_rad) + z * np.cos(angle_rad)
        else:
            # Rotate around z-axis
            x, y = c[:, 0], c[:, 1]
            c[:, 0] = x * np.cos(angle_rad) - y * np.sin(angle_rad)
            c[:, 1] = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        
        return np.clip(c, -0.499, 0.499)

    # Generate diverse initializations
    inits = []
    
    # FCC lattice
    inits.append(fcc_lattice_init(N, seed=0))
    inits.append(fcc_lattice_init(N, seed=1))
    
    # BCC lattice
    inits.append(bcc_lattice_init(N, seed=0))
    inits.append(bcc_lattice_init(N, seed=1))
    
    # Random inits
    for seed in range(4):
        inits.append(random_init(N, seed=seed))
    
    # Grid
    inits.append(grid_init(N))
    
    # Rotated versions of best initializations
    base_inits = inits[:4]  # FCC and BCC
    for base in base_inits:
        for angle in [np.pi/6, np.pi/4, np.pi/3]:
            for axis in range(3):
                inits.append(rotated_init(base, angle, axis))
    
    # Run optimization on each initialization
    best_x = None
    best_score = 0.0
    
    for i, x0 in enumerate(inits):
        if remaining() < 10.0:
            break
        
        # Time allocation: give more time to earlier, more structured inits
        time_per_init = max(5.0, remaining() / max(1, len(inits) - i) * 0.8)
        
        x_optimized, s_optimized = annealed_optimization(x0)
        s = get_radius(x_optimized)
        
        if s > best_score:
            best_score = s
            best_x = x_optimized.copy()
    
    # Final polish on the best result
    if best_x is not None and remaining() > 3.0:
        x_final, s_final = active_set_refinement(best_x, max_iter=100)
        if s_final > best_score:
            best_score = s_final
            best_x = x_final.copy()
    
    if best_x is None:
        # Fallback to simple grid
        best_x = grid_init(N)
    
    # Ensure valid output
    best_x = np.clip(best_x, -0.5, 0.5)
    return best_x
