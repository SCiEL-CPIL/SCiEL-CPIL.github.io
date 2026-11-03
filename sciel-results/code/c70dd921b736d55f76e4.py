"""
Optimized packing of 50 equal spheres in a unit cube.
Combines soft-min surrogate annealing with a repulsive-potential warm-up
and vectorized gradient computation for speed.
"""
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 50
DEADLINE_SECONDS = 145.0

# Precompute pair indices
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
N_PAIRS = len(PAIR_I)

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE_SECONDS - elapsed()

    def compute_min_radius(positions):
        """Compute the minimum radius given positions of shape (N, 3)."""
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if not np.all(np.isfinite(positions)):
            return 0.0
        if positions.shape[0] != N or positions.shape[1] != 3:
            return 0.0
        if np.any(np.abs(positions) >= 0.5):
            return 0.0

        dists = pdist(positions, 'euclidean')
        min_dist = np.min(dists) if len(dists) > 0 else 1.0
        min_r_pairs = min_dist / 2.0

        walls = 0.5 - np.abs(positions)
        min_wall = np.min(walls)

        return min(min_r_pairs, min_wall)

    def smooth_surrogate_and_grad(x_flat, beta):
        """
        Compute the smooth surrogate objective and its gradient.
        Objective: -1/beta * log(sum(exp(-beta * v_i)))  (soft-min)
        We minimize this, which is equivalent to maximizing the soft-min of v_i.
        """
        positions = x_flat.reshape(-1, 3)

        # Compute pairwise distances
        dists = pdist(positions, 'euclidean')
        pair_r = dists / 2.0

        # Compute wall distances (6 per point: +/-x, +/-y, +/-z)
        walls = np.empty((N, 6))
        walls[:, 0] = positions[:, 0] + 0.5
        walls[:, 1] = 0.5 - positions[:, 0]
        walls[:, 2] = positions[:, 1] + 0.5
        walls[:, 3] = 0.5 - positions[:, 1]
        walls[:, 4] = positions[:, 2] + 0.5
        walls[:, 5] = 0.5 - positions[:, 2]
        wall_r = walls.ravel()

        all_vals = np.concatenate([pair_r, wall_r])

        # Stable log-sum-exp
        neg_beta_vals = -beta * all_vals
        max_val = np.max(neg_beta_vals)
        exp_vals = np.exp(neg_beta_vals - max_val)
        sum_exp = np.sum(exp_vals)
        log_sum_exp = max_val + np.log(sum_exp)

        # Objective: -log_sum_exp / beta (we want to minimize this)
        obj_val = -log_sum_exp / beta

        # Weights
        weights = exp_vals / sum_exp

        grad_pos = np.zeros_like(positions)

        # Pairwise gradient
        pos_i = positions[PAIR_I]
        pos_j = positions[PAIR_J]
        diffs = pos_i - pos_j  # (n_pairs, 3)

        safe_dists = np.where(dists < 1e-12, 1e-12, dists)
        # d(pair_r)/d(pos_i) = (pos_i - pos_j) / (2 * dist)
        # Gradient of obj = -weights * d(v)/d(pos)
        # So grad_pos[i] -= weights[i] * diffs[i] / (2 * dists[i])
        # grad_pos[j] += weights[i] * diffs[i] / (2 * dists[i])

        w_pair = weights[:N_PAIRS]
        factors = w_pair / (2.0 * safe_dists)  # (n_pairs,)

        # Vectorized accumulation using np.add.at
        contrib = factors[:, np.newaxis] * diffs  # (n_pairs, 3)
        np.add.at(grad_pos, PAIR_I, -contrib)
        np.add.at(grad_pos, PAIR_J, contrib)

        # Wall gradient
        # For wall component k of point i:
        # d(wall)/d(pos) is -sign or +sign depending on which wall
        # wall[:, 0] = x + 0.5 -> d/dx = 1
        # wall[:, 1] = 0.5 - x -> d/dx = -1
        # etc.
        wall_weights = weights[N_PAIRS:]  # (N*6,)
        wall_weights_reshaped = wall_weights.reshape(N, 6)

        # grad_pos[i, 0] -= w[:, 0] (from x+0.5)
        # grad_pos[i, 0] += w[:, 1] (from 0.5-x)
        # grad_pos[i, 1] -= w[:, 2]
        # grad_pos[i, 1] += w[:, 3]
        # grad_pos[i, 2] -= w[:, 4]
        # grad_pos[i, 2] += w[:, 5]

        grad_pos[:, 0] -= wall_weights_reshaped[:, 0]
        grad_pos[:, 0] += wall_weights_reshaped[:, 1]
        grad_pos[:, 1] -= wall_weights_reshaped[:, 2]
        grad_pos[:, 1] += wall_weights_reshaped[:, 3]
        grad_pos[:, 2] -= wall_weights_reshaped[:, 4]
        grad_pos[:, 2] += wall_weights_reshaped[:, 5]

        return obj_val, grad_pos.ravel()

    def repulsive_potential_and_grad(x_flat, p=4.0, eps=1e-6):
        """
        Compute sum of 1/d^p repulsive potentials for pairs and walls.
        This is a smooth, convex-like landscape that helps escape local minima.
        """
        positions = x_flat.reshape(-1, 3)

        # Pairwise
        dists = pdist(positions, 'euclidean')
        safe_dists = np.maximum(dists, eps)
        pair_pot = np.sum(1.0 / (safe_dists ** p))

        # Walls
        walls = np.empty((N, 6))
        walls[:, 0] = positions[:, 0] + 0.5
        walls[:, 1] = 0.5 - positions[:, 0]
        walls[:, 2] = positions[:, 1] + 0.5
        walls[:, 3] = 0.5 - positions[:, 1]
        walls[:, 4] = positions[:, 2] + 0.5
        walls[:, 5] = 0.5 - positions[:, 2]
        safe_walls = np.maximum(walls, eps)
        wall_pot = np.sum(1.0 / (safe_walls ** p))

        total_pot = pair_pot + wall_pot

        # Gradient
        grad_pos = np.zeros_like(positions)

        # Pairwise gradient: d(1/d^p)/d(pos_i) = -p/d^(p+1) * (pos_i - pos_j)/d = -p/d^(p+2) * (pos_i - pos_j)
        pos_i = positions[PAIR_I]
        pos_j = positions[PAIR_J]
        diffs = pos_i - pos_j
        factors = -p / (safe_dists ** (p + 2))
        contrib = factors[:, np.newaxis] * diffs
        np.add.at(grad_pos, PAIR_I, contrib)
        np.add.at(grad_pos, PAIR_J, -contrib)

        # Wall gradient: d(1/w^p)/d(pos) = -p/w^(p+1) * d(w)/d(pos)
        # For wall[:, 0] = x + 0.5: d/dx = 1, so grad = -p/w^(p+1)
        # For wall[:, 1] = 0.5 - x: d/dx = -1, so grad = +p/w^(p+1)
        wall_weights = 1.0 / (safe_walls ** (p + 1)) * p
        wall_weights_reshaped = wall_weights.reshape(N, 6)

        grad_pos[:, 0] -= wall_weights_reshaped[:, 0]
        grad_pos[:, 0] += wall_weights_reshaped[:, 1]
        grad_pos[:, 1] -= wall_weights_reshaped[:, 2]
        grad_pos[:, 1] += wall_weights_reshaped[:, 3]
        grad_pos[:, 2] -= wall_weights_reshaped[:, 4]
        grad_pos[:, 2] += wall_weights_reshaped[:, 5]

        return total_pot, grad_pos.ravel()

    def local_optimize_smooth(positions, beta, maxiter, ftol):
        """Run L-BFGS-B on the smooth surrogate."""
        if remaining() < 5:
            return positions

        bounds = [(-0.499999, 0.499999)] * (N * 3)

        def obj_flat(x):
            val, _ = smooth_surrogate_and_grad(x, beta)
            return val

        def grad_flat(x):
            _, g = smooth_surrogate_and_grad(x, beta)
            return g

        try:
            res = minimize(obj_flat, positions.flatten(), jac=grad_flat, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-8})
            pos_opt = res.x.reshape(-1, 3)
            pos_opt = np.clip(pos_opt, -0.499999, 0.499999)
            return pos_opt
        except Exception:
            return positions

    def local_optimize_repulsive(positions, p=4.0, maxiter=100):
        """Run L-BFGS-B on the repulsive potential (warm-up)."""
        if remaining() < 5:
            return positions

        bounds = [(-0.499999, 0.499999)] * (N * 3)

        def obj_flat(x):
            val, _ = repulsive_potential_and_grad(x, p)
            return val

        def grad_flat(x):
            _, g = repulsive_potential_and_grad(x, p)
            return g

        try:
            res = minimize(obj_flat, positions.flatten(), jac=grad_flat, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-12})
            pos_opt = res.x.reshape(-1, 3)
            pos_opt = np.clip(pos_opt, -0.499999, 0.499999)
            return pos_opt
        except Exception:
            return positions

    def local_optimize_direct(positions, maxiter=200):
        """Run L-BFGS-B on the raw -min_radius objective."""
        if remaining() < 5:
            return positions

        bounds = [(-0.499999, 0.499999)] * (N * 3)

        def obj_flat(x):
            return -compute_min_radius(x.reshape(-1, 3))

        try:
            res = minimize(obj_flat, positions.flatten(), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-14})
            pos_opt = res.x.reshape(-1, 3)
            pos_opt = np.clip(pos_opt, -0.499999, 0.499999)
            return pos_opt
        except Exception:
            return positions

    def make_initial_grid():
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    def make_bcc_lattice():
        positions = []
        a = 0.5
        step = 2*a/3
        for iz in range(3):
            for iy in range(3):
                for ix in range(3):
                    x = -a + (ix + 0.5) * step
                    y = -a + (iy + 0.5) * step
                    z = -a + (iz + 0.5) * step
                    positions.append([x, y, z])
                    if len(positions) < N:
                        positions.append([x + step/2, y + step/2, z + step/2])
        positions = np.array(positions[:N])
        return positions

    def make_random(seed):
        np.random.seed(seed)
        return np.random.uniform(-0.45, 0.45, size=(N, 3))

    best_positions = None
    best_radius = -1.0

    initializations = [
        ("grid", make_initial_grid),
        ("bcc", make_bcc_lattice),
        ("rand0", lambda: make_random(0)),
        ("rand1", lambda: make_random(42)),
        ("rand2", lambda: make_random(123)),
        ("rand3", lambda: make_random(777)),
        ("rand4", lambda: make_random(2024)),
        ("rand5", lambda: make_random(999)),
    ]

    beta_schedule = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0]

    for name, init_func in initializations:
        if remaining() < 15:
            break

        try:
            positions = init_func()
            positions = np.clip(positions, -0.499999, 0.499999)

            # Warm-up with repulsive potential
            if remaining() > 30:
                current_pos = local_optimize_repulsive(positions, p=4.0, maxiter=50)
            else:
                current_pos = positions

            current_r = compute_min_radius(current_pos)

            # Annealing loop
            for beta in beta_schedule:
                if remaining() < 8:
                    break

                if beta < 100:
                    maxiter = 150
                elif beta < 1000:
                    maxiter = 100
                else:
                    maxiter = 50

                current_pos = local_optimize_smooth(current_pos, beta=beta, maxiter=maxiter, ftol=1e-12)
                current_r = compute_min_radius(current_pos)

            # Final direct polish
            if remaining() > 8:
                current_pos = local_optimize_direct(current_pos, maxiter=100)
                current_r = compute_min_radius(current_pos)

            if current_r > best_radius:
                best_radius = current_r
                best_positions = current_pos.copy()

        except Exception:
            continue

    # Perturbations of the best solution
    if best_positions is not None and remaining() > 25:
        for perturb_seed in range(8):
            if remaining() < 12:
                break

            np.random.seed(1000 + perturb_seed)
            perturbed = best_positions + np.random.normal(0, 0.01, size=best_positions.shape)
            perturbed = np.clip(perturbed, -0.499999, 0.499999)

            for beta in [100.0, 500.0, 2000.0, 10000.0]:
                if remaining() < 5:
                    break
                perturbed = local_optimize_smooth(perturbed, beta=beta, maxiter=40, ftol=1e-12)

            if remaining() > 5:
                perturbed = local_optimize_direct(perturbed, maxiter=50)

            r = compute_min_radius(perturbed)
            if r > best_radius:
                best_radius = r
                best_positions = perturbed.copy()

    # Final high-precision polish
    if best_positions is not None and remaining() > 10:
        polished = local_optimize_smooth(best_positions, beta=50000.0, maxiter=100, ftol=1e-15)
        polished = local_optimize_direct(polished, maxiter=50)
        r_polished = compute_min_radius(polished)
        if r_polished > best_radius:
            best_radius = r_polished
            best_positions = polished.copy()

    if best_positions is None:
        best_positions = make_initial_grid()

    best_positions = np.clip(best_positions, -0.499999, 0.499999)

    return best_positions

if __name__ == "__main__":
    pos = construct()
    print("Radius:", compute_min_radius(pos))
