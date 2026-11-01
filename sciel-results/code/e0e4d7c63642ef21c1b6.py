import numpy as np
import time

N = 100

def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill

    # --- Initial grid: 5x5x5, take 100 points ---
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    centers = centers.astype(np.float64)

    def compute_r(c):
        """Compute the radius: min of half-pair-distances and boundary distances."""
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        # Boundary
        bd = 0.5 - np.abs(c)
        min_bound = np.min(bd)
        return min(min_pair, min_bound)

    def compute_min_pair_and_bound(c):
        """Return (min_pair_dist, min_boundary) separately for SA scoring."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        bd = 0.5 - np.abs(c)
        min_bound = np.min(bd)
        return min_pair, min_bound

    def soft_min_r(c, beta):
        """Smooth approximation of the radius using log-sum-exp for min."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists2, np.inf)
        # soft min of distances / 2
        # log-sum-exp trick: min(d_i) ≈ -(1/beta) * log(sum(exp(-beta * d_i)))
        # We work with distances (not squared) for accuracy
        dists = np.sqrt(np.maximum(dists2, 0))
        np.fill_diagonal(dists, np.inf)
        # Boundary distances
        bd = 0.5 - np.abs(c)
        # Combine all constraints: we want min over all pair distances/2 and boundary distances
        # Use a large beta for soft min
        all_dists = np.concatenate([dists.flatten(), bd.flatten()])
        all_dists = all_dists[all_dists < np.inf]
        if len(all_dists) == 0:
            return 0.0
        # Soft min: -(1/beta)*log(sum(exp(-beta*x)))
        max_val = np.max(all_dists)
        shifted = -beta * (all_dists - max_val)
        # Clip to avoid overflow
        shifted = np.clip(shifted, -700, 700)
        log_sum = max_val * beta + np.log(np.sum(np.exp(shifted)))
        soft_min = -(1.0 / beta) * log_sum + max_val
        # Actually: soft_min = -(1/beta)*log(sum(exp(-beta*x)))
        # = -(1/beta) * [ log(sum(exp(-beta*x + beta*max_val))) - beta*max_val ]
        # = max_val - (1/beta)*log(sum(exp(-beta*(x - max_val))))
        # Let me redo:
        # soft_min(x) = -(1/beta) * log(sum_i exp(-beta * x_i))
        # = -(1/beta) * log(exp(-beta*max_val) * sum_i exp(-beta*(x_i - max_val)))
        # = -(1/beta) * [ -beta*max_val + log(sum_i exp(-beta*(x_i - max_val))) ]
        # = max_val - (1/beta) * log(sum_i exp(-beta*(x_i - max_val)))
        # Since x_i - max_val <= 0, -beta*(x_i-max_val) >= 0
        exp_terms = np.exp(np.clip(-beta * (all_dists - max_val), -700, 700))
        soft_min_val = max_val - (1.0 / beta) * np.log(np.sum(exp_terms))
        return soft_min_val

    # --- Simulated Annealing ---
    best_centers = centers.copy()
    best_r = compute_r(best_centers)
    current_r = best_r
    current_centers = centers.copy()

    T_start = 0.01
    T_end = 1e-6
    n_sa_iters = 500000
    # Adaptive: run as many as fit in time

    np.random.seed(42)

    it = 0
    while it < n_sa_iters:
        if time.time() > deadline - 20.0:
            break

        # Pick a random sphere
        i = np.random.randint(N)

        # Propose a move: small random perturbation
        step_scale = T_start * (T_end / T_start) ** (it / n_sa_iters) * 0.5
        delta = np.random.randn(3) * step_scale
        new_centers = current_centers.copy()
        new_centers[i] += delta

        # Check boundary constraints (reject if outside)
        if np.any(np.abs(new_centers[i]) >= 0.5):
            it += 1
            continue

        # Compute new radius (approximate for speed)
        new_min_pair, new_min_bound = compute_min_pair_and_bound(new_centers)
        new_r = min(new_min_pair / 2.0, new_min_bound)

        # Accept/reject
        if new_r >= current_r:
            current_r = new_r
            current_centers = new_centers
            if current_r > best_r:
                best_r = current_r
                best_centers = current_centers.copy()
        else:
            # Metropolis acceptance
            delta_r = current_r - new_r
            T = T_start * (T_end / T_start) ** (it / n_sa_iters)
            if delta_r < T and np.random.random() < np.exp(-delta_r / max(T, 1e-15)):
                current_r = new_r
                current_centers = new_centers

        it += 1

        # Periodic: try larger moves for some spheres to escape local minima
        if it % 5000 == 0:
            # Try moving a random subset with larger steps
            for _ in range(3):
                j = np.random.randint(N)
                step = 0.02 * np.random.randn(3)
                trial = current_centers.copy()
                trial[j] += step
                if np.any(np.abs(trial[j]) >= 0.5):
                    continue
                tr_min_pair, tr_min_bound = compute_min_pair_and_bound(trial)
                tr_r = min(tr_min_pair / 2.0, tr_min_bound)
                if tr_r >= current_r - 0.005:
                    current_centers = trial
                    current_r = tr_r
                    if current_r > best_r:
                        best_r = current_r
                        best_centers = current_centers.copy()

    # --- LogSumExp Polish ---
    # Use scipy if available for L-BFGS-B
    from scipy.optimize import minimize

    def neg_soft_min_r(c_flat, beta):
        c = c_flat.reshape(N, 3)
        return -soft_min_r(c, beta)

    def neg_r_exact(c_flat):
        c = c_flat.reshape(N, 3)
        return -compute_r(c)

    # Polish with increasing beta
    c_flat = best_centers.flatten()
    for beta in [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]:
        if time.time() > deadline - 5.0:
            break
        try:
            res = minimize(
                neg_soft_min_r,
                c_flat,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            c_new = res.x.reshape(N, 3)
            # Project back into cube
            c_new = np.clip(c_new, -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = compute_r(c_new)
            if r_new > best_r:
                best_r = r_new
                best_centers = c_new
                c_flat = c_new.flatten()
        except Exception:
            pass

    # Final exact polish with tight tolerance
    if time.time() < deadline - 3.0:
        try:
            res = minimize(
                neg_r_exact,
                best_centers.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12},
            )
            c_new = res.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = compute_r(c_new)
            if r_new > best_r:
                best_r = r_new
                best_centers = c_new
        except Exception:
            pass

    # Ensure validity: clip to cube, check finiteness
    best_centers = np.clip(best_centers, -0.5 + 1e-10, 0.5 - 1e-10)
    if not np.all(np.isfinite(best_centers)):
        # Fallback to grid
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].astype(np.float64)

    return best_centers
