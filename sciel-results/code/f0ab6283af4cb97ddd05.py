import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    def soft_min(x, beta):
        """Smooth approximation of min(x) using log-sum-exp."""
        m = np.min(x)
        exp_terms = np.exp(beta * (x - m))
        log_sum = np.log(np.sum(exp_terms))
        return m - (1.0 / beta) * log_sum

    def objective_and_grad(params, beta):
        centers = params.reshape(N, 3)

        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)

        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        pair_clears = pair_dists / 2.0

        # Wall clearances
        abs_c = np.abs(centers)
        wall_clears = 0.5 - np.max(abs_c, axis=1)

        all_clears = np.concatenate([pair_clears, wall_clears])

        # Soft min via log-sum-exp
        m = np.min(all_clears)
        exp_terms = np.exp(beta * (all_clears - m))
        log_sum = np.log(np.sum(exp_terms))
        soft_min_val = m - (1.0 / beta) * log_sum

        val = -soft_min_val

        weights = exp_terms / np.sum(exp_terms)

        grad = np.zeros_like(centers)

        n_pairs = len(pair_clears)
        w_pairs = weights[:n_pairs]
        w_walls = weights[n_pairs:]

        i_idx, j_idx = iu

        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]

        W = np.zeros((N, N))
        W[i_idx, j_idx] = w_pairs
        W[j_idx, i_idx] = -w_pairs

        grad_pairs = 0.5 * np.einsum('ij,ijk->ik', W, unit_vecs)

        gamma = 50.0
        exp_wall = np.exp(gamma * (abs_c - np.max(abs_c, axis=1, keepdims=True)))
        p_wall = exp_wall / np.sum(exp_wall, axis=1, keepdims=True)

        signs = np.sign(centers)
        grad_walls = -signs * p_wall * w_walls[:, np.newaxis]

        grad_total = grad_pairs + grad_walls

        return val, grad_total.flatten()

    def score(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0

        max_c = np.max(np.abs(centers), axis=1)
        min_wall = 0.5 - np.min(max_c)

        return min(min_pair, min_wall)

    def init_lattice():
        a = 0.32
        pts = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    pts.append([i*a, j*a, k*a])
        pts = np.array(pts)
        mask = np.all(np.abs(pts) < 0.48, axis=1)
        pts = pts[mask]
        np.random.shuffle(pts)
        if len(pts) < N:
            extra = np.random.uniform(-0.4, 0.4, (N - len(pts), 3))
            pts = np.vstack([pts, extra])
        return pts[:N]

    def init_random():
        return np.random.uniform(-0.4, 0.4, (N, 3))

    def optimize(centers_init, time_limit=3.0):
        import time
        start_time = time.time()
        betas = [20, 50, 100, 200, 500, 1000]
        current = centers_init.copy()

        for i, beta in enumerate(betas):
            if time.time() - start_time > time_limit - 0.5:
                break
            max_iter = 50 + i * 30
            try:
                res = minimize(
                    lambda p: objective_and_grad(p, beta)[0],
                    current.flatten(),
                    jac=lambda p: objective_and_grad(p, beta)[1],
                    method='L-BFGS-B',
                    bounds=[(-0.499, 0.499)] * (N * 3),
                    options={'maxiter': max_iter, 'gtol': 1e-8}
                )
                current = res.x.reshape(N, 3)
            except Exception:
                pass
        return current

    best_centers = None
    best_score = -np.inf

    import time
    start_total = time.time()
    timeout = 110.0

    for seed in range(30):
        if time.time() - start_total > timeout:
            break
        np.random.seed(seed)
        if seed % 2 == 0:
            c_init = init_lattice()
        else:
            c_init = init_random()
        try:
            c_opt = optimize(c_init, time_limit=3.5)
            s = score(c_opt)
            if s > best_score:
                best_score = s
                best_centers = c_opt
        except Exception:
            pass

    return best_centers
