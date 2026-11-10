import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin below 630s limit

    def _compute_P(support, weights):
        """Compute P[X1+X2+X3 < 2*X4] exactly in float64."""
        s = np.asarray(support, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        n = len(s)
        if n == 0:
            return 0.0

        w_sum = w.sum()
        if w_sum <= 1e-9:
            return 0.0
        w = w / w_sum

        # Precompute all pairwise sums and their probabilities
        pair_sums = s[:, None] + s[None, :]
        pair_probs = w[:, None] * w[None, :]

        # Flatten and sort pair sums for efficient prefix-sum queries
        ps_flat = pair_sums.flatten()
        pp_flat = pair_probs.flatten()

        sort_idx = np.argsort(ps_flat)
        ps_sorted = ps_flat[sort_idx]
        pp_sorted = pp_flat[sort_idx]
        pp_prefix = np.cumsum(pp_sorted)

        # T[k,l] = 2*s_l - s_k
        T = 2.0 * s[None, :] - s[:, None]  # shape (n, n), T[k,l] = 2*s_l - s_k
        T_flat = T.flatten()

        # For each T value, find count of pair sums strictly less than T
        # searchsorted with side='left' gives index of first element >= T
        # so elements [0, idx) are < T
        idx = np.searchsorted(ps_sorted, T_flat, side='left')
        idx = idx.reshape(n, n)

        # prob that s_i + s_j < T[k,l] = pp_prefix[idx-1] if idx>0 else 0
        safe_idx = np.maximum(idx - 1, 0)
        prob_z_less = pp_prefix[safe_idx]
        prob_z_less = np.where(idx > 0, prob_z_less, 0.0)

        # P = sum over k,l of w_k * w_l * prob_z_less[k,l]
        wk = w[:, None]
        wl = w[None, :]
        total_prob = np.sum(wk * wl * prob_z_less)
        return float(total_prob)

    def _neg_P(params, support):
        """Negative P for minimization. params are weights."""
        w = np.maximum(params, 0.0)
        w_sum = w.sum()
        if w_sum <= 1e-9:
            return 0.0
        w = w / w_sum
        return -_compute_P(support, w)

    def _optimize_weights(support, rng, time_budget):
        """Optimize weights for a fixed support using projected gradient."""
        n = len(support)
        best_w = np.ones(n) / n
        best_score = _compute_P(support, best_w)

        # Use scipy minimize with SLSQP or Nelder-Mead for small n
        from scipy.optimize import minimize

        t0 = time.time()
        w_init = np.ones(n) / n

        # Try a few random initializations
        for attempt in range(5):
            if time.time() > t0 + time_budget:
                break

            if attempt == 0:
                w0 = np.ones(n) / n
            else:
                w0 = rng.dirichlet(np.ones(n))

            res = minimize(
                _neg_P, w0, args=(support,),
                method='SLSQP',
                bounds=[(0.0, 1.0)] * n,
                constraints=[{'type': 'eq', 'fun': lambda x: x.sum() - 1.0}],
                options={'maxiter': 200, 'ftol': 1e-10}
            )

            if res.success:
                w_opt = np.maximum(res.x, 0.0)
                w_sum = w_opt.sum()
                if w_sum > 1e-9:
                    w_opt = w_opt / w_sum
                    score = _compute_P(support, w_opt)
                    if score > best_score:
                        best_score = score
                        best_w = w_opt.copy()

        return best_w, best_score

    def _local_search_joint(support, weights, rng, budget_iters, base_step):
        """Joint local search on support and weights."""
        n = len(support)
        cur_s = support.copy()
        cur_w = weights.copy()
        cur_score = _compute_P(cur_s, cur_w)
        step = base_step

        for it in range(budget_iters):
            if time.time() > deadline - 5:
                break
            eff_step = step * max(0.1, 1.0 - 0.7 * it / budget_iters)

            new_s = cur_s.copy()
            idx = rng.integers(n)
            new_s[idx] += rng.normal(0, eff_step)
            new_s = np.maximum(new_s, 0.0)

            new_w = cur_w.copy()
            new_w += rng.normal(0, eff_step * 0.3, n)
            new_w = np.maximum(new_w, 0.0)
            ws = np.sum(new_w)
            if ws < 1e-9:
                continue
            new_w = new_w / ws

            new_score = _compute_P(new_s, new_w)
            if new_score > cur_score:
                cur_s = new_s
                cur_w = new_w
                cur_score = new_score
                step *= 0.97
            else:
                if rng.random() < 0.03:
                    step = min(step * 1.25, base_step)

        return cur_s, cur_w, cur_score

    def _generate_structured_support(n, rng):
        """Generate diverse structured supports."""
        method = rng.integers(0, 6)

        if method == 0:
            # Uniform on [0, S]
            S = rng.uniform(0.5, 5.0)
            return np.linspace(0.0, S, n)

        elif method == 1:
            # Beta-like: concentrated near 0
            S = rng.uniform(1.0, 4.0)
            alpha = rng.uniform(0.5, 3.0)
            beta = rng.uniform(1.0, 5.0)
            u = rng.beta(alpha, beta, n)
            return np.sort(u * S)

        elif method == 2:
            # Two-point mixture: mass at 0 and uniform on [a, b]
            a = rng.uniform(0.1, 2.0)
            b = rng.uniform(a + 0.5, 5.0)
            half = n // 2
            s0 = np.zeros(half)
            s1 = np.linspace(a, b, n - half)
            return np.sort(np.concatenate([s0, s1]))

        elif method == 3:
            # Geometric spacing
            S = rng.uniform(1.0, 5.0)
            r = rng.uniform(0.5, 0.95)
            base = S / (1.0 - r)
            s = base * (1.0 - r ** np.arange(n))
            return np.sort(s)

        elif method == 4:
            # Power law spacing
            S = rng.uniform(1.0, 5.0)
            p = rng.uniform(0.5, 3.0)
            u = np.linspace(0.0, 1.0, n)
            return np.sort(u ** p * S)

        else:
            # Random sorted
            S = rng.uniform(0.5, 5.0)
            return np.sort(rng.uniform(0.0, S, n))

    best_score = -1.0
    best_support = None
    best_weights = None

    rng = np.random.default_rng(2024)

    # Phase 1: Explore different support structures with weight optimization
    n_sizes = [8, 12, 16, 20, 24, 32, 48, 64]

    for n in n_sizes:
        if time.time() > deadline - 30:
            break

        n_trials = max(3, min(10, int((deadline - time.time()) / (n * n * 0.05))))

        for trial in range(n_trials):
            if time.time() > deadline - 10:
                break

            support = _generate_structured_support(n, rng)
            # Add small perturbation to break symmetries
            support = support + rng.normal(0, 0.01, n)
            support = np.maximum(support, 0.0)
            support = np.sort(support)

            # Optimize weights
            time_per = max(2.0, min(15.0, (deadline - time.time()) / max(1, n_trials - trial)))
            weights, score = _optimize_weights(support, rng, time_per)

            if score > best_score:
                best_score = score
                best_support = support.copy()
                best_weights = weights.copy()

    # Phase 2: Joint local search refinement on best solution
    if best_support is not None and time.time() < deadline - 10:
        remaining = deadline - 10 - time.time()
        budget_iters = max(50, int(remaining / 0.5))
        s0, w0, sc0 = _local_search_joint(best_support, best_weights, rng, budget_iters, 0.05)
        if sc0 > best_score:
            best_score = sc0
            best_support = s0
            best_weights = w0

    # Phase 3: Final weight-only optimization on best support
    if best_support is not None and time.time() < deadline - 5:
        w_opt, sc_opt = _optimize_weights(best_support, rng, deadline - 5 - time.time())
        if sc_opt > best_score:
            best_score = sc_opt
            best_weights = w_opt

    # Fallback
    if best_support is None:
        n = 32
        return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64) / n

    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    w_sum = best_weights.sum()
    if w_sum <= 1e-9:
        best_weights = np.ones(len(best_support)) / len(best_support)
    else:
        best_weights = best_weights / w_sum

    return best_support, best_weights
