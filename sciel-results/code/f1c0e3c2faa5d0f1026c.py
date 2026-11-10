# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Maximizes H_{-1} / max(H_0, H_1, H_2) where H_r is the Shannon entropy
    of the distribution of x + r*y.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 570  # Safety margin for 600s limit

    # Pre-compute grid coordinates
    y, x = np.mgrid[0:n, 0:n]
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    # Pre-compute projection indices for fast bincount
    idx_m1 = (x - y + n - 1).astype(np.int32).ravel()
    idx_0 = x.astype(np.int32).ravel()
    idx_1 = (x + y).astype(np.int32).ravel()
    idx_2 = (x + 2 * y).astype(np.int32).ravel()

    size_m1 = 2 * n - 1
    size_0 = n
    size_1 = 2 * n - 1
    size_2 = 3 * n - 1

    def compute_entropies(P):
        """Compute H_{-1}, H_0, H_1, H_2 for a given distribution P."""
        w = P.ravel()

        counts_m1 = np.bincount(idx_m1, weights=w, minlength=size_m1)
        p_m1 = counts_m1[counts_m1 > 0]
        H_m1 = -np.sum(p_m1 * np.log2(p_m1)) if len(p_m1) > 0 else 0.0

        counts_0 = np.bincount(idx_0, weights=w, minlength=size_0)
        p_0 = counts_0[counts_0 > 0]
        H_0 = -np.sum(p_0 * np.log2(p_0)) if len(p_0) > 0 else 0.0

        counts_1 = np.bincount(idx_1, weights=w, minlength=size_1)
        p_1 = counts_1[counts_1 > 0]
        H_1 = -np.sum(p_1 * np.log2(p_1)) if len(p_1) > 0 else 0.0

        counts_2 = np.bincount(idx_2, weights=w, minlength=size_2)
        p_2 = counts_2[counts_2 > 0]
        H_2 = -np.sum(p_2 * np.log2(p_2)) if len(p_2) > 0 else 0.0

        return H_m1, H_0, H_1, H_2

    def get_score(P):
        H_m1, H_0, H_1, H_2 = compute_entropies(P)
        M = max(H_0, H_1, H_2)
        if M <= 1e-9:
            return 0.0
        return H_m1 / M

    # Baseline: Uniform
    P_uniform = np.ones((n, n)) / (n * n)
    best_P = P_uniform.copy()
    best_score = get_score(best_P)

    rng = np.random.default_rng(42)

    # --- Phase 1: Sparse combinatorial support search ---
    # The key insight: for a fixed support S, the optimal weights that maximize
    # H_{-1}/max(H_0,H_1,H_2) can be found. But first, let's find good supports.
    #
    # Strategy: Build supports that spread out x-y values (to maximize H_{-1})
    # while keeping x, x+y, x+2y projections concentrated (to minimize M).
    #
    # x-y ranges from -9 to 9 (19 values). To maximize H_{-1}, we want mass
    # spread across many x-y values.
    # x ranges 0-9 (10 values), x+y ranges 0-18 (19 values), x+2y ranges 0-27 (28 values).
    #
    # Key structural idea: Use cells where x-y varies widely but x is concentrated.
    # E.g., cells (0,0), (1,1), ..., (9,9) all have x-y=0 (bad for H_{-1}).
    # Cells (0,0), (0,1), ..., (0,9) have x-y = 0,-1,...,-9 (good for H_{-1})
    # but x=0 always (bad for H_0, but H_0 would be 0).
    #
    # Let's try: pick a set of cells that covers many x-y values,
    # and check the resulting score with uniform weights.

    all_cells = [(i, j) for i in range(n) for j in range(n)]  # (y, x)

    # Precompute projections for each cell
    cell_proj = {}
    for (cy, cx) in all_cells:
        cell_proj[(cy, cx)] = (cx - cy, cx, cx + cy, cx + 2 * cy)

    # Generate candidate supports
    candidate_supports = []

    # 1. Diagonal strips: cells with x-y in a range, but concentrated in x
    #    For each x value, take a range of y values to spread x-y
    for x_start in range(0, n, 2):
        for x_width in [1, 2, 3]:
            for y_start in range(0, n, 2):
                for y_width in [1, 2, 3, 4, 5]:
                    support = []
                    for cy in range(y_start, min(y_start + y_width, n)):
                        for cx in range(x_start, min(x_start + x_width, n)):
                            support.append((cy, cx))
                    if 4 <= len(support) <= 40:
                        candidate_supports.append(set(support))

    # 2. Cross patterns: spread in both x and y but structured
    for cx_center in [2, 4, 6, 8]:
        for cy_center in [2, 4, 6, 8]:
            for radius in [2, 3, 4]:
                support = set()
                for cy in range(max(0, cy_center - radius), min(n, cy_center + radius + 1)):
                    for cx in range(max(0, cx_center - radius), min(n, cx_center + radius + 1)):
                        # Only include if on a "cross" pattern
                        if abs(cx - cx_center) + abs(cy - cy_center) <= radius:
                            support.add((cy, cx))
                if 4 <= len(support) <= 40:
                    candidate_supports.append(support)

    # 3. Random sparse supports
    for _ in range(200):
        k = rng.integers(8, 35)
        idx = rng.choice(n * n, size=k, replace=False)
        support = set()
        for i in idx:
            cy, cx = divmod(i, n)
            support.add((cy, cx))
        candidate_supports.append(support)

    # 4. Structured: pick cells to maximize x-y spread while minimizing x spread
    #    For each x in a small set, pick y values that give distinct x-y
    for x_set in [[0, 9], [0, 1, 8, 9], [2, 7], [1, 4, 5, 8]]:
        for y_set in [[0, 9], [0, 1, 8, 9], [2, 7], [1, 4, 5, 8], [0, 2, 7, 9]]:
            support = set((cy, cx) for cy in y_set for cx in x_set)
            if 4 <= len(support) <= 40:
                candidate_supports.append(support)

    # 5. Concentrated x, spread y
    for cx in range(n):
        for y_span in [5, 8, 10]:
            for y_offset in range(0, n - y_span + 1):
                support = set((cy, cx) for cy in range(y_offset, y_offset + y_span))
                if 4 <= len(support) <= 40:
                    candidate_supports.append(support)

    # 6. Concentrated y, spread x
    for cy in range(n):
        for x_span in [5, 8, 10]:
            for x_offset in range(0, n - x_span + 1):
                support = set((cy, cx) for cx in range(x_offset, x_offset + x_span))
                if 4 <= len(support) <= 40:
                    candidate_supports.append(support)

    # Evaluate all candidate supports with uniform weights
    for support in candidate_supports:
        if time.time() > deadline - 120:
            break
        P_cand = np.zeros((n, n))
        for (cy, cx) in support:
            P_cand[cy, cx] = 1.0 / len(support)
        score_cand = get_score(P_cand)
        if score_cand > best_score:
            best_score = score_cand
            best_P = P_cand.copy()

    # --- Phase 2: Refine best support with non-uniform weights ---
    # Use SLSQP to optimize weights on the support of best_P
    support_cells = [(cy, cx) for cy in range(n) for cx in range(n) if best_P[cy, cx] > 1e-12]
    if not support_cells:
        support_cells = [(0, 0)]

    # Build index map
    support_indices = np.array([cy * n + cx for (cy, cx) in support_cells])
    k_support = len(support_indices)

    # Precompute projection indices for support cells
    idx_m1_s = idx_m1[support_indices]
    idx_0_s = idx_0[support_indices]
    idx_1_s = idx_1[support_indices]
    idx_2_s = idx_2[support_indices]

    def objective_support(w, beta=50.0):
        """Objective for optimizing weights on support."""
        if time.time() > deadline - 30:
            return 1000.0, np.zeros_like(w)

        # Compute entropies
        q_m1 = np.bincount(idx_m1_s, weights=w, minlength=size_m1)
        mask_m1 = q_m1 > 0
        if np.sum(mask_m1) == 0:
            return 1000.0, np.zeros_like(w)
        p_m1 = q_m1[mask_m1]
        H_m1 = -np.sum(p_m1 * np.log2(p_m1))
        grad_H_m1 = np.zeros(size_m1)
        grad_H_m1[mask_m1] = -np.log2(p_m1) - 1.0 / np.log(2.0)
        grad_w_m1 = grad_H_m1[idx_m1_s]

        Hs = []
        grad_ws = []
        for idx_s, n_bins in [(idx_0_s, size_0), (idx_1_s, size_1), (idx_2_s, size_2)]:
            q = np.bincount(idx_s, weights=w, minlength=n_bins)
            mask = q > 0
            if np.sum(mask) == 0:
                H = 0.0
                gw = np.zeros_like(w)
            else:
                p = q[mask]
                H = -np.sum(p * np.log2(p))
                grad_q = np.zeros(n_bins)
                grad_q[mask] = -np.log2(p) - 1.0 / np.log(2.0)
                gw = grad_q[idx_s]
            Hs.append(H)
            grad_ws.append(gw)

        Hs = np.array(Hs)
        if np.max(Hs) < 1e-9:
            return 1000.0, np.zeros_like(w)

        # Soft max for M
        H_max = np.max(Hs)
        exp_Hs = np.exp(beta * (Hs - H_max))
        sum_exp = np.sum(exp_Hs)
        M = (H_max + np.log(sum_exp)) / beta

        if M < 1e-9:
            return 1000.0, np.zeros_like(w)

        w_sm = exp_Hs / sum_exp
        grad_M = np.zeros_like(w)
        for i in range(3):
            grad_M += w_sm[i] * grad_ws[i]

        grad_R = (M * grad_w_m1 - H_m1 * grad_M) / (M * M)
        obj = -H_m1 / M
        grad_obj = -grad_R

        return obj, grad_obj

    # Optimize weights on support
    w0 = np.ones(k_support) / k_support
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                   'jac': lambda x: np.ones_like(x)}]
    bounds = [(0.0, 1.0) for _ in range(k_support)]

    for beta in [30, 50, 100, 200]:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(
                lambda x: objective_support(x, beta=beta)[0],
                w0,
                jac=lambda x: objective_support(x, beta=beta)[1],
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            w_opt = np.clip(res.x, 0, None)
            s = np.sum(w_opt)
            if s > 1e-12:
                w_opt /= s
                P_cand = np.zeros((n, n))
                for i, idx in enumerate(support_indices):
                    P_cand.ravel()[idx] = w_opt[i]
                score_cand = get_score(P_cand)
                if score_cand > best_score:
                    best_score = score_cand
                    best_P = P_cand.copy()
                    w0 = w_opt.copy()
        except Exception:
            pass

    # --- Phase 3: Full-space SLSQP refinement ---
    def objective_full(P_flat, beta=50.0):
        """Objective for full 100D space."""
        if time.time() > deadline - 10:
            return 1000.0, np.zeros_like(P_flat)

        q_m1 = np.bincount(idx_m1, weights=P_flat, minlength=size_m1)
        mask_m1 = q_m1 > 0
        if np.sum(mask_m1) == 0:
            return 1000.0, np.zeros_like(P_flat)
        p_m1 = q_m1[mask_m1]
        H_m1 = -np.sum(p_m1 * np.log2(p_m1))
        grad_H_m1 = np.zeros(size_m1)
        grad_H_m1[mask_m1] = -np.log2(p_m1) - 1.0 / np.log(2.0)
        grad_P_m1 = grad_H_m1[idx_m1]

        Hs = []
        grad_Ps = []
        for idx_flat, n_bins in [(idx_0, size_0), (idx_1, size_1), (idx_2, size_2)]:
            q = np.bincount(idx_flat, weights=P_flat, minlength=n_bins)
            mask = q > 0
            if np.sum(mask) == 0:
                H = 0.0
                gP = np.zeros_like(P_flat)
            else:
                p = q[mask]
                H = -np.sum(p * np.log2(p))
                grad_q = np.zeros(n_bins)
                grad_q[mask] = -np.log2(p) - 1.0 / np.log(2.0)
                gP = grad_q[idx_flat]
            Hs.append(H)
            grad_Ps.append(gP)

        Hs = np.array(Hs)
        if np.max(Hs) < 1e-9:
            return 1000.0, np.zeros_like(P_flat)

        H_max = np.max(Hs)
        exp_Hs = np.exp(beta * (Hs - H_max))
        sum_exp = np.sum(exp_Hs)
        M = (H_max + np.log(sum_exp)) / beta

        if M < 1e-9:
            return 1000.0, np.zeros_like(P_flat)

        w_sm = exp_Hs / sum_exp
        grad_M = np.zeros_like(P_flat)
        for i in range(3):
            grad_M += w_sm[i] * grad_Ps[i]

        grad_R = (M * grad_P_m1 - H_m1 * grad_M) / (M * M)
        obj = -H_m1 / M
        grad_obj = -grad_R

        return obj, grad_obj

    constraints_full = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                        'jac': lambda x: np.ones_like(x)}]
    bounds_full = [(0.0, 1.0) for _ in range(n * n)]

    for beta in [30, 50, 100, 200, 400]:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(
                lambda x: objective_full(x, beta=beta)[0],
                best_P.ravel(),
                jac=lambda x: objective_full(x, beta=beta)[1],
                method='SLSQP',
                bounds=bounds_full,
                constraints=constraints_full,
                options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
            )
            P_cand = np.clip(res.x, 0, None).reshape(n, n)
            s = np.sum(P_cand)
            if s > 1e-12:
                P_cand /= s
                score_cand = get_score(P_cand)
                if score_cand > best_score:
                    best_score = score_cand
                    best_P = P_cand.copy()
        except Exception:
            pass

    # Final validation
    best_P = np.clip(best_P, 0, None)
    s = np.sum(best_P)
    if s > 1e-9:
        best_P = best_P / s
    else:
        best_P = P_uniform

    return best_P

# EVOLVE-BLOCK-END
