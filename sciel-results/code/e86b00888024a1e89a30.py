import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sequence of 50 nonnegative heights.
    Strategy: Analytical 2-point cluster + scipy L-BFGS-B polish with log-variables.
    """
    t0 = time.time()
    deadline = 100.0  # Safety margin well under 120s

    def compute_score(h):
        """Compute the ratio: L2^2 / (L1 * sup_norm_autoconv)."""
        h = np.asarray(h, dtype=float)
        l1 = float(np.sum(h))
        if l1 < 1e-15:
            return 0.0
        l2_sq = float(np.sum(h * h))
        conv = np.convolve(h, h, mode='full')
        conv_inf = float(np.max(conv))
        if conv_inf < 1e-15:
            return 0.0
        return l2_sq / (l1 * conv_inf)

    def neg_score_from_x(x):
        """Negative score from log-variables x, where h = softmax(x)."""
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        h = h / np.sum(h)
        return -compute_score(h)

    def polish(h_init, max_iter=500):
        """Polish using L-BFGS-B in log-space."""
        h0 = np.asarray(h_init, dtype=float)
        l1 = float(np.sum(h0))
        if l1 < 1e-15:
            return h0, 0.0
        h0 = h0 / l1
        # Initialize log-variables
        eps = 1e-15
        x0 = np.log(np.maximum(h0, eps))
        x0 = x0 - np.max(x0)  # center for stability

        try:
            res = minimize(
                neg_score_from_x, x0,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            if res.fun < 0:
                x_opt = res.x
                h_opt = np.exp(x_opt - np.max(x_opt))
                h_opt = h_opt / np.sum(h_opt)
                s = compute_score(h_opt)
                if s > 0:
                    return h_opt, s
        except Exception:
            pass
        return h0, compute_score(h0)

    best_score = 0.0
    best_h = None

    # --- Phase 1: 2-point cluster grid search (analytically optimal at r=2) ---
    # For 2-point at positions i,j: ratio depends only on weight ratio, not separation
    # Optimal: a:b = 2:1 gives ratio 1.25
    for sep in [1, 2, 3, 5, 10, 20, 25, 49]:
        if time.time() - t0 > deadline:
            break
        for r in np.linspace(0.5, 10.0, 50):
            if time.time() - t0 > deadline:
                break
            h = np.zeros(N)
            h[0] = r / (r + 1.0)
            h[sep] = 1.0 / (r + 1.0)
            s = compute_score(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()

    # --- Phase 2: 3-point cluster search (non-adjacent) ---
    if time.time() - t0 < deadline - 30:
        for d1 in [1, 2, 3, 5]:
            if time.time() - t0 > deadline - 30:
                break
            for d2 in [1, 2, 3, 5]:
                if time.time() - t0 > deadline - 30:
                    break
                for r1 in np.linspace(0.5, 5.0, 15):
                    if time.time() - t0 > deadline - 30:
                        break
                    for r2 in np.linspace(0.5, 5.0, 15):
                        if time.time() - t0 > deadline - 30:
                            break
                        h = np.zeros(N)
                        a, b, c = r1, 1.0, r2
                        tot = a + b + c
                        h[0] = a / tot
                        h[d1] = b / tot
                        h[d1 + d2] = c / tot
                        s = compute_score(h)
                        if s > best_score:
                            best_score = s
                            best_h = h.copy()

    # --- Phase 3: Polish best candidate with scipy ---
    if best_h is not None and time.time() - t0 < deadline - 40:
        h_pol, s_pol = polish(best_h, max_iter=300)
        if s_pol > best_score:
            best_score = s_pol
            best_h = h_pol

    # --- Phase 4: Try additional diverse starting points for polishing ---
    if time.time() - t0 < deadline - 50:
        # Uniform
        h_uni = np.ones(N) / N
        h_p, s_p = polish(h_uni, max_iter=200)
        if s_p > best_score:
            best_score = s_p
            best_h = h_p

        # 2-point with various ratios and separations
        for sep in [1, 3, 7, 15]:
            if time.time() - t0 > deadline - 50:
                break
            for r in [1.5, 2.0, 3.0, 5.0]:
                if time.time() - t0 > deadline - 50:
                    break
                h = np.zeros(N)
                h[0] = r / (r + 1.0)
                h[sep] = 1.0 / (r + 1.0)
                h_p, s_p = polish(h, max_iter=150)
                if s_p > best_score:
                    best_score = s_p
                    best_h = h_p

        # 3-point diverse
        for (d1, d2) in [(1, 3), (2, 5), (3, 7)]:
            if time.time() - t0 > deadline - 50:
                break
            for (r1, r2) in [(2.0, 1.0), (1.0, 2.0), (3.0, 0.5)]:
                if time.time() - t0 > deadline - 50:
                    break
                h = np.zeros(N)
                a, b, c = r1, 1.0, r2
                tot = a + b + c
                h[0] = a / tot
                h[d1] = b / tot
                h[d1 + d2] = c / tot
                h_p, s_p = polish(h, max_iter=150)
                if s_p > best_score:
                    best_score = s_p
                    best_h = h_p

    # --- Phase 5: Final high-precision polish if time permits ---
    if best_h is not None and time.time() - t0 < deadline - 10:
        h_final, s_final = polish(best_h, max_iter=500)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final

    if best_h is None:
        best_h = [0.0] * N
        best_h[0] = 2.0 / 3.0
        best_h[1] = 1.0 / 3.0
    else:
        best_h = best_h.tolist()

    return best_h
# EVOLVE-BLOCK-END
