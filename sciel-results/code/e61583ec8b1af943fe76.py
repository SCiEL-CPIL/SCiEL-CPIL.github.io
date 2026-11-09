# EVOLVE-BLOCK-START
import numpy as np
import time


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Maximizes D/M where D = H_{-1} (entropy of x-y distribution)
    and M = max(H_0, H_1, H_2) (max entropy of x, x+y, x+2y projections).
    """
    n = 10
    deadline = time.time() + 600.0
    safety_margin = 30.0  # seconds before deadline to stop

    # Precompute projection indices
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')  # X[i,j] = j (x), Y[i,j] = i (y)
    # P[y, x] = P[i, j] where i=y, j=x

    # Projection values: proj_r[i,j] = x + r*y = j + r*i
    proj_m1 = X - Y          # x - y (for D)
    proj_0 = X               # x (for H_0)
    proj_1 = X + Y           # x + y (for H_1)
    proj_2 = X + 2*Y         # x + 2y (for H_2)

    # Flatten for computation
    proj_m1_f = proj_m1.flatten()
    proj_0_f = proj_0.flatten()
    proj_1_f = proj_1.flatten()
    proj_2_f = proj_2.flatten()

    def compute_entropy(proj_vals, p_flat):
        """Compute Shannon entropy in bits of the projected distribution."""
        # Pool probabilities by projection value
        unique_vals, inverse = np.unique(proj_vals, return_inverse=True)
        q = np.zeros(len(unique_vals))
        np.add.at(q, inverse, p_flat)
        # Shannon entropy ignoring zero probabilities
        mask = q > 0
        if not np.any(mask):
            return 0.0
        q_m = q[mask]
        return -np.sum(q_m * np.log2(q_m))

    def objective_and_grad(log_w):
        """
        Compute objective D/M and gradient w.r.t. log_w.
        Uses soft-max for M to make it differentiable.
        """
        p = np.exp(log_w)
        p = p / p.sum()
        p_flat = p.flatten()

        D = compute_entropy(proj_m1_f, p_flat)
        H0 = compute_entropy(proj_0_f, p_flat)
        H1 = compute_entropy(proj_1_f, p_flat)
        H2 = compute_entropy(proj_2_f, p_flat)

        # Soft max with temperature for smoothness
        temp = 0.01
        Hs = np.array([H0, H1, H2])
        M_soft = temp * np.log(np.sum(np.exp(Hs / temp)))

        ratio = D / M_soft if M_soft > 1e-12 else 0.0

        # Numerical gradient
        grad = np.zeros_like(log_w)
        eps = 1e-5
        for idx in range(log_w.size):
            log_w_p = log_w.copy()
            log_w_p[idx] += eps
            p_p = np.exp(log_w_p)
            p_p = p_p / p_p.sum()
            p_p_flat = p_p.flatten()
            D_p = compute_entropy(proj_m1_f, p_p_flat)
            H0_p = compute_entropy(proj_0_f, p_p_flat)
            H1_p = compute_entropy(proj_1_f, p_p_flat)
            H2_p = compute_entropy(proj_2_f, p_p_flat)
            Hs_p = np.array([H0_p, H1_p, H2_p])
            M_soft_p = temp * np.log(np.sum(np.exp(Hs_p / temp)))
            ratio_p = D_p / M_soft_p if M_soft_p > 1e-12 else 0.0
            grad[idx] = (ratio_p - ratio) / eps

        return ratio, grad

    def optimize_from_init(log_w_init, max_iters=500, lr_init=0.1):
        """Adam-like optimization with learning rate decay."""
        log_w = log_w_init.copy()
        m = np.zeros_like(log_w)
        v = np.zeros_like(log_w)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        lr = lr_init
        best_ratio = -np.inf
        best_log_w = log_w.copy()

        for it in range(max_iters):
            if time.time() > deadline - safety_margin:
                break

            ratio, grad = objective_and_grad(log_w)
            if ratio > best_ratio:
                best_ratio = ratio
                best_log_w = log_w.copy()

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**(it+1))
            v_hat = v / (1 - beta2**(it+1))
            log_w = log_w + lr * m_hat / (np.sqrt(v_hat) + eps)

            # Decay learning rate
            lr = lr_init / (1 + 0.01 * it)

        return best_log_w, best_ratio

    # Multiple random restarts with different initializations
    best_log_w = None
    best_ratio = -np.inf

    # Restart 1: uniform
    log_w_uniform = np.zeros((n, n))
    lw, r = optimize_from_init(log_w_uniform, max_iters=300, lr_init=0.05)
    if r > best_ratio:
        best_ratio = r
        best_log_w = lw

    if time.time() > deadline - safety_margin:
        pass  # skip remaining restarts

    # Restart 2: weighted by |x-y| to boost D
    w_init = 1.0 + np.abs(X - Y)
    log_w_boost = np.log(w_init)
    lw, r = optimize_from_init(log_w_boost, max_iters=300, lr_init=0.05)
    if r > best_ratio:
        best_ratio = r
        best_log_w = lw

    if time.time() > deadline - safety_margin:
        pass

    # Restart 3: diagonal emphasis
    w_diag = np.ones((n, n))
    for i in range(n):
        w_diag[i, i] = 3.0
    log_w_diag = np.log(w_diag)
    lw, r = optimize_from_init(log_w_diag, max_iters=200, lr_init=0.05)
    if r > best_ratio:
        best_ratio = r
        best_log_w = lw

    if time.time() > deadline - safety_margin:
        pass

    # Restart 4: anti-diagonal (x+y constant) to potentially lower H_1
    w_anti = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            s = i + j
            w_anti[i, j] = 1.0 / (1.0 + s * s)
    log_w_anti = np.log(w_anti)
    lw, r = optimize_from_init(log_w_anti, max_iters=200, lr_init=0.05)
    if r > best_ratio:
        best_ratio = r
        best_log_w = lw

    # Final polish: run more iterations from best
    if best_log_w is not None and time.time() < deadline - safety_margin:
        lw, r = optimize_from_init(best_log_w, max_iters=500, lr_init=0.02)
        if r > best_ratio:
            best_ratio = r
            best_log_w = lw

    # Convert best log_w to P
    if best_log_w is None:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P = np.exp(best_log_w)
        P = P / P.sum()

    # Clip and normalize
    P = np.clip(P, 0, None)
    total = P.sum()
    if total <= 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P = P / total

    return P


# EVOLVE-BLOCK-END
