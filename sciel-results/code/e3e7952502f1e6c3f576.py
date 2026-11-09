# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _build_idx():
    """Precompute integer indices for the four projections."""
    X, Y = np.meshgrid(np.arange(10), np.arange(10))  # X[y,x]=x, Y[y,x]=y
    idx_0 = X.astype(np.int64).ravel()                 # x
    idx_1 = (X + Y).astype(np.int64).ravel()           # x + y
    idx_2 = (X + 2 * Y).astype(np.int64).ravel()       # x + 2y
    idx_m1 = (X - Y + 9).astype(np.int64).ravel()      # x - y (shifted by 9 to be non-negative)
    return idx_0, idx_1, idx_2, idx_m1


def _softmax(z):
    """Numerically stable softmax."""
    z2 = z - np.max(z)
    e = np.exp(z2)
    return e / e.sum()


def _entropy_and_grad(q):
    """
    Compute Shannon entropy in bits and its gradient w.r.t. q.
    q is a 1D array of probabilities (may have zeros).
    Returns (H, grad_q).
    """
    mask = q > 1e-300
    H = -np.sum(q[mask] * np.log2(q[mask]))
    grad_q = np.zeros_like(q)
    # d/dq_i [-q_i log2(q_i)] = -(log2(q_i) + 1/ln(2))
    grad_q[mask] = -(np.log2(q[mask]) + 1.0 / np.log(2.0))
    return H, grad_q


def _make_objective(idx_0, idx_1, idx_2, idx_m1):
    """
    Build the objective function and its gradient for L-BFGS-B.
    We maximize D / M where:
      D = H_{-1} (entropy of x - y distribution)
      M = max(H_0, H_1, H_2)
    
    We minimize -D / M.
    """
    n0 = idx_0.max() + 1  # 10
    n1 = idx_1.max() + 1  # 19
    n2 = idx_2.max() + 1  # 27
    nm1 = idx_m1.max() + 1  # 19

    def objective(z):
        """
        z: 100-dimensional vector of log-probabilities (unnormalized logits).
        Returns (value, grad_z).
        """
        p = _softmax(z)  # 100 probabilities, p[y, x] flattened
        p_flat = p  # already flat

        # Compute projected distributions
        # H_0: distribution of x (10 bins)
        q0 = np.bincount(idx_0, weights=p_flat, minlength=n0)
        # H_1: distribution of x+y (19 bins)
        q1 = np.bincount(idx_1, weights=p_flat, minlength=n1)
        # H_2: distribution of x+2y (27 bins)
        q2 = np.bincount(idx_2, weights=p_flat, minlength=n2)
        # H_{-1}: distribution of x-y (19 bins)
        qm1 = np.bincount(idx_m1, weights=p_flat, minlength=nm1)

        H0, g0 = _entropy_and_grad(q0)
        H1, g1 = _entropy_and_grad(q1)
        H2, g2 = _entropy_and_grad(q2)
        Hm1, gm1 = _entropy_and_grad(qm1)

        D = Hm1
        M = max(H0, H1, H2)

        if M < 1e-12:
            return (0.0, np.zeros_like(z))

        ratio = D / M
        # We want to maximize ratio, so minimize -ratio
        value = -ratio

        # Compute gradient of -D/M w.r.t. z
        # d(ratio)/dz = (1/M) * dD/dz - (D/M^2) * dM/dz
        # dM/dz = d(H_max)/dz where H_max = max(H0, H1, H2)
        # By envelope theorem, if M = Hk, then dM/dz = dHk/dz

        # Determine which H is the max
        Hs = np.array([H0, H1, H2])
        k_max = np.argmax(Hs)
        gHs = [g0, g1, g2]
        gM = gHs[k_max]

        # dD/dz: chain rule through softmax
        # dHm1/dz = (dHm1/dq_m1) @ (dq_m1/dp) @ (dp/dz)
        # dq_m1/dp: for each cell i, it contributes to bin idx_m1[i]
        # So dHm1/dp_i = gm1[idx_m1[i]]
        dHm1_dp = gm1[idx_m1]  # shape (100,)
        dHm1_dz = _softmax_grad(p, dHm1_dp)

        # dM/dz
        dM_dz = _softmax_grad(p, gM[idx_0] if k_max == 0 else (gM[idx_1] if k_max == 1 else gM[idx_2]))

        grad_ratio = (1.0 / M) * dHm1_dz - (D / (M * M)) * dM_dz
        grad_z = -grad_ratio  # because we minimize -ratio

        return (value, grad_z)

    def _softmax_grad(p, dp):
        """
        Compute dp/dz @ dp where dp is dH/dp (shape 100).
        dp/dz = p * (I - p^T) in vector form:
        (dp/dz @ dp)_i = p_i * (dp_i - sum_j p_j * dp_j)
        """
        s = np.dot(p, dp)
        return p * (dp - s)

    return objective


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    
    Returns a 10x10 array P of finite weights, P[y, x] for integers x, y in 0..9.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin before 600s

    idx_0, idx_1, idx_2, idx_m1 = _build_idx()
    objective = _make_objective(idx_0, idx_1, idx_2, idx_m1)

    # Baseline: uniform distribution
    n = 10
    P_uniform = np.ones((n, n), dtype=np.float64) / (n * n)
    z_uniform = np.log(P_uniform.ravel())
    
    # Evaluate baseline
    val_base, _ = objective(z_uniform)
    best_val = val_base
    best_z = z_uniform.copy()

    # Multiple random restarts with L-BFGS-B
    np.random.seed(42)
    n_restarts = 30
    max_iter_per_restart = 200

    for i in range(n_restarts):
        if time.time() > deadline:
            break

        # Random initialization
        if i == 0:
            z0 = z_uniform.copy()
        else:
            # Perturb around best found so far, or random
            if i % 2 == 0:
                z0 = best_z + np.random.normal(0, 1.0, size=100)
            else:
                z0 = np.random.normal(0, 2.0, size=100)

        try:
            res = minimize(
                objective,
                z0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter_per_restart, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_z = res.x.copy()
        except Exception:
            pass

    # Convert best_z back to probability distribution
    P_best = _softmax(best_z).reshape((n, n))

    # Ensure validity: clip negative, normalize
    P_best = np.maximum(P_best, 0.0)
    total = P_best.sum()
    if total > 1e-9:
        P_best = P_best / total
    else:
        P_best = P_uniform

    return P_best


# EVOLVE-BLOCK-END
