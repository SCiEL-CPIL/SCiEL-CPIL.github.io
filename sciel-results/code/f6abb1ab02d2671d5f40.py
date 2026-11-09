# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for given support and normalized weights."""
    n = len(support)
    # Pairwise sums S = X1+X2
    pair_sums = support[:, None] + support[None, :]  # (n, n)
    pair_weights = weights[:, None] * weights[None, :]  # (n, n)
    
    P = 0.0
    for k in range(n):
        if weights[k] == 0.0:
            continue
        # For fixed X4 = s_k and X3 = s_l, we need X1+X2 < 2*s_k - s_l
        for l in range(n):
            if weights[l] == 0.0:
                continue
            threshold = 2.0 * support[k] - support[l]
            mask = pair_sums < threshold
            inner_sum = np.sum(pair_weights * mask)
            P += weights[k] * weights[l] * inner_sum
    return P


def _optimize_weights_fixed_support(support, weights_init, max_iter=200, tol=1e-8):
    """Optimize weights for fixed support using gradient-free Nelder-Mead style approach."""
    n = len(support)
    w = weights_init.copy()
    w = np.maximum(w, 0.0)
    w = w / w.sum()
    
    best_P = _compute_P(support, w)
    
    # Simple coordinate ascent / hill climbing on weights
    step = 0.01
    for it in range(max_iter):
        improved = False
        for i in range(n):
            for delta in [step, -step]:
                w_trial = w.copy()
                w_trial[i] += delta
                if w_trial[i] < 0:
                    continue
                w_trial = np.maximum(w_trial, 0.0)
                w_sum = w_trial.sum()
                if w_sum <= 1e-12:
                    continue
                w_trial = w_trial / w_sum
                P_trial = _compute_P(support, w_trial)
                if P_trial > best_P + tol:
                    w = w_trial.copy()
                    best_P = P_trial
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
        if step < 1e-6:
            break
    
    return w, best_P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Use a small number of support points with carefully chosen positions and weights.
    Key insight: We want X4 to be large (so 2*X4 is large) while X1+X2+X3 is small.
    A good strategy is to put mass on 0 and on some large value M.
    
    For a two-point distribution on {0, M} with P(X=M)=p, P(X=0)=1-p:
    P[X1+X2+X3 < 2*X4] = sum over (x4, x1,x2,x3) where x1+x2+x3 < 2*x4
    
    Let's analyze:
    - If X4=0: need x1+x2+x3 < 0, impossible (all non-negative), contributes 0
    - If X4=M: need x1+x2+x3 < 2M
      Each xi is 0 or M. So x1+x2+x3 = M*(# of M's among x1,x2,x3)
      Need # of M's < 2, i.e., 0 or 1 M's.
      P[0 M's] = (1-p)^3
      P[1 M's] = 3*p*(1-p)^2
      Total for X4=M: p * [(1-p)^3 + 3*p*(1-p)^2] = p*(1-p)^2*(1-p+3p) = p*(1-p)^2*(1+2p)
    
    So P = p*(1-p)^2*(1+2p)
    Maximize f(p) = p*(1-p)^2*(1+2p) for p in [0,1].
    
    f(p) = p*(1-2p+p^2)*(1+2p) = p*(1+2p-2p-4p^2+p^2+2p^3) = p*(1-3p^2+2p^3) = p - 3p^3 + 2p^4
    
    f'(p) = 1 - 9p^2 + 8p^3
    Set to 0: 8p^3 - 9p^2 + 1 = 0
    (p-1)(8p^2 - p - 1) = 0
    8p^2 - p - 1 = 0 => p = (1+sqrt(1+32))/16 = (1+sqrt(33))/16 ≈ (1+5.7446)/16 ≈ 0.4215
    
    f(0.4215) ≈ 0.4215 - 3*(0.4215)^3 + 2*(0.4215)^4
    ≈ 0.4215 - 3*0.0749 + 2*0.0316
    ≈ 0.4215 - 0.2247 + 0.0632 ≈ 0.2599
    
    Let's verify with three points to see if we can do better.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill
    
    # Two-point baseline
    p_opt = (1 + np.sqrt(33)) / 16
    sup_2pt = np.array([0.0, 1.0], dtype=np.float64)
    w_2pt = np.array([1 - p_opt, p_opt], dtype=np.float64)
    P_2pt = _compute_P(sup_2pt, w_2pt / w_2pt.sum())
    
    best_support = sup_2pt.copy()
    best_weights = w_2pt.copy()
    best_P = P_2pt
    
    # Try three-point distributions: {0, a, 1} with weights (w0, wa, w1)
    if time.time() < deadline:
        for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            if time.time() > deadline:
                break
            sup_3pt = np.array([0.0, a, 1.0], dtype=np.float64)
            # Initialize weights
            w_init = np.array([0.3, 0.2, 0.5], dtype=np.float64)
            w_opt, P_opt = _optimize_weights_fixed_support(sup_3pt, w_init, max_iter=100)
            if P_opt > best_P:
                best_P = P_opt
                best_support = sup_3pt.copy()
                best_weights = w_opt.copy()
    
    # Try four-point distributions
    if time.time() < deadline:
        for a in [0.1, 0.25, 0.5]:
            for b in [0.5, 0.75, 0.9]:
                if time.time() > deadline:
                    break
                sup_4pt = np.array([0.0, a, b, 1.0], dtype=np.float64)
                w_init = np.array([0.25, 0.15, 0.15, 0.45], dtype=np.float64)
                w_opt, P_opt = _optimize_weights_fixed_support(sup_4pt, w_init, max_iter=80)
                if P_opt > best_P:
                    best_P = P_opt
                    best_support = sup_4pt.copy()
                    best_weights = w_opt.copy()
    
    # Refine the best support points
    if time.time() < deadline and len(best_support) >= 3:
        n = len(best_support)
        for _ in range(10):
            if time.time() > deadline:
                break
            sup_pert = best_support.copy()
            for i in range(1, n):
                factor = np.random.uniform(0.7, 1.3)
                sup_pert[i] *= factor
                sup_pert[i] = max(sup_pert[i], 1e-6)
            # Renormalize to [0, 1] range if needed
            sup_pert = sup_pert / sup_pert[-1]
            w_opt, P_opt = _optimize_weights_fixed_support(sup_pert, best_weights, max_iter=60)
            if P_opt > best_P:
                best_P = P_opt
                best_support = sup_pert.copy()
                best_weights = w_opt.copy()
    
    # Ensure validity
    best_support = np.maximum(best_support, 0.0).astype(np.float64)
    best_weights = np.maximum(best_weights, 0.0).astype(np.float64)
    w_sum = np.sum(best_weights)
    if w_sum <= 1e-9:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([1 - p_opt, p_opt], dtype=np.float64)
        w_sum = np.sum(best_weights)
    best_weights = best_weights / w_sum
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
