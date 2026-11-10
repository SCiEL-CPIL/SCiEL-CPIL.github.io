# EVOLVE-BLOCK-START
import time
import numpy as np

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Maximize P[X1+X2+X3 < 2*X4] for i.i.d. X ~ mu on [0, inf).
    
    Strategy: Use a two-point distribution (1-eps)*delta_0 + eps*delta_M,
    which achieves P = eps*(1-eps)^2*(1+2*eps).
    The optimal eps is ~0.4247, giving P ~ 0.2597.
    
    But we can do better with more points. Use a distribution with mass at 0
    and a few positive points, optimized via gradient ascent.
    """
    start = time.time()
    deadline = start + 620.0

    # Start with the two-point solution as initialization
    eps_opt = 0.4247
    n = 16
    support = np.zeros(n)
    support[0] = 0.0
    # Spread positive mass over a few points
    for i in range(1, n):
        support[i] = i * 2.0  # spaced out
    
    weights = np.zeros(n)
    weights[0] = 1.0 - eps_opt
    weights[1] = eps_opt  # put all positive mass at one point initially

    # Precompute pairwise sums
    S = support[:, None] + support[None, :]

    def compute_P(w):
        """Compute P = sum_{i,j,k,l} w_i w_j w_k w_l * 1[s_i+s_j < 2*s_l - s_k]"""
        P_val = 0.0
        for k in range(n):
            for l in range(n):
                mask = S < (2.0 * support[l] - support[k])
                T_kl = np.sum(w[:, None] * w[None, :] * mask)
                P_val += w[k] * w[l] * T_kl
        return P_val

    def compute_grad(w):
        """Compute gradient of P w.r.t. w using the quartic form."""
        # dP/dw_a = 4 * sum over all tuples where a appears
        grad = np.zeros(n)
        for a in range(n):
            # a = l: sum_{i,j,k} w_i w_j w_k w_a * 1[s_i+s_j < 2*s_a - s_k]
            t = 0.0
            for k in range(n):
                th = 2.0 * support[a] - support[k]
                mask = S < th
                t += w[k] * np.sum(w[:, None] * w[None, :] * mask)
            grad[a] += 4.0 * w[a] * t

            # a = k: sum_{i,j,l} w_i w_j w_a w_l * 1[s_i+s_j < 2*s_l - s_a]
            t = 0.0
            for l in range(n):
                th = 2.0 * support[l] - support[a]
                mask = S < th
                t += w[l] * np.sum(w[:, None] * w[None, :] * mask)
            grad[a] += 4.0 * w[a] * t

            # a = i: sum_{j,k,l} w_a w_j w_k w_l * 1[s_a+s_j < 2*s_l - s_k]
            t = 0.0
            for k in range(n):
                for l in range(n):
                    th = 2.0 * support[l] - support[k]
                    # Need s_a + s_j < th, i.e., s_j < th - s_a
                    if th > support[a]:
                        mask_j = support < (th - support[a])
                        t += w[k] * w[l] * np.sum(w[mask_j])
            grad[a] += 4.0 * w[a] * t

            # a = j: same as a=i by symmetry
            grad[a] += 4.0 * w[a] * t

        return grad

    # Gradient ascent with projection onto simplex
    best_w = weights.copy()
    best_P = compute_P(weights)
    lr = 0.1
    for step in range(500):
        if time.time() > deadline - 10:
            break
        g = compute_grad(weights)
        weights = weights + lr * g
        # Project onto simplex: w >= 0, sum(w) = 1
        weights = np.maximum(weights, 0.0)
        if np.sum(weights) > 1e-12:
            weights = weights / np.sum(weights)
        else:
            weights = best_w.copy()
            break
        P_val = compute_P(weights)
        if P_val > best_P:
            best_P = P_val
            best_w = weights.copy()
        # Adaptive learning rate
        if step % 50 == 49:
            lr *= 0.9

    if time.time() > deadline - 5:
        return support, best_w

    return support, best_w

# EVOLVE-BLOCK-END
