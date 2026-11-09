# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_p(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly."""
    n = len(support)
    w = weights / weights.sum()
    # Build all pairwise sums
    s_sums = support[:, None] + support[None, :]  # n x n
    # For each (k, l), compute sum of w_i*w_j where s_i + s_j < 2*s_l - s_k
    # Vectorize: for each k, l, count pairs (i,j) with s_i+s_j < 2*s_l - s_k
    # P = sum_{k,l} w_k * w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i*w_j)
    
    # Create threshold matrix: thresholds[k,l] = 2*s_l - s_k
    thresholds = 2 * support[None, :] - support[:, None]  # k x l
    
    # For each threshold, compute the probability mass of pairs (i,j) with s_i+s_j < threshold
    # Use broadcasting: s_sums is n x n, thresholds is n x n (k x l)
    # We need: for each (k,l), sum of w_i*w_j where s_sums[i,j] < thresholds[k,l]
    
    # This is O(n^4) if done naively. Let's use sorting.
    # Sort s_sums and use cumulative sums of w_i*w_j
    
    # Actually, let's do it more efficiently:
    # Sort the support values
    order = np.argsort(support)
    s_sorted = support[order]
    w_sorted = w[order]
    
    # Compute all pairwise sums in sorted order
    # s_sorted[i] + s_sorted[j] for all i,j
    # We can use the fact that s_sorted is sorted to do binary search
    
    # For each threshold t, we need sum of w_i*w_j where s_i+s_j < t
    # Since s_sorted is sorted, for each i, the valid j are those with s_j < t - s_i
    # i.e., j < bisect_left(s_sorted, t - s_i)
    
    # Precompute cumulative weights
    cum_w = np.cumsum(w_sorted)
    cum_w = np.concatenate([[0], cum_w])
    
    n = len(s_sorted)
    
    # For each (k, l) pair, compute the inner sum
    # thresholds[k,l] = 2*s_l - s_k
    # We need sum over (k,l) of w_k * w_l * inner(k,l)
    
    # Let's compute inner for all (k,l) using vectorized approach
    # For each k, for each l: threshold = 2*s_l - s_k
    # inner = sum_i w_i * (cum_w[bisect_left(s_sorted, threshold - s_i)] - ... )
    
    # Actually simpler: for each threshold t, inner(t) = sum_{i,j: s_i+s_j < t} w_i*w_j
    # We can precompute this for all unique thresholds
    
    # Let's just do it with a loop over k and l since n is small (<= 32)
    p_total = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2 * s_sorted[l] - s_sorted[k]
            if threshold <= 0:
                continue
            # Count pairs (i,j) with s_i + s_j < threshold
            inner = 0.0
            for i in range(n):
                # j must satisfy s_j < threshold - s_i
                max_j = np.searchsorted(s_sorted, threshold - s_sorted[i], side='left')
                if max_j > 0:
                    inner += w_sorted[i] * cum_w[max_j]
            p_total += w_sorted[k] * w_sorted[l] * inner
    
    return p_total


def _neg_p(params, n_points, tail_power):
    """Negative log-likelihood objective for optimization."""
    # params: [log_w0, log_w1, ..., log_w_{n-2}, log_tail_weight, log_tail_scale]
    # Distribution: n_points-1 points near 0 with geometric spacing, plus one tail point
    # Actually, let's use a different parameterization
    
    # Support: [0, d, 2d, ..., (n-2)*d, S] where S is the tail point
    # Weights: w_i for i=0..n-2, and w_tail for the tail
    # Parameterize: w_i = exp(log_w_i) / sum, w_tail = exp(log_tail_w) / sum
    
    n = n_points
    d = 1.0  # spacing
    
    # First n-1 points: 0, 1, 2, ..., n-2
    # Last point: S (large)
    
    # params: [log_w_0, log_w_1, ..., log_w_{n-2}, log_S, log_w_tail]
    # That's (n-1) + 2 = n+1 parameters
    
    log_ws = params[:n-1]
    log_S = params[n-1]
    log_w_tail = params[n]
    
    S = np.exp(log_S)
    
    support = np.zeros(n)
    for i in range(n-1):
        support[i] = float(i)
    support[n-1] = S
    
    weights = np.exp(log_ws)
    weights = np.concatenate([weights, [np.exp(log_w_tail)]])
    weights = weights / weights.sum()
    
    p = _compute_p(support, weights)
    return -p


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1+X2+X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Approach: Use a distribution with several small points and one large tail point.
    The tail point, when drawn as X4, makes 2*X4 very large, so X1+X2+X3 < 2*X4
    is likely. Optimize the weights and tail location using derivative-free methods.
    """
    n_points = 8  # 7 small points + 1 tail
    
    # Initial guess: geometric weights on small points, small weight on tail
    # Small points: 0, 1, 2, 3, 4, 5, 6
    # Tail: S ~ 20
    # Weights: decay geometrically
    
    r = 0.7
    log_ws_init = np.array([np.log(r ** i) for i in range(n_points - 1)])
    log_S_init = np.log(20.0)
    log_w_tail_init = np.log(0.05)
    
    x0 = np.concatenate([log_ws_init, [log_S_init], [log_w_tail_init]])
    
    # Use Nelder-Mead for derivative-free optimization
    result = minimize(
        _neg_p,
        x0,
        args=(n_points, None),
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
    )
    
    x_opt = result.x
    log_ws = x_opt[:n_points-1]
    log_S = x_opt[n_points-1]
    log_w_tail = x_opt[n_points]
    
    S = np.exp(log_S)
    
    support = np.zeros(n_points)
    for i in range(n_points - 1):
        support[i] = float(i)
    support[n_points-1] = S
    
    weights = np.exp(log_ws)
    weights = np.concatenate([weights, [np.exp(log_w_tail)]])
    weights = weights / weights.sum()
    
    return support, weights


# EVOLVE-BLOCK-END
