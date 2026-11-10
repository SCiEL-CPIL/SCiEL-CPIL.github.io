# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy:
    1. Start with a fine grid of support points.
    2. Use projected gradient ascent on the log-weights (to enforce positivity and sum-to-1 via softmax-like normalization or direct projection).
    3. The objective P is a quartic form in weights. We can compute it efficiently using convolutions or matrix multiplications if we structure it carefully.
    
    Let p be the probability vector.
    P = sum_{k,l,i,j} p_k p_l p_i p_j * I(s_i + s_j < 2*s_l - s_k)
    
    Let's define the indicator matrix M such that M_{(i,j), (k,l)} = I(s_i + s_j < 2*s_l - s_k).
    Then P = sum_{i,j,k,l} p_i p_j p_k p_l M_{(i,j),(k,l)}.
    
    This is equivalent to:
    Let A be a matrix of size n x n where A_{i,j} = p_i p_j.
    Let B be a matrix of size n x n where B_{k,l} = p_k p_l.
    The condition is s_i + s_j < 2*s_l - s_k.
    
    We can rewrite the sum as:
    P = sum_{k,l} p_k p_l * ( sum_{i,j: s_i+s_j < 2*s_l-s_k} p_i p_j )
    
    Let F(x, y) = sum_{i,j: s_i+s_j < x} p_i p_j. This is the CDF of the sum of two iid variables from p, evaluated at x.
    Then P = sum_{k,l} p_k p_l * F(2*s_l - s_k).
    
    Since s_i are on a grid, we can precompute the distribution of S = X1+X2.
    Let q_m be the probability that X1+X2 = g_m, where g_m are the possible sums.
    Then F(x) = sum_{g_m < x} q_m.
    
    We can compute q via convolution of p with p.
    Then for each pair (k,l), we need the cumulative sum of q up to 2*s_l - s_k.
    
    To optimize p, we can use gradient ascent.
    The gradient of P with respect to p_a is:
    dP/dp_a = 4 * sum_{k,l,i,j: a in {k,l,i,j}} p_k p_l p_i p_j M_{(i,j),(k,l)}  (roughly, due to symmetries)
    
    Actually, let's derive it properly.
    P = sum_{k,l,i,j} p_k p_l p_i p_j M_{i,j,k,l}
    
    dP/dp_a = sum_{k,l,i,j} (delta_{k,a} p_l p_i p_j + p_k delta_{l,a} p_i p_j + p_k p_l delta_{i,a} p_j + p_k p_l p_i delta_{j,a}) M_{i,j,k,l}
    
    = sum_{l,i,j} p_l p_i p_j M_{i,j,a,l} + sum_{k,i,j} p_k p_i p_j M_{i,j,k,a} 
      + sum_{k,l,j} p_k p_l p_j M_{a,j,k,l} + sum_{k,l,i} p_k p_l p_i M_{i,a,k,l}
    
    Due to symmetry of M and p, these terms might be related.
    M_{i,j,k,l} = I(s_i+s_j < 2s_l - s_k).
    
    Term 1 (k=a): sum_{l,i,j} p_l p_i p_j I(s_i+s_j < 2s_l - s_a)
    Term 2 (l=a): sum_{k,i,j} p_k p_i p_j I(s_i+s_j < 2s_a - s_k)
    Term 3 (i=a): sum_{k,l,j} p_k p_l p_j I(s_a+s_j < 2s_l - s_k)
    Term 4 (j=a): sum_{k,l,i} p_k p_l p_i I(s_i+s_a < 2s_l - s_k)
    
    Note that Term 3 and 4 are similar.
    Term 3: sum_{k,l,j} p_k p_l p_j I(s_j < 2s_l - s_k - s_a)
    Term 4: sum_{k,l,i} p_k p_l p_i I(s_i < 2s_l - s_k - s_a)
    These are identical if we swap i and j.
    
    So Gradient_a = T1_a + T2_a + 2 * T3_a
    
    Where:
    T1_a = sum_{l} p_l * (sum_{i,j} p_i p_j I(s_i+s_j < 2s_l - s_a))
          = sum_{l} p_l * F(2s_l - s_a)
    
    T2_a = sum_{k} p_k * (sum_{i,j} p_i p_j I(s_i+s_j < 2s_a - s_k))
          = sum_{k} p_k * F(2s_a - s_k)
    
    T3_a = sum_{k,l} p_k p_l * (sum_{j} p_j I(s_j < 2s_l - s_k - s_a))
          = sum_{k,l} p_k p_l * F(2s_l - s_k - s_a)
    
    This allows for O(n^2) or O(n^3) gradient computation depending on how we handle F.
    If we precompute the CDF of the sum distribution (size ~2n), evaluating F(x) is O(1) (binary search) or O(n) if not sorted.
    Since s is sorted, 2s_l - s_a is sorted for fixed a? No, s_l is increasing, so 2s_l - s_a is increasing.
    So for fixed a, T1_a is a dot product of p_l with F(2s_l - s_a).
    We can compute F values for all l in O(n log n) or O(n) if we sweep.
    
    Let's implement this gradient ascent.
    
    Start with a uniform distribution on a grid.
    Grid: 0, 1, ..., n-1.
    n = 64 or 128.
    
    Constraint: p_i >= 0, sum p_i = 1.
    We can use multiplicative updates or projected gradient.
    Projected gradient: 
    p_new = p + lr * grad
    p_new = np.maximum(p_new, 0)
    p_new = p_new / np.sum(p_new)
    
    We need to be careful with the learning rate.
    """
    
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep margin
    
    # Parameters
    n = 64  # Number of support points
    support = np.arange(n, dtype=np.float64)
    
    # Initial weights: Uniform
    weights = np.ones(n, dtype=np.float64) / n
    
    # Precompute sum support for fast CDF evaluation?
    # The sums s_i + s_j range from 0 to 2*(n-1).
    # We can precompute the convolution of weights with weights to get the sum distribution.
    
    def compute_sum_distribution(w):
        # Convolution of w with w
        # Using FFT for speed
        N = 2 * n
        w_padded = np.zeros(N)
        w_padded[:n] = w
        W = np.fft.rfft(w_padded)
        sum_dist = np.fft.irfft(W * W, N)
        # The sum distribution is on grid 0, 1, ..., 2n-2
        # sum_dist[m] is prob of sum = m
        return sum_dist[:2*n-1]
    
    def compute_cdf(sum_dist):
        return np.cumsum(sum_dist)
    
    def evaluate_P(support, weights, sum_dist, cdf_sum):
        # P = sum_{k,l} p_k p_l F(2*s_l - s_k)
        # F(x) = cdf_sum[ floor(x) ] if x is integer, but x might not be integer?
        # Support is integer, so 2*s_l - s_k is integer.
        # If x < 0, F(x) = 0.
        # If x >= max_sum, F(x) = 1.
        
        # Vectorized computation
        # k, l are indices
        K, L = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        # Thresholds
        thresholds = 2.0 * support[L] - support[K]
        
        # Map thresholds to cdf indices
        # cdf_sum[i] is P(S <= i)
        # We need F(t) = P(S < t) = P(S <= t-1)
        # So index = t - 1
        
        indices = np.floor(thresholds).astype(int) - 1
        
        # Handle bounds
        # If index < 0, F = 0
        # If index >= len(cdf_sum), F = 1
        
        F_vals = np.zeros_like(thresholds, dtype=np.float64)
        
        # Valid range: 0 <= index < len(cdf_sum)
        mask_low = indices >= 0
        mask_high = indices < len(cdf_sum)
        mask_valid = mask_low & mask_high
        
        F_vals[mask_valid] = cdf_sum[indices[mask_valid]]
        F_vals[~mask_low] = 0.0
        F_vals[~mask_high] = 1.0
        
        # P = sum_{k,l} w_k w_l F_{k,l}
        # weights are 1D, need outer product
        # P = np.sum((weights[:, None] * weights[None, :]) * F_vals)
        # To save memory/time, we can do:
        # P = np.sum(weights[:, None] * (weights * F_vals))
        
        weighted_F = weights[None, :] * F_vals  # (n, n)
        P = np.sum(weights[:, None] * weighted_F)
        
        return P
    
    def compute_gradient(support, weights, sum_dist, cdf_sum):
        # grad_a = T1_a + T2_a + 2 * T3_a
        
        # T1_a = sum_l w_l F(2*s_l - s_a)
        # T2_a = sum_k w_k F(2*s_a - s_k)
        # T3_a = sum_{k,l} w_k w_l F(2*s_l - s_k - s_a)
        
        # Precompute F values for all possible arguments?
        # Arguments for T1: 2*s_l - s_a. Range: -s_a to 2*s_{n-1}-s_a.
        # Arguments for T2: 2*s_a - s_k. Range: 2*s_a - s_{n-1} to 2*s_a.
        # Arguments for T3: 2*s_l - s_k - s_a. Range: -s_k-s_a to 2*s_{n-1}-s_k-s_a.
        
        # It's easier to just loop over a and compute the sums.
        # n is small (64), so O(n^3) is fine (64^3 ~ 262k ops).
        
        grad = np.zeros(n, dtype=np.float64)
        
        # Precompute F for all pairs (k,l) for T3?
        # T3_a = sum_{k,l} w_k w_l F(2*s_l - s_k - s_a)
        # Let H_{k,l} = w_k w_l.
        # We need to sum H_{k,l} F(2*s_l - s_k - s_a) over k,l for each a.
        
        # Let's compute T1 and T2 first.
        # T1: For each a, sum_l w_l F(2*s_l - s_a)
        # T2: For each a, sum_k w_k F(2*s_a - s_k)
        
        # We can compute F(x) quickly if we have cdf_sum.
        
        # Let's create a helper to get F(x) for an array of x
        def get_F(x_array):
            # x_array is 1D array of floats (integers)
            idx = np.floor(x_array).astype(int) - 1
            F = np.zeros_like(x_array, dtype=np.float64)
            m_low = idx >= 0
            m_high = idx < len(cdf_sum)
            m_valid = m_low & m_high
            F[m_valid] = cdf_sum[idx[m_valid]]
            F[~m_low] = 0.0
            F[~m_high] = 1.0
            return F
        
        # T1
        # x = 2*s_l - s_a
        # Shape (n, n) where x[a, l] = 2*s_l - s_a
        x_T1 = 2.0 * support[None, :] - support[:, None]
        F_T1 = get_F(x_T1)
        T1 = np.sum(weights[None, :] * F_T1, axis=1)  # Sum over l
        
        # T2
        # x = 2*s_a - s_k
        # x[a, k] = 2*s_a - s_k
        x_T2 = 2.0 * support[:, None] - support[None, :]
        F_T2 = get_F(x_T2)
        T2 = np.sum(weights[None, :] * F_T2, axis=1)  # Sum over k
        
        # T3
        # x = 2*s_l - s_k - s_a
        # x[a, k, l] = 2*s_l - s_k - s_a
        # This is a 3D array of size n x n x n.
        # Memory: 64^3 * 8 bytes = 2MB. OK.
        # Computation:
        # T3[a] = sum_{k,l} w_k w_l F(x[a,k,l])
        
        # We can compute this efficiently.
        # Let's loop over a to save memory if needed, or just do it all at once.
        # Let's do it all at once for vectorization.
        
        # x_T3[a, k, l] = 2*s_l - s_k - s_a
        # s_a is shape (n,), s_k is (n,), s_l is (n,)
        # x_T3 = 2*s_l[None, None, :] - s_k[None, :, None] - s_a[:, None, None]
        
        s_a = support[:, None, None]
        s_k = support[None, :, None]
        s_l = support[None, None, :]
        
        x_T3 = 2.0 * s_l - s_k - s_a
        
        F_T3 = get_F(x_T3)
        
        # T3[a] = sum_{k,l} w_k w_l F_T3[a,k,l]
        # w_k w_l is outer product W = w[:, None] * w[None, :]
        W = weights[:, None] * weights[None, :]
        
        # T3 = np.sum(W[None, :, :] * F_T3, axis=(1, 2))
        
        grad = T1 + T2 + 2.0 * np.sum(W[None, :, :] * F_T3, axis=(1, 2))
        
        return grad
    
    best_P = -np.inf
    best_weights = weights.copy()
    
    # Initial evaluation
    sum_dist = compute_sum_distribution(weights)
    cdf_sum = compute_cdf(sum_dist)
    P = evaluate_P(support, weights, sum_dist, cdf_sum)
    if P > best_P:
        best_P = P
        best_weights = weights.copy()
    
    # Gradient ascent
    lr = 0.05  # Learning rate
    # Adaptive LR?
    
    for step in range(2000):
        if time.time() > deadline - 10:
            break
        
        grad = compute_gradient(support, weights, sum_dist, cdf_sum)
        
        # Update
        weights_new = weights + lr * grad
        
        # Project
        weights_new = np.maximum(weights_new, 0)
        sum_w = np.sum(weights_new)
        if sum_w < 1e-9:
            break
        weights_new = weights_new / sum_w
        
        # Evaluate
        sum_dist = compute_sum_distribution(weights_new)
        cdf_sum = compute_cdf(sum_dist)
        P_new = evaluate_P(support, weights_new, sum_dist, cdf_sum)
        
        if P_new > best_P:
            best_P = P_new
            best_weights = weights_new.copy()
        
        weights = weights_new
        
        # Decay learning rate
        if step % 100 == 0:
            lr *= 0.9
    
    # Ensure validity
    # If best_weights is invalid, return uniform
    if np.any(np.isnan(best_weights)) or np.any(np.isinf(best_weights)) or np.sum(best_weights) <= 1e-9:
        return support, np.ones(n) / n
    
    return support, best_weights

# EVOLVE-BLOCK-END
