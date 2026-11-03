"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct the step heights h to minimize max overlap of h with shifted 1-h.
    
    Theoretical optimality proof:
    C(k) = sum_i h_i * (1 - h_{i+k})
    Sum over all shifts k: Sum_k C(k) = Sum_i h_i * (N - Sum_j h_j) = (N/2)*(N/2) = N^2/4
    Therefore average C(k) = N/4, so max_k C(k) >= N/4.
    
    Uniform h_i = 0.5 achieves C(k) = N * 0.5 * 0.5 = N/4 for every k,
    meeting the lower bound exactly.
    
    We start with the uniform solution and attempt a local refinement
    using projected gradient descent on a smooth surrogate (soft-max)
    of the max-over-shifts objective, with a deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill
    
    h = np.full(N, 0.5)
    
    def compute_max_overlap(h_vec):
        """Compute max over all cyclic shifts k of sum_i h_i * (1 - h_{i+k})."""
        max_c = 0.0
        for k in range(1, N):
            shifted = np.roll(h_vec, k)
            c = np.dot(h_vec, 1.0 - shifted)
            if c > max_c:
                max_c = c
        return max_c
    
    # Verify uniform solution
    uniform_score = compute_max_overlap(h)
    
    # Try projected gradient descent on smooth surrogate
    # Soft-max: log(sum(exp(beta * C_k))) / beta approximates max_k C_k
    # Gradient of C(k) w.r.t. h: d/dh_i C(k) = (1 - h_{i+k}) for term h_i*(1-h_{i+k})
    #                           and -h_{i-k} for the term where h_i appears in (1-h_{i-k})
    # So dC(k)/dh_i = (1 - h_{i+k}) - h_{i-k}
    
    best_h = h.copy()
    best_score = uniform_score
    
    beta = 20.0  # temperature for soft-max
    lr = 0.01
    iterations = 5000
    
    for it in range(iterations):
        if time.time() > deadline - 5.0:
            break
        
        # Compute all C(k) values
        c_vals = np.zeros(N - 1)
        for k in range(1, N):
            shifted = np.roll(h, k)
            c_vals[k - 1] = np.dot(h, 1.0 - shifted)
        
        # Soft-max surrogate
        # Use log-sum-exp for numerical stability
        max_c = np.max(c_vals)
        log_sum_exp = max_c + np.log(np.sum(np.exp(c_vals - max_c)))
        surrogate = log_sum_exp / beta
        
        # Gradient of surrogate w.r.t. h
        # d/dh_i [log(sum_k exp(beta*C_k))/beta]
        # = (1/beta) * sum_k exp(beta*C_k) * (beta * dC_k/dh_i) / sum_k exp(beta*C_k)
        # = sum_k [exp(beta*C_k) / sum_j exp(beta*C_j)] * dC_k/dh_i
        # = weighted average of gradients with softmax weights
        
        weights = np.exp(beta * (c_vals - max_c))
        weights /= np.sum(weights)
        
        grad = np.zeros(N)
        for k in range(1, N):
            # dC(k)/dh_i = (1 - h_{(i+k)%N}) - h_{(i-k)%N}
            idx_plus = np.arange(N) + k
            idx_plus = idx_plus % N
            idx_minus = np.arange(N) - k
            idx_minus = idx_minus % N
            grad += weights[k - 1] * ((1.0 - h[idx_plus]) - h[idx_minus])
        
        # Gradient step
        h = h - lr * grad
        
        # Project: clip to [0, 1] and adjust sum to be exactly N/2
        h = np.clip(h, 0.0, 1.0)
        # Adjust to fix the sum
        current_sum = np.sum(h)
        if current_sum > 0:
            # Distribute correction proportionally, but respect bounds
            correction = N / 2.0 - current_sum
            h = h + correction / N
            h = np.clip(h, 0.0, 1.0)
            # Second pass for exact sum
            residual = N / 2.0 - np.sum(h)
            if abs(residual) > 1e-15:
                # Add/subtract from elements with room
                if residual > 0:
                    mask = h < 1.0
                else:
                    mask = h > 0.0
                if np.any(mask):
                    h[mask] += residual / np.sum(mask)
                    h = np.clip(h, 0.0, 1.0)
        
        # Check score
        score = compute_max_overlap(h)
        if score < best_score - 1e-10:
            best_score = score
            best_h = h.copy()
        
        # If we've hit the theoretical lower bound, stop
        if best_score <= N / 4.0 + 1e-10:
            break
        
        # Adaptive learning rate
        if it % 500 == 0 and it > 0:
            lr *= 0.9
    
    # Ensure best solution is valid
    best_h = np.clip(best_h, 0.0, 1.0)
    current_sum = np.sum(best_h)
    if abs(current_sum - N / 2.0) > 1e-12:
        # Fall back to uniform if sum is off
        best_h = np.full(N, 0.5)
    
    assert np.all(best_h >= 0.0) and np.all(best_h <= 1.0)
    assert abs(np.sum(best_h) - N / 2.0) < 1e-12
    return best_h
# EVOLVE-BLOCK-END
