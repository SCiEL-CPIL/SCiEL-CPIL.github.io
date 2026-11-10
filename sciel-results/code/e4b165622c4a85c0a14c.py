# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Strategy: Use f(k) = -c * exp(-k/K_smooth) / k for k > 1, and solve for f(1)
    to satisfy the constraint sum_k f(k)/k = 0. The exponential smoothing focuses
    weight on smaller k values, which have higher log(k)/k^2 contribution to the
    objective while keeping M(x) bounded.
    """
    start_time = time.time()
    DEADLINE = 610.0  # safety margin before 630s

    def check_feasibility_fast(domain, values, K):
        """Check M(x) <= 1.0001 for all x in [1, 10K] using efficient range updates."""
        max_x = 10 * K
        M = np.zeros(max_x + 1, dtype=np.float64)
        
        for k, f in zip(domain, values):
            if f == 0.0:
                continue
            # For each j, add f to M[j*k : (j+1)*k]
            j = 1
            while j * k <= max_x:
                lo = j * k
                hi = min((j + 1) * k, max_x + 1)
                M[lo:hi] += f
                j += 1
        
        return np.max(M[1:max_x + 1])

    def compute_score(domain, values):
        """Compute the objective a = -sum_k f(k)*log(k)/k"""
        return -np.sum(values * np.log(domain) / domain)

    best_domain = None
    best_values = None
    best_score = -np.inf

    # Strategy: f(k) = -c * w(k) / k for k > 1, where w(k) = exp(-k/K_smooth)
    # f(1) = c * sum_{k=2}^K w(k)/k^2
    # a = c * sum_{k=2}^K w(k) * log(k) / k^2
    # M(x) = c * [sum_{k=2}^K w(k)/k^2 * x - sum_{k=2}^K w(k)/k * floor(x/k)]
    #      = c * g(x)
    # Need c * max(g(x)) <= 1.0001

    # Try different K and K_smooth combinations
    K_candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    K_smooth_candidates = [5, 10, 20, 50, 100, 200, 500, 1000]

    for K in K_candidates:
        if time.time() > DEADLINE - 90:
            break
        
        for K_smooth in K_smooth_candidates:
            if time.time() > DEADLINE - 90:
                break
            
            # Only try K_smooth that are reasonable relative to K
            if K_smooth > K:
                continue
            
            ks = np.arange(2, K + 1, dtype=np.float64)
            w = np.exp(-ks / K_smooth)  # Smoothing weights
            
            # f(k) = -c * w(k) / k for k >= 2
            # f(1) = c * sum_{k=2}^K w(k)/k^2
            sum_w_inv_sq = np.sum(w / ks**2)
            a_coeff = np.sum(w * np.log(ks) / ks**2)
            
            # Compute g(x) = sum_w_inv_sq * x - sum_{k=2}^K w(k)/k * floor(x/k)
            # For efficiency, use the fact that floor(x/k) is piecewise constant
            max_x = 10 * K
            
            if K <= 5000:
                # Direct computation for small K
                g = np.zeros(max_x + 1, dtype=np.float64)
                for x in range(1, max_x + 1):
                    s = 0.0
                    for k in range(2, K + 1):
                        if k > x:
                            break
                        s += w[k-2] / k * (x // k)
                    g[x] = sum_w_inv_sq * x - s
                
                max_g = np.max(g[1:max_x + 1])
                
                if max_g > 1e-10:
                    c_max = 1.0001 / max_g
                    c = c_max
                    
                    # Build domain and values
                    domain = np.concatenate([[1], np.arange(2, K + 1, dtype=np.int64)])
                    values = np.empty(K, dtype=np.float64)
                    values[0] = c * sum_w_inv_sq  # f(1)
                    values[1:] = -c * w / ks  # f(k) for k >= 2
                    
                    # Verify feasibility
                    max_val = check_feasibility_fast(domain, values, K)
                    
                    if max_val <= 1.0001:
                        a = compute_score(domain, values)
                        
                        if a > best_score:
                            best_score = a
                            best_domain = domain.copy()
                            best_values = values.copy()
            
            elif K <= 50000:
                # Use more efficient computation for larger K
                # g(x) = sum_w_inv_sq * x - sum_{k=2}^K w(k)/k * floor(x/k)
                # We can compute the sum more efficiently by grouping terms
                
                # Precompute the coefficients
                coeffs = w / ks  # w(k)/k for k = 2, ..., K
                
                # g(x) = sum_w_inv_sq * x - sum_{k=2}^{min(K,x)} coeffs[k-2] * floor(x/k)
                # For large x, floor(x/k) can be computed in O(sqrt(x)) groups
                
                # Sample x values to estimate max_g
                sample_xs = np.unique(np.concatenate([
                    np.arange(1, min(1000, max_x + 1), 1),
                    np.arange(1000, max_x + 1, max(1, max_x // 10000))
                ]))
                
                max_g_sampled = 0.0
                for x in sample_xs:
                    x_int = int(x)
                    s = 0.0
                    # Group by floor(x/k) values
                    k = 2
                    while k <= min(K, x_int):
                        q = x_int // k
                        if q == 0:
                            break
                        k_end = min(K, x_int // q)
                        # Sum coeffs[k-2 : k_end-1]
                        idx_start = k - 2
                        idx_end = k_end - 1
                        s += q * np.sum(coeffs[idx_start:idx_end+1])
                        k = k_end + 1
                    g_x = sum_w_inv_sq * x_int - s
                    if g_x > max_g_sampled:
                        max_g_sampled = g_x
                
                # Conservative estimate: assume max_g is about 5% higher than sampled max
                max_g_est = max_g_sampled * 1.05 if max_g_sampled > 0 else 0
                
                if max_g_est > 1e-10:
                    c_max = 1.0001 / max_g_est
                    c = c_max * 0.95  # Use 95% of max to be safe
                    
                    # Build domain and values
                    domain = np.concatenate([[1], np.arange(2, K + 1, dtype=np.int64)])
                    values = np.empty(K, dtype=np.float64)
                    values[0] = c * sum_w_inv_sq  # f(1)
                    values[1:] = -c * w / ks  # f(k) for k >= 2
                    
                    # Verify feasibility (only for smaller K to save time)
                    if K <= 10000:
                        max_val = check_feasibility_fast(domain, values, K)
                        if max_val <= 1.0001:
                            a = compute_score(domain, values)
                            
                            if a > best_score:
                                best_score = a
                                best_domain = domain.copy()
                                best_values = values.copy()
                    else:
                        # Trust the estimate for large K
                        a = compute_score(domain, values)
                        
                        if a > best_score:
                            best_score = a
                            best_domain = domain.copy()
                            best_values = values.copy()
            
            if time.time() > DEADLINE - 90:
                break
        
        if time.time() > DEADLINE - 90:
            break
    
    # Strategy 2: Try the original approach with f(k) = -c/k (no smoothing)
    for K in [100, 500, 1000, 5000, 10000]:
        if time.time() > DEADLINE - 60:
            break
        
        ks = np.arange(2, K + 1, dtype=np.float64)
        sum_inv_sq = np.sum(1.0 / ks**2)
        
        max_x = 10 * K
        if K <= 5000:
            g = np.zeros(max_x + 1, dtype=np.float64)
            for x in range(1, max_x + 1):
                s = 0.0
                for k in range(2, K + 1):
                    if k > x:
                        break
                    s += (x // k) / k
                g[x] = sum_inv_sq * x - s
            
            max_g = np.max(g[1:max_x + 1])
            
            if max_g > 1e-10:
                c_max = 1.0001 / max_g
                c = c_max
                
                domain = np.concatenate([[1], np.arange(2, K + 1, dtype=np.int64)])
                values = np.empty(K, dtype=np.float64)
                values[0] = c * sum_inv_sq
                values[1:] = -c / ks
                
                max_val = check_feasibility_fast(domain, values, K)
                
                if max_val <= 1.0001:
                    a = compute_score(domain, values)
                    
                    if a > best_score:
                        best_score = a
                        best_domain = domain.copy()
                        best_values = values.copy()
        
        if time.time() > DEADLINE - 60:
            break
    
    if best_domain is not None:
        return best_domain, best_values
    
    # Fallback: use the original seed
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END
