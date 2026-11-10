import numpy as np

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    N = 64
    support = np.arange(N, dtype=np.float64)
    weights = np.ones(N, dtype=np.float64) / N
    
    # Precompute pairwise sums? No, need triple sums for Q and R.
    # Q_m = sum_{i,j,k} w_i w_j w_k I(s_i+s_j+s_k < 2s_m)
    # R_m = sum_{j,k,l} w_j w_k w_l I(s_m+s_j+s_k < 2s_l)
    
    # To speed up, we can use convolution or FFT?
    # N=64, N^4 = 16M ops per iteration. 500 iters = 8B ops. Too slow in Python.
    # Need vectorization.
    
    # s_i + s_j + s_k < 2 s_m
    # Let S3 = s_i + s_j + s_k.
    # We can compute the distribution of S = X1+X2+X3 via convolution.
    # But weights change each step.
    
    # Vectorized calculation:
    # Q_m: for each m, sum of w_i w_j w_k where s_i+s_j+s_k < 2 s_m.
    # This is a 4D sum.
    # We can compute P_S3[t] = Prob(X1+X2+X3 = t) using np.convolve 3 times.
    # Then Q_m = sum_{t < 2 s_m} P_S3[t].
    # This is O(N^2) for convolution (or N log N) and O(N) for prefix sum.
    
    # R_m: sum_{j,k,l} w_j w_k w_l I(s_m+s_j+s_k < 2 s_l).
    # Let Y = X2+X3. P_Y[u] = Prob(X2+X3=u).
    # Then for fixed m, we need sum_{l} w_l P( s_m + Y < 2 s_l ).
    # P(s_m + Y < 2 s_l) = sum_{u < 2 s_l - s_m} P_Y[u].
    # So R_m = sum_l w_l * CDF_Y[2 s_l - s_m - 1].
    
    # Algorithm per iteration:
    # 1. Compute P_Y = conv(w, w).
    # 2. Compute CDF_Y = cumsum(P_Y).
    # 3. Compute P_S3 = conv(P_Y, w).
    # 4. Compute CDF_S3 = cumsum(P_S3).
    # 5. For each m:
    #    Q_m = CDF_S3[2*s_m - 1] (clipped)
    #    R_m = sum_l w_l * CDF_Y[2*s_l - s_m - 1] (clipped)
    
    # This is O(N^2) per iteration. N=64 -> 4096 ops. Very fast.
    
    for _ in range(200):
        # Compute P_Y (sum of 2)
        P_Y = np.convolve(weights, weights, mode='full')
        # P_Y index k corresponds to sum = k
        # Max sum is 2*(N-1)
        
        CDF_Y = np.cumsum(P_Y)
        
        # Compute P_S3 (sum of 3)
        P_S3 = np.convolve(P_Y, weights, mode='full')
        CDF_S3 = np.cumsum(P_S3)
        
        grad = np.zeros(N)
        
        for m in range(N):
            s_m = support[m]
            
            # Q_m = P(S3 < 2 s_m)
            # S3 max is 3*(N-1)
            # We need sum of P_S3[t] for t < 2*s_m
            limit_q = int(2 * s_m)
            if limit_q > len(CDF_S3):
                q_val = CDF_S3[-1]
            elif limit_q <= 0:
                q_val = 0.0
            else:
                # CDF_S3[k] is sum up to k. We want sum up to limit_q - 1.
                idx_q = min(limit_q - 1, len(CDF_S3) - 1)
                q_val = CDF_S3[idx_q]
            
            # R_m = sum_l w_l * P(s_m + Y < 2 s_l)
            r_val = 0.0
            for l in range(N):
                s_l = support[l]
                # Condition: s_m + Y < 2 s_l  => Y < 2 s_l - s_m
                limit_r = int(2 * s_l - s_m)
                if limit_r > len(CDF_Y):
                    p_val = CDF_Y[-1]
                elif limit_r <= 0:
                    p_val = 0.0
                else:
                    idx_r = min(limit_r - 1, len(CDF_Y) - 1)
                    p_val = CDF_Y[idx_r]
                
                r_val += weights[l] * p_val
            
            grad[m] = q_val + 3 * r_val
        
        # Update weights
        weights += 0.01 * grad
        
        # Project to simplex
        # Simple projection: clip and normalize
        weights = np.clip(weights, 0, None)
        sum_w = np.sum(weights)
        if sum_w > 0:
            weights /= sum_w
        else:
            weights = np.ones(N) / N

    return support, weights
