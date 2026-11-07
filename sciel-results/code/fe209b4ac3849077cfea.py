<<<<<<< SEARCH
    # Compute log of ratios (for log-space combination)
    # Clip to avoid log of very small values
    log_ratios = np.log(np.maximum(ratios, eps))
    
    # Geometric mean in log-space: mean of log ratios
    log_geo_mean = np.mean(log_ratios, axis=0)
    
    # Minimum in log-space: min of log ratios
    log_min = np.min(log_ratios, axis=0)
    
    # Blend in log-space: 0.65 weight on geometric mean, 0.35 on minimum
    w_geo = 0.65
    w_min = 0.35
    log_score = w_geo * log_geo_mean + w_min * log_min
=======
    # Compute log of ratios (for log-space combination)
    # Clip to avoid log of very small values
    log_ratios = np.log(np.maximum(ratios, eps))
    
    # Geometric mean in log-space: mean of log ratios
    log_geo_mean = np.mean(log_ratios, axis=0)
    
    # Soft-min in log-space: smooth approximation of min of log ratios
    # softmin(x) = -1/beta * log(sum(exp(-beta * x)))
    beta = 2.0
    X = -beta * log_ratios
    M = np.max(X, axis=0)
    S = np.sum(np.exp(X - M[:, np.newaxis]), axis=0)
    log_sum_exp = M + np.log(S)
    log_min_soft = -log_sum_exp / beta
    
    # Blend in log-space: 0.65 weight on geometric mean, 0.35 on soft-min
    w_geo = 0.65
    w_min = 0.35
    log_score = w_geo * log_geo_mean + w_min * log_min_soft
>>>>>>> REPLACE
