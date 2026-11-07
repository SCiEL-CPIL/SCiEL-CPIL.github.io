def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    fb = float(f_best)
    sigma_safe = np.maximum(sigma, 1e-8)
    
    z = (fb - mu) / sigma_safe
    ei = (fb - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    if fb > 1.0:
        # Explore more
        beta = 2.0
        score = -mu + beta * sigma_safe
    else:
        # Exploit
        score = ei
        
    return np.where(np.isfinite(score), score, 0.0)
