    ranks_to_try = [44, 50, 60, 63]
    seeds = [42, 1, 2, 3]
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        for seed in seeds:
            if time.time() > deadline:
                break
            np.random.seed(seed)
            x0 = np.random.randn(n * m * R + m * p * R + p * n * R) * 0.5
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=2000)
                if result.cost < 1e-10:
                    # ... return
            except:
                pass
