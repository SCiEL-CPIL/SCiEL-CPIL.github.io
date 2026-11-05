<<<<<<< SEARCH
    # 2. Differential Evolution
    # We define the objective for DE.
    # DE works on continuous variables [0, 1].
    def de_objective(x):
        # x is a numpy array of floats
        mask = (x > 0.5).astype(np.int64)
        if not np.any(mask):
            return 10000
        counts = get_counts(mask)
        if np.any(counts == 0):
            return 10000 + np.sum(counts == 0)
        return float(np.max(counts))

    if time.time() < deadline - 30:
        try:
            bounds = [(0, 1) for _ in range(n + 1)]
            # Use 'best1bin' for binary-like problems
            res = differential_evolution(
                de_objective, 
                bounds, 
                strategy='best1bin',
                popsize=15,
                maxiter=30,
                tol=1e-4,
                seed=42,
                polish=False, # Disable polish to save time, we do our own local search
                init='latinhypercube'
            )
            if res.success or res.fun < 10000:
                best_de_mask = (res.x > 0.5).astype(np.int64)
                update_best(best_de_mask)
        except Exception:
            pass
=======
    # 2. Differential Evolution with soft-max surrogate
    # Use log-sum-exp to create a smooth surrogate for max(counts)
    def de_objective(x):
        # x is a numpy array of floats
        mask = (x > 0.5).astype(np.int64)
        if not np.any(mask):
            return 10000
        counts = get_counts(mask)
        if np.any(counts == 0):
            # Penalize uncovered sums heavily
            return 10000 + float(np.sum(counts == 0))
        # Soft max using log-sum-exp for smoothness
        # Use temperature to control smoothness
        temp = 0.5
        max_c = np.max(counts)
        # soft_max ≈ max_c + temp * log(sum(exp((c - max_c)/temp)))
        exp_counts = np.exp((counts.astype(float) - max_c) / temp)
        soft_max = max_c + temp * np.log(np.sum(exp_counts))
        return float(soft_max)

    if time.time() < deadline - 30:
        try:
            bounds = [(0, 1) for _ in range(n + 1)]
            # Use 'best1bin' for binary-like problems
            res = differential_evolution(
                de_objective, 
                bounds, 
                strategy='best1bin',
                popsize=20,
                maxiter=50,
                tol=1e-4,
                seed=42,
                polish=False, # Disable polish to save time, we do our own local search
                init='latinhypercube'
            )
            if res.success or res.fun < 10000:
                best_de_mask = (res.x > 0.5).astype(np.int64)
                update_best(best_de_mask)
        except Exception:
            pass
>>>>>>> REPLACE

