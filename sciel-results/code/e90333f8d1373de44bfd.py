    best_centers = None
    best_score = -np.inf

    # Run multiple optimizations from different jittered starts
    for seed in range(5):
        rng = np.random.default_rng(seed)
        # Re-initialize with different jitter
        points = []
        coords = np.linspace(-0.5, 0.5, 4)
        for x in coords:
            for y in coords:
                for z in coords:
                    points.append([x, y, z])
        points = np.array(points)
        indices = rng.choice(len(points), size=N, replace=False)
        centers_init = points[indices]
        jitter = rng.uniform(-0.05, 0.05, size=(N, 3))
        centers_init = centers_init + jitter
        centers_init = np.clip(centers_init, -0.49, 0.49)
        params_init = centers_init.flatten()
        
        try:
            result = minimize(objective, params_init, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
            centers_candidate = result.x.reshape(-1, 3)
            
            # Calculate actual radius score
            # Pairwise distances
            diff = centers_candidate[:, np.newaxis, :] - centers_candidate[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            
            # Boundary clearance
            boundary_clearance = np.min(0.5 - np.abs(centers_candidate))
            
            r = min(min_pair_dist / 2.0, boundary_clearance)
            
            if r > best_score:
                best_score = r
                best_centers = centers_candidate.copy()
        except Exception:
            continue
            
    if best_centers is None:
        # Fallback to initial grid
        return grid_init(N)
        
    return best_centers
