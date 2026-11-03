<<<<<<< SEARCH
    # Try several initial configurations
    configs = []
    
    # Config 1: Regular grid
    side = 5
    for i in range(side):
        for j in range(side):
            configs.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side])
    configs = configs[:n]
    centers1 = np.array([c[0] for c in configs])
    centers1 = np.column_stack([np.array([c[0] for c in configs]), np.array([c[1] for c in configs])])
    radii1 = np.array([c[2] for c in configs])
    try_configuration(centers1, radii1)
    
    # Config 2: Perturbed grid
    centers2 = centers1 + np.random.randn(n, 2) * 0.02
    centers2 = np.clip(centers2, 0.05, 0.95)
    radii2 = radii1.copy()
    try_configuration(centers2, radii2)
    
    # Config 3: Hexagonal-like packing
    rows = 5
    cols = 5
    configs3 = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) / cols + (0.5 / cols if i % 2 == 1 else 0)
            y = (i + 0.5) / rows
            if x < 1:
                configs3.append([x, y, 0.5 / cols])
    configs3 = configs3[:n]
    centers3 = np.column_stack([np.array([c[0] for c in configs3]), np.array([c[1] for c in configs3])])
    radii3 = np.array([c[2] for c in configs3])
    try_configuration(centers3, radii3)
    
    # Config 4: Random initialization
    centers4 = np.random.rand(n, 2) * 0.8 + 0.1
    radii4 = np.full(n, 0.05)
    try_configuration(centers4, radii4)
=======
    # Try several initial configurations
    
    def _ensure_n_points(centers, radii, n_target):
        """Ensure centers and radii have exactly n_target rows."""
        n_current = len(centers)
        if n_current == n_target:
            return centers, radii
        if n_current > n_target:
            return centers[:n_target], radii[:n_target]
        # Pad with small circles in random positions
        n_pad = n_target - n_current
        pad_centers = np.random.rand(n_pad, 2) * 0.8 + 0.1
        pad_radii = np.full(n_pad, 0.02)
        return np.vstack([centers, pad_centers]), np.concatenate([radii, pad_radii])

    # Config 1: Regular grid
    side = 5
    grid_configs = []
    for i in range(side):
        for j in range(side):
            grid_configs.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side])
    centers1 = np.column_stack([np.array([c[0] for c in grid_configs]), np.array([c[1] for c in grid_configs])])
    radii1 = np.array([c[2] for c in grid_configs])
    centers1, radii1 = _ensure_n_points(centers1, radii1, n)
    try_configuration(centers1, radii1)
    
    # Config 2: Perturbed grid
    centers2 = centers1 + np.random.randn(n, 2) * 0.02
    centers2 = np.clip(centers2, 0.05, 0.95)
    radii2 = radii1.copy()
    try_configuration(centers2, radii2)
    
    # Config 3: Hexagonal-like packing
    rows = 5
    cols = 5
    configs3 = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) / cols + (0.5 / cols if i % 2 == 1 else 0)
            y = (i + 0.5) / rows
            if x < 1:
                configs3.append([x, y, 0.5 / cols])
    centers3 = np.column_stack([np.array([c[0] for c in configs3]), np.array([c[1] for c in configs3])])
    radii3 = np.array([c[2] for c in configs3])
    centers3, radii3 = _ensure_n_points(centers3, radii3, n)
    try_configuration(centers3, radii3)
    
    # Config 4: Random initialization
    centers4 = np.random.rand(n, 2) * 0.8 + 0.1
    radii4 = np.full(n, 0.05)
    try_configuration(centers4, radii4)
>>>>>>> REPLACE
