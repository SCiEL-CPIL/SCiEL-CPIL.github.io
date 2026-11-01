def construct():
    """Construct the packing configuration."""
    
    # Initial guess
    x0 = _initial_guess()
    
    # Bounds: radii in [0, 1], phases in [0, 2*pi]
    bounds = [(0.0, 1.0)] * num_rings + [(0.0, 2 * np.pi)] * num_rings
    
    # Use Nelder-Mead for robustness with non-smooth objective
    result = minimize(
        _objective, x0, args=(ring_counts,),
        method='Nelder-Mead',
        options={'maxiter': 8000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    )
    
    best_params = result.x
    radii = np.maximum(best_params[:num_rings], 0.0)
    phases = best_params[num_rings:]
    
    points = _build_points(radii, phases, ring_counts)
    
    # Verify it's valid
    if len(points) != N:
        # Fallback to hexagonal lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            pts = np.array(rows)
            radius = np.hypot(pts[:, 0], pts[:, 1])
            pts, radius = pts[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(pts) >= N:
                return pts[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    return points
