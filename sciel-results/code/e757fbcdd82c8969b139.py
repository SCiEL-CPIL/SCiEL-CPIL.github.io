<<<<<<< SEARCH
    sqrt3 = np.sqrt(3.0)
    centers = []
    for q, r in axial_positions:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
=======
    sqrt3 = np.sqrt(3.0)
    # Scale the lattice to ensure disjointness (small gap between hexagons)
    # Distance between centers in standard grid is sqrt(3).
    # We scale positions by 1.01 to create a gap of ~1%.
    scale_factor = 1.01 
    centers = []
    for q, r in axial_positions:
        x = sqrt3 * (q + r / 2.0) * scale_factor
        y = 1.5 * r * scale_factor
        centers.append((x, y))
>>>>>>> REPLACE
