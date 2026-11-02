    sqrt3 = _SQRT3
    # Scale factor to ensure non-overlap for aligned hexagons
    # Min distance required is 2.0. Current lattice distance is sqrt(3).
    # Ratio: 2.0 / sqrt(3)
    scale = (2.0 / sqrt3) * 1.001
    u = (2.0 * scale, 0.0) # This makes u length 2*scale
    # Wait, if I change u to 2*scale, I should adjust v to maintain the hexagonal symmetry of the lattice
    # Standard basis for hex lattice with distance d: (d, 0) and (d/2, d*sqrt(3)/2)
    # Here d = 2.0 * scale
    d = 2.0 * scale
    u = (d, 0.0)
    v = (d/2.0, d*sqrt3/2.0)
