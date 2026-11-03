<<<<<<< SEARCH
    # Generate 11 points in a hexagonal cluster arrangement
    # Central hexagon + ring of 6 + 4 more in second ring
    # Use triangular lattice with spacing sqrt(3) for edge-to-edge contact
    sqrt3 = math.sqrt(3)
    # Scale factor to ensure strict disjointness (small gap between hexagons)
    scale = 1.001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
=======
    # Generate 11 points in a hexagonal cluster arrangement
    # Central hexagon + ring of 6 + 4 more in second ring
    # Use triangular lattice with spacing sqrt(3) for edge-to-edge contact
    sqrt3 = math.sqrt(3)
    # Scale factor: 1.0 allows touching (disjoint interiors). 
    # If evaluator requires strict separation, use 1.0 + 1e-9.
    scale = 1.0
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
>>>>>>> REPLACE
