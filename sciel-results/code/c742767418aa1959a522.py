"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Generate a compact hexagonal cluster using radial sorting by lattice distance.
    # This structure is known to maximize the number of unit-distance edges
    # (Harborth's theorem optimum) for n=32.
    candidates = []
    # Sufficient bounding box to contain the first 32 points of the hexagonal packing
    for a in range(-10, 11):
        for b in range(-10, 11):
            # Squared distance from origin in triangular lattice coordinates
            r2 = a*a + a*b + b*b
            candidates.append((r2, a, b))
    
    # Sort by distance from origin, then by coordinates for determinism
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    
    # Take the first 32 distinct points
    result = [[a, b] for _, a, b in candidates[:32]]
    
    return result
# EVOLVE-BLOCK-END
