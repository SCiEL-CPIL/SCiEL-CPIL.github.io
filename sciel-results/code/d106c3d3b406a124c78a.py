<<<<<<< SEARCH
    for q, r in axial_coords:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        positions.append((x, y))
=======
    # Add a small scaling factor to ensure strict non-overlap due to floating point precision
    scale = 1.000001
    for q, r in axial_coords:
        x = math.sqrt(3) * scale * (q + r / 2.0)
        y = 1.5 * scale * r
        positions.append((x, y))
>>>>>>> REPLACE
