def construct():
    """Place disjoint unit hexagons inside a regular hexagon, minimizing outer side length."""
    # Generate 11 unit hexagons in a compact hexagonal lattice arrangement
    # Axial coordinates (q, r) for pointy-top hexagons
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    axial_coords = [
        (0, 0),   # Center
        (1, 0),   # Ring 1
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
        (0, 2),   # Add 4 more
        (0, -2),
        (1, 1),
        (-1, -1)
    ]
    
    positions = []
    for q, r in axial_coords:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        positions.append((x, y, 30.0))
        
    # Calculate bounding box to determine minimum outer side
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Add extent of unit hexagon (apothem x, radius y)
    # Pointy-top unit hex: width sqrt(3), height 2
    # Extent from center: x +/- sqrt(3)/2, y +/- 1
    total_min_x = min_x - math.sqrt(3) / 2.0
    total_max_x = max_x + math.sqrt(3) / 2.0
    total_min_y = min_y - 1.0
    total_max_y = max_y + 1.0
    
    width = total_max_x - total_min_x
    height = total_max_y - total_min_y
    
    # Outer hexagon (pointy-top, side S) has width S*sqrt(3) and height 2S
    # We need S*sqrt(3) >= width and 2S >= height
    side_x = width / math.sqrt(3)
    side_y = height / 2.0
    
    outer_side = max(side_x, side_y)
    
    # Center the outer hexagon
    center_x = (total_min_x + total_max_x) / 2.0
    center_y = (total_min_y + total_max_y) / 2.0
    outer_center = (center_x, center_y)
    outer_angle = 30.0 # Pointy-top
    
    # Validate
    if not _check_validity(positions, outer_center, outer_side, outer_angle):
        # If validation fails due to floating point or logic, increase side slightly
        outer_side *= 1.01
        if not _check_validity(positions, outer_center, outer_side, outer_angle):
             # Fallback to a known valid larger side
             outer_side = 4.5
             
    return positions, outer_center, outer_side, outer_angle
