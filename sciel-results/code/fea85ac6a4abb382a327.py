def _tight_honeycomb_343():
    # Lattice vectors
    u = np.array([1.5, np.sqrt(3)/2])
    v = np.array([0, np.sqrt(3)])
    
    # 3-4-3 arrangement in lattice coords (i, j)
    # Centered at 0,0
    # Visual Row 1 (top, 3 hexes): j=1, i=-1, 0, 1 ?
    # Let's check y-coords.
    # y = j*sqrt(3) + i*sqrt(3)/2
    # If j=1, i=-1: y = sqrt(3) - sqrt(3)/2 = sqrt(3)/2
    # If j=1, i=0: y = sqrt(3)
    # If j=1, i=1: y = sqrt(3) + sqrt(3)/2 = 3sqrt(3)/2
    # These are not same y.
    
    # We want visual rows to have same y.
    # Condition: 2j + i = const.
    # Let's pick points that form a 3-4-3 shape.
    # Points (i,j):
    # Row 1 (3 pts): i=-1, 1, 3? No.
    
    # Let's just take the first 12 points from a spiral in the lattice, 
    # or a specific compact set.
    
    # A known compact set for 12:
    # Center (0,0)
    # 6 neighbors: (1,0), (0,1), (-1,0), (0,-1), (1,-1), (-1,1)
    # 5 more from ring 2:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1) ? 
    # Let's check symmetry.
    # This set is not perfectly symmetric.
    
    # Alternative: 3-4-3 in "axial" coordinates.
    # Let's use the coordinates from the previous code but scaled down to touch.
    # Previous: dx=2, dy=1.5*sqrt(3).
    # Touching: dx=1.5, dy=sqrt(3)/2 for diagonal steps.
    # For a horizontal row of touching hexagons, they must share slanted edges.
    # So x_step = 1.5, y_step = sqrt(3)/2.
    # This is not a horizontal row.
    
    # To get a horizontal row, we need to alternate steps?
    # No, a "row" of hexagons in a honeycomb is a diagonal line in the grid.
    # If we rotate the entire packing by 30 degrees, the "diagonals" become horizontal/vertical.
    
    # So: Generate tight lattice points, then rotate the whole configuration by 30 degrees?
    # Or just generate them and let the optimizer align the outer hexagon.
    
    points = [
        (0,0),
        (1,0), (0,1), (-1,0), (0,-1), (1,-1), (-1,1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    
    inner = []
    for i,j in points:
        x = i * u[0] + j * v[0]
        y = i * u[1] + j * v[1]
        inner.append([x, y, 0])
        
    inner = np.array(inner)
    # Center it
    inner[:, 0] -= np.mean(inner[:, 0])
    inner[:, 1] -= np.mean(inner[:, 1])
    return inner
