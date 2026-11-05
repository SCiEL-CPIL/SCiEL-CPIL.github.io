def construct():
    # Generate candidate points in a central region
    # We want 30 distinct points.
    # A circle of radius ~3-4 in a grid has about 30 points.
    # Area of circle r=3 is ~28. r=3.5 is ~38.
    # So we take points with x^2 + y^2 <= R^2.
    
    candidates = []
    for x in range(-10, 11):
        for y in range(-10, 11):
            candidates.append((x, y))
            
    # Sort by distance from origin to get the "densest" central cluster
    candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the first 30
    points = candidates[:30]
    
    return points
