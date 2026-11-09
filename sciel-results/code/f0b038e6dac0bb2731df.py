<<<<<<< SEARCH
    best_result = {
        'n': 26,
        'edges': [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    }
    
    # Try to find a larger graph
    # R(3, 10) <= 41, so max n is 40
    for n in range(40, 26, -1):
=======
    # Deterministic construction: Polarity graph of PG(2, 5)
    # n = 31
    points = []
    for y in range(5):
        for z in range(5):
            points.append((1, y, z))
    for z in range(5):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n_det = len(points)
    edges_det = []
    for i in range(n_det):
        x1, y1, z1 = points[i]
        for j in range(i + 1, n_det):
            x2, y2, z2 = points[j]
            dot = (x1 * x2 + y1 * y2 + z1 * z2) % 5
            if dot == 0:
                edges_det.append([i, j])
                
    best_result = {'n': n_det, 'edges': edges_det}
    
    # Try to find a larger graph
    # R(3, 10) <= 41, so max n is 40
    for n in range(40, n_det, -1):
>>>>>>> REPLACE
