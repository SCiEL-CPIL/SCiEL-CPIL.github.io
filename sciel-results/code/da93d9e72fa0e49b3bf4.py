    points = []
    # Type 1: x = 1
    for a in range(q):
        for b in range(q):
            points.append([1, a, b])
    # Type 2: x = 0, y = 1
    for b in range(q):
        points.append([0, 1, b])
    # Type 3: x = 0, y = 0, z = 1
    points.append([0, 0, 1])
