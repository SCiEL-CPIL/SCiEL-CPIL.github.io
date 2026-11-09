def construct():
    edges = []
    # Component 1: K_{3,3} on vertices 0-5
    # Left partition: 0, 1, 2
    # Right partition: 3, 4, 5
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.append([u, v])
            
    # Component 2: K_{3,3} on vertices 6-11
    # Left partition: 6, 7, 8
    # Right partition: 9, 10, 11
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.append([u, v])
            
    return edges
