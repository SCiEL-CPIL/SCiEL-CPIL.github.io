def construct():
    # Build K_{3,3} on vertices 0..5 as one component.
    edges = []
    for u in range(3):
        for v in range(3, 6):
            edges.append([u, v])

    # Second K_{3,3} on vertices 6..11.
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([u, v])

    # Return as a fresh list of [int, int] pairs, no duplicates, no loops.
    return [list(e) for e in edges]
