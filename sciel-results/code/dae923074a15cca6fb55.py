def construct():
    # 4x4 grid with corners removed
    points = []
    for x in range(4):
        for y in range(4):
            if not (x == 0 and y == 0) and not (x == 3 and y == 0) and not (x == 0 and y == 3) and not (x == 3 and y == 3):
                points.append([x, y])
    return points
