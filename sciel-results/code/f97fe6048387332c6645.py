def _build_optimal_pencil():
    """
    Constructs 12 points on 4 concurrent lines through the origin, 3 points per line.
    Uses symmetric directions to maximize 3-point cross-lines.
    """
    # Directions for the 4 lines
    dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
    pts = []
    for d in range(1, 4):  # 3 points per line at distances 1, 2, 3
        for (ux, uy) in dirs:
            pts.append([d * ux, d * uy])
    return pts
