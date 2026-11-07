def construct():
    # K_8,8 on vertices 0..15: partition A = {0..7}, B = {8..15}
    # Diameter exactly 2, every edge critical (removing any edge makes
    # some pair at distance >= 3 or disconnected).
    # 64 edges is optimal for this constraint set.
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    return edges
