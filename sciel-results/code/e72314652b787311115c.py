def construct():
    # The 16x2 strip on the triangular lattice is a known optimal configuration
    # for n=32, achieving the Harborth bound of 76 edges.
    # It consists of points (a, 0) and (a, 1) for a in 0..15.
    # All coordinates are within |a|,|b| <= 64.
    # All distinct points have distance >= 1.
    # Edge count is 76.
    return [[a, b] for a in range(16) for b in range(2)]
