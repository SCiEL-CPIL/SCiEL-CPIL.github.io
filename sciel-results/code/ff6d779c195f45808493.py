"""Triangular lattice packing: maximize edges for 32 points."""

def construct():
    # A 5x6 block (30 points) plus 2 additional points yields 74 edges.
    # Base block: a in 0..5, b in 0..4
    points = [[a, b] for a in range(6) for b in range(5)]
    # Add (6, 2) and (6, 3) to maximize connections
    points.append([6, 2])
    points.append([6, 3])
    return points
